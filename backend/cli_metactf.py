"""CLI — fetch MetaCTF Compete problems, solve in parallel (tiered models), submit flags."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import click
import httpx
from rich.console import Console

from backend.agents.metactf_swarm import MetaCTFChallengeSwarm
from backend.config import Settings
from backend.cost_tracker import CostTracker
from backend.metactf import (
    USER_AGENT,
    cookie_header_for_metactf,
    fetch_problems_json,
    model_specs_for_points,
    normalize_metactf_base_url,
    problem_to_challenge_files,
    select_problems,
    slug_challenge_dir,
    solved_ids_from_payload,
)
from backend.models import DEFAULT_MODELS
from backend.prompts import ChallengeMeta
from backend.sandbox import cleanup_orphan_containers, configure_semaphore
from backend.solver_base import FLAG_FOUND

console = Console()
logger = logging.getLogger(__name__)

QWEN_SPEC = "openrouter/qwen/qwen3.6-plus:free"


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%X"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


@click.command()
@click.argument("contest_url")
@click.option(
    "--cookie",
    default="",
    envvar="METACTF_COOKIE",
    help="Browser cookie string (METACTF_COMPETE=...). MCS_OPTIONS= is stripped automatically.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Maximum challenges to run after sorting (omit or 0 = all unsolved).",
)
@click.option(
    "--skip",
    default="",
    help="Challenge titles to skip, separated by semicolons (e.g. 'Dead Drop;License To Rev').",
)
@click.option("--no-submit", is_flag=True, help="Dry run - do not POST flags to MetaCTF.")
@click.option("-v", "--verbose", is_flag=True)
def main(
    contest_url: str,
    cookie: str,
    limit: int | None,
    skip: str,
    no_submit: bool,
    verbose: bool,
) -> None:
    """Solve MetaCTF Compete problems using tiered models and API submission.

    CONTEST_URL may be compete.metactf.com/576 or full https URL.

    Points tiers: <=150 Qwen only; 151-200 three OpenRouter models; >200 three + Gemini rotate.

    Requires OPENROUTER_API_KEY; for >200 pt challenges, GEMINI_API_KEY must be set.
    """
    _setup_logging(verbose)

    raw_cookie = (cookie or "").strip()
    if not raw_cookie:
        console.print("[red]Pass --cookie or set METACTF_COOKIE in the environment.[/red]")
        sys.exit(1)

    cookie_clean = cookie_header_for_metactf(raw_cookie)

    try:
        base = normalize_metactf_base_url(contest_url)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    settings = Settings()
    openrouter_keys = settings.get_openrouter_keys()
    gemini_keys = settings.get_gemini_keys()

    if not openrouter_keys:
        console.print("[red]Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS.[/red]")
        sys.exit(1)

    skip_titles = {t.strip() for t in skip.split(";") if t.strip()}
    eff_limit = None if limit is None or limit <= 0 else limit

    try:
        asyncio.run(
            _run_metactf(
                settings=settings,
                base_url=base,
                cookie=cookie_clean,
                eff_limit=eff_limit,
                skip_titles=skip_titles,
                no_submit=no_submit,
                openrouter_keys=openrouter_keys,
                gemini_keys=gemini_keys,
            )
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


async def _run_metactf(
    *,
    settings: Settings,
    base_url: str,
    cookie: str,
    eff_limit: int | None,
    skip_titles: set[str],
    no_submit: bool,
    openrouter_keys: list[str],
    gemini_keys: list[str],
) -> None:
    console.print("[bold]MetaCTF agent[/bold]")
    console.print(f"  Contest: {base_url}")
    console.print(f"  Skip: {skip_titles or '(none)'}")
    console.print(f"  Limit: {eff_limit or 'all unsolved'}")
    console.print(f"  OpenRouter keys: {len(openrouter_keys)}")
    console.print(f"  Gemini keys: {len(gemini_keys)}")
    console.print(f"  Submit: {'no (dry)' if no_submit else 'yes'}")
    console.print(f"  Parallel challenges (max): {settings.max_concurrent_challenges}")
    console.print()

    work_parent = Path(tempfile.mkdtemp(prefix="metactf_agent_"))
    total_cost = 0.0

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            payload = await fetch_problems_json(client, base_url, cookie)
            solved_ids = solved_ids_from_payload(payload)
            selected = select_problems(payload, limit=eff_limit, skip_titles=skip_titles)

            console.print(
                f"  API `solved` ids excluded: {len(solved_ids)} — "
                f"to run: {len(selected)} unsolved challenge(s)"
            )
            console.print()

            if not selected:
                console.print("[yellow]No unsolved challenges matched your filters.[/yellow]")
                return

            needs_gemini = any(int(p.get("points") or 0) > 200 for p in selected)
            if needs_gemini and not gemini_keys:
                console.print("[red]Challenges over 200 points need GEMINI_API_KEY / GEMINI_API_KEYS.[/red]")
                sys.exit(1)

            max_models = 1
            for p in selected:
                pts = int(p.get("points") or 0)
                n = len(model_specs_for_points(pts, default_three=list(DEFAULT_MODELS), qwen_spec=QWEN_SPEC))
                max_models = max(max_models, n)
            configure_semaphore(settings.max_concurrent_challenges * max_models)

            await cleanup_orphan_containers()

            sem = asyncio.Semaphore(settings.max_concurrent_challenges)
            # Swarms still running (for poll: stop if API marks id solved elsewhere)
            running_swarms: dict[int, MetaCTFChallengeSwarm] = {}
            poll_shutdown = asyncio.Event()

            async def poll_problems_json_loop() -> None:
                """Every ~10s, refetch problems; if a running id appears in solved[], stop that swarm only."""
                while not poll_shutdown.is_set():
                    try:
                        await asyncio.wait_for(poll_shutdown.wait(), timeout=10.0)
                        return
                    except TimeoutError:
                        pass
                    try:
                        data = await fetch_problems_json(client, base_url, cookie)
                        solved_now = solved_ids_from_payload(data)
                        for pid in list(running_swarms.keys()):
                            if pid in solved_now:
                                sw = running_swarms.get(pid)
                                if sw is not None:
                                    logger.info(
                                        "MetaCTF poll: id=%s now in solved[] — stopping this swarm only",
                                        pid,
                                    )
                                    console.print(
                                        f"[yellow]Poll: id={pid} now solved on server — "
                                        f"stopping its containers (other challenges keep running)[/yellow]"
                                    )
                                    sw.kill()
                    except Exception as e:
                        logger.warning("MetaCTF problems_json poll: %s", e)

            async def run_challenge(index: int, prob: dict) -> tuple[float, object | None, str, int]:
                async with sem:
                    title = str(prob.get("title") or "?")
                    pid = int(prob.get("id") or 0)
                    pts = int(prob.get("points") or 0)
                    specs = model_specs_for_points(
                        pts, default_three=list(DEFAULT_MODELS), qwen_spec=QWEN_SPEC
                    )

                    if pts > 200:
                        st = settings.model_copy(
                            update={"gemini_rotate_chain": "gemini-3-flash-preview,gemini-2.5-flash"}
                        )
                    else:
                        st = settings.model_copy(update={"gemini_rotate_chain": ""})

                    slug = slug_challenge_dir(title)
                    ch_dir = str(work_parent / f"id{pid}_{slug}")
                    problem_to_challenge_files(prob, ch_dir)
                    meta = ChallengeMeta.from_directory(ch_dir)

                    console.print(
                        f"[bold][{index}/{len(selected)}][/bold] start {title!r} "
                        f"(id={pid}, {pts} pts, {len(specs)} model lane(s))"
                    )

                    cost_tracker = CostTracker()
                    swarm = MetaCTFChallengeSwarm(
                        challenge_dir=ch_dir,
                        meta=meta,
                        cost_tracker=cost_tracker,
                        settings=st,
                        model_specs=specs,
                        no_submit=no_submit,
                        metactf_base_url=base_url,
                        metactf_cookie=cookie,
                        metactf_problem_id=pid,
                        metactf_http=client,
                    )

                    running_swarms[pid] = swarm
                    try:
                        result = await swarm.run()
                    finally:
                        running_swarms.pop(pid, None)

                    status = getattr(result, "status", None) if result else None
                    console.print(
                        f"[dim]Challenge task done: id={pid} {title!r} status={status}[/dim]"
                    )
                    return cost_tracker.total_cost_usd, result, title, pid

            tasks = [
                asyncio.create_task(run_challenge(i, prob), name=f"metactf-{prob.get('id')}")
                for i, prob in enumerate(selected, 1)
            ]
            poll_task = asyncio.create_task(poll_problems_json_loop(), name="metactf-poll-problems")

            try:
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            except (KeyboardInterrupt, asyncio.CancelledError):
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                console.print("\n[yellow]Interrupted — cancelling challenges.[/yellow]")
                raise
            finally:
                poll_shutdown.set()
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await poll_task

            for item in outcomes:
                if isinstance(item, Exception):
                    console.print(f"[red]Challenge task failed: {item}[/red]")
                    continue
                cst, result, title, _pid = item
                total_cost += cst
                if result and getattr(result, "status", None) == FLAG_FOUND:
                    console.print(f"  [green]Solved[/green] {title!r} — {getattr(result, 'flag', None)}")
                else:
                    console.print(f"  [yellow]Not solved[/yellow] {title!r}")

        console.print(f"\n[bold]Total API cost (estimated):[/bold] ${total_cost:.4f}")
    finally:
        shutil.rmtree(work_parent, ignore_errors=True)


if __name__ == "__main__":
    main()
