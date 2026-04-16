"""CLI — fetch MetaCTF Compete problems, solve in parallel (tiered models), submit flags."""

from __future__ import annotations

import asyncio
import contextlib
import json
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
    is_instance_based_remote_challenge,
    model_specs_for_points,
    normalize_metactf_base_url,
    problem_to_challenge_files,
    select_problems,
    slug_challenge_dir,
    solved_ids_from_payload,
)
from backend.models import DEFAULT_MODELS, openrouter_spec_from_user_id
from backend.prompts import ChallengeMeta
from backend.sandbox import cleanup_orphan_containers, configure_semaphore
from backend.solver_base import FLAG_FOUND

console = Console()
logger = logging.getLogger(__name__)

KIMI_NVIDIA_SPEC = "nvidia/moonshotai/kimi-k2.5"
GLM_NVIDIA_SPEC = "nvidia/z-ai/glm5"
METACTF_PAY_MODELS = [
    "openrouter/qwen/qwen3.6-plus",
    "nvidia/moonshotai/kimi-k2.5",
    "openrouter/google/gemma-4-26b-a4b-it",
    "openrouter/openai/gpt-oss-120b",
    "openrouter/openai/gpt-oss-120b:free",
    "openrouter/nemotron-3-super-120b-a12b:free",
]
# MetaCTF: never run more than this many challenges at once (batches run one after another).
METACTF_PARALLEL_CHALLENGES = 3


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
@click.option(
    "--custom",
    "custom_openrouter",
    default=None,
    help="OpenRouter model id for the first slot/lane of multi-model tiers (e.g. z-ai/glm-4.5-air:free).",
)
@click.option(
    "--pay",
    is_flag=True,
    help="Paid mode: use only the first configured OpenRouter API key and run Qwen, Kimi, and Gemma in parallel.",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    contest_url: str,
    cookie: str,
    limit: int | None,
    skip: str,
    no_submit: bool,
    custom_openrouter: str | None,
    pay: bool,
    verbose: bool,
) -> None:
    """Solve MetaCTF Compete problems using tiered models and API submission.

    CONTEST_URL may be compete.metactf.com/576 or full https URL.

    Points tiers: <=150 three OpenRouter + Kimi(NVIDIA); >150 adds GLM(NVIDIA) + Gemini rotate.

    With --pay: uses configured pay models (Qwen, Kimi, Gemma) simultaneously on every challenge.

    Requires OPENROUTER_API_KEY; for >150 pt challenges, GEMINI_API_KEY must be set (unless --pay).
    For <=150 pt challenges, GEMINI_API_KEY is optional: without it, no Gemini lane is added after 60s.
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
    nvidia_keys = settings.get_nvidia_keys()

    if not openrouter_keys:
        console.print("[red]Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS.[/red]")
        sys.exit(1)

    skip_titles = {t.strip() for t in skip.split(";") if t.strip()}
    eff_limit = None if limit is None or limit <= 0 else limit

    custom_spec: str | None = None
    if custom_openrouter and not pay:
        try:
            custom_spec = openrouter_spec_from_user_id(custom_openrouter)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    elif custom_openrouter and pay:
        console.print("[yellow]--pay ignores --custom[/yellow]")

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
                nvidia_keys=nvidia_keys,
                custom_openrouter_spec=custom_spec,
                pay=pay,
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
    nvidia_keys: list[str],
    custom_openrouter_spec: str | None = None,
    pay: bool = False,
) -> None:
    runner_settings = (
        settings.model_copy(update={"openrouter_use_first_key_only": True}) if pay else settings
    )

    tier_default_three = (
        [custom_openrouter_spec] + list(DEFAULT_MODELS[1:])
        if custom_openrouter_spec and not pay
        else list(DEFAULT_MODELS)
    )

    console.print("[bold]MetaCTF agent[/bold]")
    console.print(f"  Contest: {base_url}")
    console.print(f"  Skip: {skip_titles or '(none)'}")
    console.print(f"  Limit: {eff_limit or 'all unsolved'}")
    console.print(f"  OpenRouter keys: {len(openrouter_keys)}")
    console.print(f"  NVIDIA keys: {len(nvidia_keys)}")
    console.print(f"  Gemini keys: {len(gemini_keys)}")
    console.print(f"  Submit: {'no (dry)' if no_submit else 'yes'}")
    console.print(f"  Parallel challenges (max): {METACTF_PARALLEL_CHALLENGES} (batched)")
    if pay:
        console.print(
            f"  Pay mode: first OpenRouter key only, models {', '.join(METACTF_PAY_MODELS)}"
        )
    if custom_openrouter_spec:
        console.print(f"  Custom OpenRouter (first lane): {custom_openrouter_spec}")
    console.print(
        "[dim]Press [bold]Enter[/bold] at any time to open the operator menu (skip/add models/view trace).[/dim]\n"
    )

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

            if not nvidia_keys:
                if not pay or any(m.startswith("nvidia/") for m in METACTF_PAY_MODELS):
                    console.print("[red]Set NVIDIA_API_KEY or NVIDIA_API_KEYS (needed for Nvidia lanes).[/red]")
                    sys.exit(1)

            if not pay:
                needs_gemini = any(int(p.get("points") or 0) > 150 for p in selected)
                if needs_gemini and not gemini_keys:
                    console.print(
                        "[red]Challenges over 150 points need GEMINI_API_KEY / GEMINI_API_KEYS.[/red]"
                    )
                    sys.exit(1)

            max_models = 1
            if pay:
                max_models = len(METACTF_PAY_MODELS)
            else:
                for p in selected:
                    pts = int(p.get("points") or 0)
                    n = len(
                        model_specs_for_points(
                            pts,
                            default_three=tier_default_three,
                            kimi_nvidia_spec=KIMI_NVIDIA_SPEC,
                            glm_nvidia_spec=GLM_NVIDIA_SPEC,
                        )
                    )
                    max_models = max(max_models, n)
                if any(int(p.get("points") or 0) <= 150 for p in selected) and gemini_keys:
                    # <=150 tiers can add Gemini rotate after 60s if still unsolved.
                    max_models += 1
            configure_semaphore(METACTF_PARALLEL_CHALLENGES * max_models)

            await cleanup_orphan_containers()

            parallel_lock = asyncio.Lock()
            active_parallel: list[tuple[int, str, MetaCTFChallengeSwarm]] = []
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
                        err_str = str(e).lower()
                        if "name resolution" in err_str or "[errno -3]" in err_str:
                            logger.debug("MetaCTF problems_json poll: DNS error (ignored)")
                        else:
                            logger.warning("MetaCTF problems_json poll: %s", e)

            async def stdin_menu_loop() -> None:
                """Listen for Enter (\n) or 'menu' to trigger the operator actions menu."""
                loop = asyncio.get_running_loop()
                while True:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        break
                    
                    text = line.strip().lower()
                    if text not in ("", "menu", "skip", "add", "trace"):
                        continue
                        
                    async with parallel_lock:
                        pairs = list(active_parallel)
                    if not pairs:
                        console.print(
                            "[yellow]No parallel challenges running currently — wait for a batch to start.[/yellow]"
                        )
                        continue

                    def _prompt_menu() -> tuple[str, int, str | None]:
                        console.print("\n[bold]Operator Menu[/bold]")
                        console.print("  1. Skip a challenge")
                        console.print("  2. Add Gemini 3.1 Pro Preview (OpenRouter) to a challenge")
                        console.print("  3. View model trace/messages")
                        console.print("  4. Cancel")
                        action_num = click.prompt("Choose an action (1-4)", type=click.IntRange(1, 4))
                        
                        if action_num == 4:
                            return ("cancel", -1, None)
                        
                        console.print("\n[bold]Active Challenges:[/bold]")
                        for i, (pid, title, _) in enumerate(pairs, 1):
                            console.print(f"  {i}. id={pid} {title!r}")
                        
                        prompt_text = {
                            1: "Skip which challenge",
                            2: "Add Gemini Pro to which challenge",
                            3: "View trace for which challenge"
                        }[action_num]
                        
                        choice = click.prompt(
                            f"{prompt_text} (1–{len(pairs)})?",
                            type=click.IntRange(1, len(pairs)),
                        )
                        
                        if action_num == 3:
                            _, _, swarm = pairs[choice - 1]
                            active_models = list(swarm._started_specs)
                            if not active_models:
                                console.print("[yellow]No active models for this challenge yet.[/yellow]")
                                return ("cancel", -1, None)
                            
                            console.print("\n[bold]Active Models:[/bold]")
                            for i, mod in enumerate(active_models, 1):
                                console.print(f"  {i}. {mod}")
                            mod_choice = click.prompt(f"View which model (1-{len(active_models)})?", type=click.IntRange(1, len(active_models)))
                            return ("view_trace", choice - 1, active_models[mod_choice - 1])

                        return ("skip" if action_num == 1 else "add_gemini", choice - 1, None)

                    action, choice_idx, model_spec = await asyncio.to_thread(_prompt_menu)
                    
                    if action == "cancel":
                        console.print("[dim]Menu cancelled.[/dim]")
                        continue
                        
                    _, _, swarm = pairs[choice_idx]
                    
                    if action == "skip":
                        swarm.kill()
                        console.print("[yellow]Skip requested — stopping that challenge.[/yellow]")
                    elif action == "add_gemini":
                        gemini_spec = "openrouter/google/gemini-3.1-pro-preview"
                        swarm.launch_solver(gemini_spec)
                        console.print(f"[green]Added {gemini_spec} to challenge.[/green]")
                    elif action == "view_trace" and model_spec:
                        solver = swarm.solvers.get(model_spec)
                        if not solver or not getattr(solver, "tracer", None):
                            console.print("[red]No active solver or tracer found for that model.[/red]")
                            continue
                        try:
                            path = solver.tracer.path if hasattr(solver.tracer, "path") else str(solver.tracer)
                            lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
                            recent = lines[-30:]
                            console.print(f"\n[bold]Trace for {model_spec} (last 30 events)[/bold]")
                            for line in recent:
                                if not line.strip(): continue
                                try:
                                    d = json.loads(line)
                                    t = d.get("type", "?")
                                    if t == "tool_call":
                                        args_str = str(d.get("args", ""))[:80].replace("\n", " ")
                                        console.print(f"  🛠️  [cyan]{d.get('tool')}[/cyan] {args_str}")
                                    elif t == "tool_result":
                                        res_str = str(d.get("result", ""))[:100].replace("\n", " ")
                                        console.print(f"  📄 [dim]{res_str}[/dim]")
                                    elif t == "model_response":
                                        text = str(d.get("text", ""))[:120].replace("\n", " ")
                                        console.print(f"  🤖 [white]{text}[/white]")
                                    elif t == "error":
                                        console.print(f"  ❌ [red]{d.get('error', '')}[/red]")
                                    elif t == "bump":
                                        console.print(f"  💡 [yellow]Bump: {str(d.get('insights', ''))[:80]}[/yellow]")
                                except Exception:
                                    console.print(f"  [dim]{line[:100]}[/dim]")
                            console.print()
                        except Exception as e:
                            console.print(f"[red]Failed to read trace: {e}[/red]")

            async def run_challenge(index: int, prob: dict) -> tuple[float, object | None, str, int]:
                title = str(prob.get("title") or "?")
                pid = int(prob.get("id") or 0)
                pts = int(prob.get("points") or 0)
                if pay:
                    specs = list(METACTF_PAY_MODELS)
                    st = runner_settings.model_copy(update={"gemini_rotate_chain": ""})
                    slow_escalate_specs: list[str] = []
                    slow_escalate_st: Settings | None = None
                else:
                    specs = model_specs_for_points(
                        pts,
                        default_three=tier_default_three,
                        kimi_nvidia_spec=KIMI_NVIDIA_SPEC,
                        glm_nvidia_spec=GLM_NVIDIA_SPEC,
                    )

                    if pts > 150:
                        st = runner_settings.model_copy(
                            update={"gemini_rotate_chain": "gemini-3-flash-preview,gemini-2.5-flash"}
                        )
                    else:
                        st = runner_settings.model_copy(update={"gemini_rotate_chain": ""})

                    slow_escalate_specs = []
                    slow_escalate_st = None
                    if pts <= 150 and gemini_keys:
                        slow_escalate_specs = ["gemini/gemini-3-flash-preview"]
                        slow_escalate_st = st.model_copy(
                            update={"gemini_rotate_chain": "gemini-3-flash-preview,gemini-2.5-flash"}
                        )

                slug = slug_challenge_dir(title)
                ch_dir = str(work_parent / f"id{pid}_{slug}")
                html_raw = str(prob.get("description") or "")
                problem_to_challenge_files(prob, ch_dir)
                meta = ChallengeMeta.from_directory(ch_dir)
                if is_instance_based_remote_challenge(html_raw, meta.description):
                    console.print(
                        f"[yellow]Instance-based challenge[/yellow] — {title!r} "
                        "expects a target you spawn (e.g. Kubernetes / remote instance)."
                    )
                    url = click.prompt(
                        f"Spawn the container for challenge {title!r}, then paste the full challenge URL",
                        default="",
                        show_default=False,
                    ).strip()
                    if url:
                        Path(ch_dir).joinpath("connection.txt").write_text(
                            url + "\n", encoding="utf-8"
                        )
                        meta = ChallengeMeta.from_directory(ch_dir)
                    else:
                        console.print(
                            f"[dim]No URL saved for {title!r}; solvers may miss the live target.[/dim]"
                        )

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
                    slow_solve_alert=lambda m: console.print(f"[yellow]{m}[/yellow]"),
                    slow_solve_seconds=60.0,
                    slow_solve_escalate_specs=slow_escalate_specs,
                    slow_solve_escalate_settings=slow_escalate_st if slow_escalate_specs else None,
                )

                running_swarms[pid] = swarm
                async with parallel_lock:
                    active_parallel.append((pid, title, swarm))
                try:
                    result = await swarm.run()
                finally:
                    running_swarms.pop(pid, None)
                    async with parallel_lock:
                        try:
                            active_parallel.remove((pid, title, swarm))
                        except ValueError:
                            pass

                status = getattr(result, "status", None) if result else None
                console.print(
                    f"[dim]Challenge task done: id={pid} {title!r} status={status}[/dim]"
                )
                return cost_tracker.total_cost_usd, result, title, pid

            poll_task = asyncio.create_task(poll_problems_json_loop(), name="metactf-poll-problems")
            menu_task: asyncio.Task[None] | None = None
            if sys.stdin.isatty():
                menu_task = asyncio.create_task(stdin_menu_loop(), name="metactf-stdin-menu")

            batch_tasks: list[asyncio.Task] = []
            try:
                outcomes: list[object] = []
                for batch_start in range(0, len(selected), METACTF_PARALLEL_CHALLENGES):
                    batch = selected[batch_start : batch_start + METACTF_PARALLEL_CHALLENGES]
                    batch_tasks = [
                        asyncio.create_task(
                            run_challenge(batch_start + j + 1, prob),
                            name=f"metactf-{prob.get('id')}",
                        )
                        for j, prob in enumerate(batch)
                    ]
                    outcomes.extend(await asyncio.gather(*batch_tasks, return_exceptions=True))
            except (KeyboardInterrupt, asyncio.CancelledError):
                for t in batch_tasks:
                    t.cancel()
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                console.print("\n[yellow]Interrupted — cancelling challenges.[/yellow]")
                raise
            finally:
                poll_shutdown.set()
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await poll_task
                if menu_task is not None:
                    menu_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await menu_task

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