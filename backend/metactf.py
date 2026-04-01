"""MetaCTF Compete API — fetch problems and submit flags (cookie auth)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; ctf-agent-metactf/1.0)"

_ATTEMPTS_LEFT_RE = re.compile(r"(\d+)\s+attempts?\s+left", re.IGNORECASE)


@dataclass
class MetaSubmitResult:
    ok: bool
    display: str
    attempts_left: int | None  # None if unknown


def normalize_metactf_base_url(raw: str) -> str:
    """Strip scheme, build https://host/contestId."""
    s = raw.strip()
    s = re.sub(r"^https?://", "", s, flags=re.I).strip().rstrip("/")
    if "/" not in s:
        raise ValueError(
            "MetaCTF URL must include the contest path, e.g. compete.metactf.com/576 "
            "or https://compete.metactf.com/576"
        )
    host, _, path = s.partition("/")
    host = host.strip()
    path = path.strip("/")
    if not host or not path:
        raise ValueError(f"Invalid MetaCTF URL: {raw!r}")
    return f"https://{host}/{path}"


def cookie_header_for_metactf(raw: str) -> str:
    """Drop MCS_OPTIONS=... segments (UI filters); keep session cookies only."""
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    kept: list[str] = []
    for p in parts:
        if p.upper().startswith("MCS_OPTIONS="):
            continue
        kept.append(p)
    return "; ".join(kept)


def _parse_attempts_left(text: str) -> int | None:
    if not text:
        return None
    m = _ATTEMPTS_LEFT_RE.search(text)
    if m:
        return int(m.group(1))
    return None


async def fetch_problems_json(client: httpx.AsyncClient, base_url: str, cookie: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/problems_json.php"
    resp = await client.get(
        url,
        headers={
            "Cookie": cookie_header_for_metactf(cookie),
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"MetaCTF API error: {data!r}")
    return data


async def submit_flag(
    client: httpx.AsyncClient,
    base_url: str,
    cookie: str,
    problem_id: int,
    answer: str,
) -> MetaSubmitResult:
    url = f"{base_url.rstrip('/')}/api/submit.php"
    clean = cookie_header_for_metactf(cookie)
    resp = await client.post(
        url,
        data={"id": str(problem_id), "answer": answer.strip()},
        headers={
            "Cookie": clean,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json, text/plain, */*",
        },
    )
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception as e:
        return MetaSubmitResult(ok=False, display=f"MetaCTF submit: invalid JSON ({e}): {resp.text[:300]}", attempts_left=None)

    status = str(data.get("status", "")).lower()
    mes = str(data.get("mes", "") or "")
    title = str(data.get("title", "") or "")
    combined = f"{mes} {title}".strip()
    attempts = _parse_attempts_left(combined) or _parse_attempts_left(title) or _parse_attempts_left(mes)

    if status == "success":
        return MetaSubmitResult(
            ok=True,
            display=f"CORRECT — MetaCTF: {mes or title or 'accepted'}",
            attempts_left=attempts,
        )

    return MetaSubmitResult(ok=False, display=combined or str(data), attempts_left=attempts)


def select_problems(
    payload: dict[str, Any],
    *,
    limit: int | None,
    skip_titles: set[str],
) -> list[dict[str, Any]]:
    """Filter unsolved, skip titles, sort by points ascending then id."""
    solved = set(payload.get("solved") or [])
    rows: list[dict[str, Any]] = []
    for p in payload.get("problems") or []:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if pid in solved:
            continue
        if not int(p.get("solvable") or 0):
            continue
        title = str(p.get("title") or "").strip()
        if title in skip_titles:
            continue
        rows.append(p)

    rows.sort(key=lambda x: (int(x.get("points") or 0), int(x.get("id") or 0)))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def slug_challenge_dir(title: str) -> str:
    raw = re.sub(r"[^\w\s\-]+", "", title, flags=re.UNICODE).strip()[:80]
    raw = re.sub(r"\s+", "_", raw).strip("_")
    return raw or "challenge"


def problem_to_challenge_files(problem: dict[str, Any], dest_dir: str) -> None:
    """Write challenge.txt (+ optional files) for ChallengeMeta.from_directory."""
    from pathlib import Path

    from markdownify import markdownify

    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)

    html = str(problem.get("description") or "")
    desc_md = markdownify(html, heading_style="ATX").strip() or "(empty description)"

    pid = int(problem.get("id") or 0)
    pts = int(problem.get("points") or 0)
    cat = str(problem.get("category") or "")
    title = str(problem.get("title") or "challenge")

    header = (
        f"<!-- MetaCTF problem_id={pid} points={pts} category={cat} -->\n"
        f"# {title}\n\n"
        f"**Points**: {pts}  \n**Category**: {cat}\n\n"
    )
    footer = (
        "\n\n---\n"
        "**Sandbox**: You may execute binaries (copy from `/challenge/challenge/` to "
        "`/challenge/workspace/` before `chmod` if needed). "
        "Use `web_fetch`, `curl`, or `wget` for remote files and web tasks.\n"
    )
    (root / "challenge.txt").write_text(header + desc_md + footer, encoding="utf-8")


def model_specs_for_points(points: int, *, default_three: list[str], qwen_spec: str) -> list[str]:
    """<=150: Qwen only; 151–200: three OpenRouter; >200: three + Gemini (rotate via settings)."""
    if points <= 150:
        return [qwen_spec]
    if points <= 200:
        return list(default_three)
    return list(default_three) + ["gemini/gemini-3-flash-preview"]
