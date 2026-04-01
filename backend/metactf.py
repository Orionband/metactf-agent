"""MetaCTF Compete API — fetch problems and submit flags (cookie auth)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

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


def _origin_and_referer(base_url: str) -> tuple[str, str]:
    u = urlparse(base_url.rstrip("/"))
    origin = f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else ""
    referer = base_url.rstrip("/") + "/"
    return origin, referer


def _response_json_or_raise(resp: httpx.Response, *, what: str) -> dict[str, Any]:
    """Parse JSON; raise RuntimeError with body snippet if not JSON (e.g. HTML login page)."""
    raw = (resp.text or "").strip().lstrip("\ufeff")
    if not raw:
        raise RuntimeError(
            f"MetaCTF {what}: empty response (HTTP {resp.status_code}). "
            "Check cookie (METACTF_COMPETE=...) and contest URL."
        )
    ct = (resp.headers.get("content-type") or "").lower()
    if "text/html" in ct and not raw.lstrip().startswith("{"):
        snippet = re.sub(r"\s+", " ", raw)[:400]
        raise RuntimeError(
            f"MetaCTF {what}: got HTML instead of JSON (HTTP {resp.status_code}). "
            "Usually invalid or expired cookie — log in again and copy the cookie. "
            f"Body starts: {snippet!r}"
        )
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        snippet = raw[:500].replace("\n", " ")
        raise RuntimeError(
            f"MetaCTF {what}: not valid JSON (HTTP {resp.status_code}): {e}. "
            f"Body starts: {snippet!r}"
        ) from e
    if not isinstance(out, dict):
        raise RuntimeError(f"MetaCTF {what}: expected JSON object, got {type(out).__name__}")
    return out


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
    clean = cookie_header_for_metactf(cookie)
    if not clean.strip():
        raise RuntimeError("MetaCTF: cookie is empty after stripping MCS_OPTIONS — set METACTF_COMPETE=...")
    origin, referer = _origin_and_referer(base_url)
    resp = await client.get(
        url,
        headers={
            "Cookie": clean,
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": referer,
            "Origin": origin,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    resp.raise_for_status()
    data = _response_json_or_raise(resp, what="problems_json")
    err = data.get("error")
    if err is True or (isinstance(err, str) and err):
        raise RuntimeError(f"MetaCTF API error flag: {data!r}")
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
    origin, referer = _origin_and_referer(base_url)
    resp = await client.post(
        url,
        data={"id": str(problem_id), "answer": answer.strip()},
        headers={
            "Cookie": clean,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json, text/plain, */*",
            "Referer": referer,
            "Origin": origin,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    resp.raise_for_status()
    try:
        data = _response_json_or_raise(resp, what="submit")
    except RuntimeError as e:
        return MetaSubmitResult(ok=False, display=str(e), attempts_left=None)

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


def solved_ids_from_payload(payload: dict[str, Any]) -> set[int]:
    """IDs from API `solved` (e.g. [147, 55, 77, ...]) as ints for matching each problem `id`."""
    out: set[int] = set()
    for raw in payload.get("solved") or []:
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def select_problems(
    payload: dict[str, Any],
    *,
    limit: int | None,
    skip_titles: set[str],
) -> list[dict[str, Any]]:
    """Exclude problems whose `id` is in `solved`, skip titles, sort by points then id."""
    solved = solved_ids_from_payload(payload)
    rows: list[dict[str, Any]] = []
    for p in payload.get("problems") or []:
        if not isinstance(p, dict):
            continue
        try:
            pid = int(p.get("id"))
        except (TypeError, ValueError):
            continue
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
