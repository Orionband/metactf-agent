"""ChallengeSwarm variant — submits flags to MetaCTF and stops on correct or low attempt budget."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from backend.agents.swarm import ChallengeSwarm
from backend.metactf import submit_flag

logger = logging.getLogger(__name__)

_STOPL_ATTEMPTS = 3


@dataclass
class MetaCTFChallengeSwarm(ChallengeSwarm):
    """POST flags to MetaCTF; cancel swarm on correct or when attempts_remaining ≤ 3."""

    metactf_base_url: str = ""
    metactf_cookie: str = ""
    metactf_problem_id: int = 0
    metactf_http: httpx.AsyncClient | None = field(default=None, repr=False)

    async def try_submit_flag(self, flag: str, model_spec: str) -> tuple[str, bool]:
        if self.no_submit:
            return await ChallengeSwarm.try_submit_flag(self, flag, model_spec)

        async with self._flag_lock:
            if self.confirmed_flag:
                return f"ALREADY SOLVED — flag already confirmed: {self.confirmed_flag}", True

            normalized = flag.strip()
            if not normalized:
                return "Empty flag — nothing to submit.", False

            if normalized in self._submitted_flags:
                return "INCORRECT — already tried this exact flag.", False

            wrong_count = self._submit_count.get(model_spec, 0)
            cooldown_idx = min(wrong_count, len(self.SUBMISSION_COOLDOWNS) - 1)
            cooldown = self.SUBMISSION_COOLDOWNS[cooldown_idx]
            if cooldown > 0:
                last_time = self._last_submit_time.get(model_spec, 0)
                elapsed = time.monotonic() - last_time
                if elapsed < cooldown:
                    remaining = int(cooldown - elapsed)
                    return (
                        f"COOLDOWN — wait {remaining}s before submitting again. "
                        f"You have {wrong_count} duplicate or rejected attempts. "
                        "Use this time to verify your flag.",
                        False,
                    )

            self._submitted_flags.add(normalized)

            client = self.metactf_http
            if client is None:
                self._submitted_flags.discard(normalized)
                return "MetaCTF: internal error (no HTTP client).", False

            try:
                result = await submit_flag(
                    client,
                    self.metactf_base_url,
                    self.metactf_cookie,
                    self.metactf_problem_id,
                    normalized,
                )
            except Exception as e:
                self._submitted_flags.discard(normalized)
                display = f"MetaCTF submit failed for flag '{normalized}': {e}"
                logger.warning("[%s] %s", self.meta.name, display)
                return display, False

            if result.ok:
                self.confirmed_flag = normalized
                self.cancel_event.set()
                logger.info("[%s] MetaCTF accepted flag: %s", self.meta.name, normalized)
                return result.display, True

            if result.attempts_left is not None and result.attempts_left <= _STOPL_ATTEMPTS:
                self.cancel_event.set()
                msg = (
                    f"{result.display} — stopping solvers (MetaCTF reports <= {_STOPL_ATTEMPTS} attempts left)."
                )
                logger.warning("[%s] %s", self.meta.name, msg)
                return msg, False

            self._submit_count[model_spec] = wrong_count + 1
            self._last_submit_time[model_spec] = time.monotonic()
            return result.display, False