"""Round-robin OpenRouter API key selection."""

from __future__ import annotations

import threading
from itertools import count

_LOCK = threading.Lock()
_COUNTER = count(0)
_LAST_KEYS: tuple[str, ...] = ()


def next_openrouter_key(keys: list[str]) -> str:
    """Return the next key in round-robin order.

    The counter is process-global so requests from all solver tasks are evenly
    distributed across available keys.
    """
    if not keys:
        raise RuntimeError("No OpenRouter API keys configured")
    key_tuple = tuple(keys)
    with _LOCK:
        global _LAST_KEYS
        if _LAST_KEYS != key_tuple:
            _LAST_KEYS = key_tuple
        idx = next(_COUNTER) % len(key_tuple)
        return key_tuple[idx]

