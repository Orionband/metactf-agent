"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter (required for all LLM calls)
    openrouter_api_key: str = ""
    # Optional multi-key support. Comma or newline separated.
    # Example:
    # OPENROUTER_API_KEYS=sk-or-v1-key1,sk-or-v1-key2
    openrouter_api_keys: str = ""

    # Optional: coordinator uses this model spec (openrouter/...); empty = first DEFAULT_MODELS entry
    coordinator_model: str = ""

    # Infra
    sandbox_image: str = "ctf-sandbox"
    max_concurrent_challenges: int = 10
    max_attempts_per_challenge: int = 3
    container_memory_limit: str = "16g"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_openrouter_keys(self) -> list[str]:
        """Get all configured OpenRouter keys (multi-key first, then single-key fallback)."""
        keys: list[str] = []
        if self.openrouter_api_keys:
            raw = self.openrouter_api_keys.replace("\n", ",")
            keys.extend([k.strip() for k in raw.split(",") if k.strip()])
        if self.openrouter_api_key and self.openrouter_api_key.strip():
            key = self.openrouter_api_key.strip()
            if key not in keys:
                keys.append(key)
        return keys
