"""Model resolution — OpenRouter via pydantic-ai OpenRouterModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from backend.config import Settings

# Default model specs
DEFAULT_MODELS: list[str] = [
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "nvidia/moonshotai/kimi-k2.5",
    "nvidia/google/gemma-4-31b-it",
    "openrouter/openai/gpt-oss-120b:free",
]

FALLBACK_MODELS: list[str] = [
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/openai/gpt-oss-120b:free",
    "openrouter/z-ai/glm-4.5-air:free",
]

CONTEXT_WINDOWS: dict[str, int] = {
    "google/gemma-4-26b-a4b-it:free": 200_000,
    "google/gemma-4-31b-it:free": 200_000,
    "google/gemma-3-27b-it": 200_000,
    "nvidia/nemotron-3-super-120b-a12b:free": 200_000,
    "openai/gpt-oss-120b:free": 200_000,
    "z-ai/glm-4.5-air:free": 200_000,
    "moonshotai/kimi-k2.5": 200_000,
    "z-ai/glm5": 200_000,
    "qwen/qwen3.6-plus": 200_000,
}

VISION_MODELS: set[str] = {
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-3-27b-it",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3.6-plus",
}


def resolve_model(spec: str, settings: Settings) -> Model:
    """Resolve an 'openrouter/model_id' spec to a Pydantic AI Model."""
    provider = provider_from_spec(spec)
    if provider != "openrouter":
        raise ValueError(f"Only openrouter is supported; got {spec!r}")
    model_id = model_id_from_spec(spec)
    keys = settings.get_openrouter_keys()
    if not keys:
        raise RuntimeError("OPENROUTER_API_KEY / OPENROUTER_API_KEYS is not set")
    return OpenRouterModel(
        model_id,
        provider=OpenRouterProvider(api_key=keys[0]),
    )


def resolve_model_settings(spec: str) -> ModelSettings:
    """OpenRouter settings with reasoning enabled (reasoning_details preserved across turns)."""
    provider = spec.split("/", 1)[0]
    if provider != "openrouter":
        return ModelSettings(max_tokens=128_000)
    return OpenRouterModelSettings(
        max_tokens=128_000,
        openrouter_reasoning={"enabled": True},
        # Pydantic AI's OpenAI-model backend may send `tool_choice` when tools are
        # present. Some OpenRouter providers/models don't support it, which causes
        # a 404 "No endpoints found that support the provided 'tool_choice' value."
        # This forces routing to only providers that support the request params.
        openrouter_provider={"require_parameters": True},
    )


def model_id_from_spec(spec: str) -> str:
    """Model id as used by OpenRouter (everything after the provider prefix)."""
    parts = spec.split("/")
    if len(parts) >= 2 and parts[0] == "openrouter":
        return "/".join(parts[1:])
    if len(parts) >= 2:
        return "/".join(parts[1:])
    return spec


def provider_from_spec(spec: str) -> str:
    """Extract the provider from a spec."""
    return spec.split("/", 1)[0]


def openrouter_spec_from_user_id(user: str) -> str:
    """Normalize CLI input like ``qwen/qwen3.7-plus:free`` to ``openrouter/qwen/...``."""
    t = (user or "").strip()
    if not t:
        raise ValueError("empty OpenRouter model id")
    if t.startswith("openrouter/"):
        return t
    return f"openrouter/{t}"


def effort_from_spec(spec: str) -> str | None:
    """Unused for OpenRouter defaults; kept for API compatibility."""
    return None


def supports_vision(spec: str) -> bool:
    return model_id_from_spec(spec) in VISION_MODELS


def context_window(spec: str) -> int:
    return CONTEXT_WINDOWS.get(model_id_from_spec(spec), 200_000)
