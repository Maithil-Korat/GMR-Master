from __future__ import annotations

import os
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

REQUEST_TIMEOUT_MS = 120_000
ENV_KEY = "GEMINI_API_KEY"


class MissingAPIKeyError(RuntimeError):
    pass


def has_api_key() -> bool:
    return bool(os.environ.get(ENV_KEY, "").strip())


def require_api_key(context: str = "Gemini API") -> str:
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"{ENV_KEY} is not set — {context} cannot run. "
            f"Add {ENV_KEY}=<your key> to a .env file in the project root and restart the server."
        )
    return key


def build_client(context: str = "Gemini API") -> Any:
    key = require_api_key(context)
    if genai is None:
        raise RuntimeError("google-genai is not installed")
    return genai.Client(
        api_key=key,
        http_options=genai_types.HttpOptions(
            api_version="v1beta",
            timeout=REQUEST_TIMEOUT_MS,
        ),
    )


def preflight(context: str = "pipeline") -> None:
    require_api_key(context)
