from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROMPTS: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    global _PROMPTS
    if _PROMPTS is None:
        path = Path(__file__).parent / "prompts.json"
        with open(path, encoding="utf-8") as f:
            _PROMPTS = json.load(f)
    return _PROMPTS


def taxonomy(product_type: str | None, audit: bool = False) -> str:
    data = load()["taxonomy"]
    base = data["audit_base"] if audit else data["base"]
    rules = data["audit_rules"] if audit else data["rules"]
    categories = data["categories"]

    section = categories.get(product_type, "")
    return f"{base}\n\n{section}\n\n{rules}"


def content(product_type: str | None, audit: bool = False) -> str:
    data = load()["content"]
    base = data["audit_base"] if audit else data["base"]
    rules = data["audit_rules"] if audit else data["rules"]
    categories = data["categories"]

    section = categories.get(product_type, "")
    return f"{base}\n\n{section}\n\n{rules}"


def title(product_type: str | None, audit: bool = False) -> str:
    data = load()["title"]
    base = data["audit_base"] if audit else data["base"]
    rules = data["audit_rules"] if audit else data["rules"]
    categories = data["categories"]

    section = categories.get(product_type, "")
    return f"{base}\n\n{section}\n\n{rules}"
