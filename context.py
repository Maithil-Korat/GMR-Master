from __future__ import annotations

from typing import Any

import database
from fields import map_file

BLOCKED_ROLES = {"VENDOR_METADATA", "IDENTIFIER", "OTHER_METADATA", "UNKNOWN"}


def build(row: dict[str, Any], mapping: list[dict[str, Any]]) -> dict[str, str]:
    primary: list[str] = []
    secondary: list[str] = []
    for item in mapping:
        column = item["column"]
        value = str(row.get(column, "") or "").strip()
        if not value or item["role"] in BLOCKED_ROLES:
            continue
        if item["use_in_primary"]:
            primary.append(value)
        if item["use_in_secondary"]:
            secondary.append(value)
    return {
        "row_id": str(row["Sr. no."]),
        "combined_description": " ".join(primary),
        "secondary_product_description": " ".join(secondary),
    }


def apply(file_id: str, mapping: list[dict[str, Any]], batch_id: str) -> list[dict[str, str]]:
    source_rows = database.rows(file_id)
    contexts = [build(row, mapping) for row in source_rows]
    database.update_contexts(file_id, batch_id, {item["column"]: item for item in mapping}, contexts)
    return contexts


def prepare(file_id: str, api: Any | None = None, poll_seconds: float = 15.0) -> list[dict[str, str]]:
    mapping = map_file(file_id)
    return apply(file_id, mapping, batch_id="heuristic-map")