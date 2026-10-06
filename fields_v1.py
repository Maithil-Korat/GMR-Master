from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import database
import logger

FieldRole = Literal[
    "PRODUCT_DESCRIPTION", "SECONDARY_PRODUCT_DESCRIPTION", "BRAND", "VARIANT", "PHYSICAL_SPECIFICATION",
    "PRODUCT_CODE", "ITEM_TYPE_SIGNAL", "CATEGORY_HINT", "VENDOR_METADATA", "IDENTIFIER", "OTHER_METADATA", "UNKNOWN",
]
MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """You are a semantic source-column mapper for a retail and duty-free product catalog.

Map every source column exactly once. Use the column name, data shape, non-empty ratio, uniqueness, and sample values together. Do not classify a product and do not generate a title.

Roles:
PRODUCT_DESCRIPTION: primary product identity or description text.
SECONDARY_PRODUCT_DESCRIPTION: an alternative product description source used for reconciliation.
BRAND: consumer-facing brand evidence.
VARIANT: product line, model, flavour, shade, collection, or variant evidence.
PHYSICAL_SPECIFICATION: size, volume, weight, pack, colour, material, age, or similar physical facts.
PRODUCT_CODE: official model or collection code that may distinguish a product.
ITEM_TYPE_SIGNAL: tester, sample, display, POSM, promotional, or saleability evidence.
CATEGORY_HINT: source category or division hint.
VENDOR_METADATA: supplier, vendor, importer, distributor, or logistics entity.
IDENTIFIER: barcode, UPC, EAN, SKU, item reference, or purely administrative identifier.
OTHER_METADATA: HSN, location, warehouse, quantity, dates, or unrelated operational data.
UNKNOWN: insufficient evidence for a safe role.

Barcode and other identifiers are never used in either product-facing context. Vendor metadata is never used in either product-facing context. Unknown fields remain archival only.

Set use_in_primary and use_in_secondary from the role and observed values. Product descriptions, brand, variant, physical specifications, and official product codes may contribute to the primary context. Alternative descriptions may contribute to the secondary context. Item Type signals may contribute to the secondary context. Vendor metadata, identifiers, other metadata, and unknown fields must not contribute to either product context.

Return only the requested structured schema. Preserve every source column name exactly."""


class ColumnProfile(BaseModel):
    column: str
    samples: list[str]
    non_empty_ratio: float = Field(ge=0.0, le=1.0)
    unique_ratio: float = Field(ge=0.0, le=1.0)
    numeric_ratio: float = Field(ge=0.0, le=1.0)


class FieldMapping(BaseModel):
    column: str
    role: FieldRole
    use_in_primary: bool
    use_in_secondary: bool
    confidence: float = Field(ge=0.0, le=1.0)


class FieldMapResponse(BaseModel):
    mappings: list[FieldMapping]


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Inlines all Pydantic $defs/$ref pointers into a flat OpenAPI schema for Gemini Batch API."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                return _resolve(defs.get(ref_key, {}))
            return {k: _resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(raw)


def profiles(rows: list[dict[str, Any]], columns: list[str], sample_limit: int = 8) -> list[dict[str, Any]]:
    result = []
    total = len(rows)
    for column in columns:
        values = [str(row.get(column, "") or "").strip() for row in rows]
        non_empty = [value for value in values if value]
        numeric = [value for value in non_empty if re.fullmatch(r"[0-9 ./%+-]+", value)]
        result.append(
            ColumnProfile(
                column=column,
                samples=non_empty[:sample_limit],
                non_empty_ratio=len(non_empty) / total if total else 0.0,
                unique_ratio=len(set(non_empty)) / len(non_empty) if non_empty else 0.0,
                numeric_ratio=len(numeric) / len(non_empty) if non_empty else 0.0,
            ).model_dump()
        )
    return result


def prompt(column_profiles: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "COLUMN PROFILES:",
            json.dumps(column_profiles, ensure_ascii=False),
            "TASK:",
            "Map every column exactly once. Use the observed samples and statistics. Do not omit columns and do not infer product content beyond the supplied evidence.",
        ]
    )


def validate(mapping: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    expected = list(columns)
    received = [str(item.get("column", "")) for item in mapping]
    if len(received) != len(set(received)):
        raise ValueError("Semantic field mapping contains duplicate columns")
    if set(received) != set(expected):
        missing = sorted(set(expected) - set(received))
        extra = sorted(set(received) - set(expected))
        raise ValueError(f"Semantic field mapping coverage failed: missing={missing} extra={extra}")
    blocked = {"VENDOR_METADATA", "IDENTIFIER", "OTHER_METADATA", "UNKNOWN"}
    result = []
    for item in mapping:
        value = FieldMapping.model_validate(item)
        if value.role in blocked:
            value = value.model_copy(update={"use_in_primary": False, "use_in_secondary": False})
        result.append(value.model_dump())
    return result


def state(job: Any) -> str:
    value = getattr(job, "state", "")
    return str(getattr(value, "name", value))


def response_text(value: dict[str, Any]) -> str:
    response = value.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return parts[0].get("text", "") if parts else ""


def client() -> Any:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for semantic field mapping")
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError("google-genai is required for semantic field mapping") from error
    return genai.Client(api_key=key)


def batch(column_profiles: list[dict[str, Any]], api: Any | None = None, poll_seconds: float = 15.0) -> tuple[list[dict[str, Any]], str]:
    service = api or client()
    request = {
        "key": "field-map",
        "request": {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt(column_profiles)}]}],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": inline_schema(FieldMapResponse),
                "temperature": 0.0,
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as file:
        file.write(json.dumps(request, ensure_ascii=False) + "\n")
        input_path = Path(file.name)
    try:
        logger.event("FIELD_MAP", "request_prepared", columns=len(column_profiles))
        from google.genai import types
        uploaded = service.files.upload(
            file=str(input_path),
            config=types.UploadFileConfig(display_name="field-map", mime_type="jsonl"),
        )
        logger.event("FIELD_MAP", "file_uploaded")
        job = service.batches.create(model=MODEL_NAME, src=uploaded.name, config={"display_name": "field-map"})
        batch_id = job.name
        logger.event("FIELD_MAP", "batch_submitted", batch_id=batch_id)
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        previous = ""
        while state(job) not in terminal:
            current = state(job)
            if current != previous:
                logger.event("FIELD_MAP", "batch_state", batch_id=batch_id, state=current)
                previous = current
            time.sleep(poll_seconds)
            job = service.batches.get(name=batch_id)
        final_state = state(job)
        logger.event("FIELD_MAP", "batch_state", batch_id=batch_id, state=final_state)
        if final_state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Field mapping Batch terminated with state {final_state}: {getattr(job, 'error', '')}")
        result_file = getattr(getattr(job, "dest", None), "file_name", None)
        if not result_file:
            raise RuntimeError("Field mapping Batch returned no result file")
        content = service.files.download(file=result_file).decode("utf-8")
        values = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not values or values[0].get("error"):
            raise RuntimeError(f"Field mapping response failed: {values[0].get('error', '') if values else 'empty'}")
        mapping = FieldMapResponse.model_validate_json(response_text(values[0])).model_dump()["mappings"]
        logger.event("FIELD_MAP", "response_parsed", mappings=len(mapping), batch_id=batch_id)
        return mapping, batch_id
    finally:
        input_path.unlink(missing_ok=True)


def map_file(file_id: str, api: Any | None = None, poll_seconds: float = 15.0) -> tuple[list[dict[str, Any]], str]:
    source_rows = database.rows(file_id)
    source_columns = database.columns(file_id)
    column_profiles = profiles(source_rows, source_columns)
    logger.event("FIELD_MAP", "file_loaded", file_id=file_id, rows=len(source_rows), columns=len(source_columns))
    mapping, batch_id = batch(column_profiles, api=api, poll_seconds=poll_seconds)
    mapping = validate(mapping, source_columns)
    return mapping, batch_id