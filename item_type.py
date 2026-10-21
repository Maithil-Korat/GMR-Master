from __future__ import annotations

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import database
import gemini
from normalize import normalize_item_type

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

ItemType = Literal["Saleable", "Non-Saleable"]
MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 30
DIRECT_EXECUTION_THRESHOLD = 200

PATTERNS = (
    re.compile(r"\bTESTER\b", re.IGNORECASE),
    re.compile(r"\bTSTR(?:_|\b)", re.IGNORECASE),
    re.compile(r"\bTST(?:_|\b)", re.IGNORECASE),
    re.compile(r"\bGWP(?:_|\b)", re.IGNORECASE),
    re.compile(r"\bPWP(?:_|\b)", re.IGNORECASE),
    re.compile(r"\bSAMPLE\b", re.IGNORECASE),
    re.compile(r"\bDUMMY\b", re.IGNORECASE),
    re.compile(r"\bPOSM\b", re.IGNORECASE),
    re.compile(r"\bDISPLAY\b", re.IGNORECASE),
    re.compile(r"\bDEMO\b", re.IGNORECASE),
    re.compile(r"\bNOT\s+FOR\s+SALE\b", re.IGNORECASE),
)

SYSTEM_PROMPT = """You are the GMR Item Type classification reviewer.
Classify every item as exactly 'Saleable' or 'Non-Saleable'.

- Saleable: Commercial products intended for retail sale.
- Non-Saleable: Testers, samples, display/demo units, POSM, promotional gifts (GWP/PWP), dummy items, or items explicitly not for sale.

Return literal evidence copied from the text. Return only the requested structured schema. Preserve every row_id exactly."""


class SingleModelDecision(BaseModel):
    row_id: str
    item_type: ItemType
    evidence: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    review_required: bool
    decision_notes: str


class BatchItemTypeResponse(BaseModel):
    items: list[SingleModelDecision]


class ItemTypeDecision(BaseModel):
    row_id: str
    item_type: ItemType
    parser_candidate: ItemType | None
    parser_evidence: list[str]
    evidence: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    review_required: bool
    decision_notes: str


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
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


def context(row: dict[str, Any]) -> str:
    values = [row.get("Combined Description", ""), row.get("Secondary Product Description", "")]
    return " ".join(str(value or "").strip() for value in values if str(value or "").strip())


def candidate(row: dict[str, Any]) -> dict[str, Any]:
    value = context(row)
    evidence: list[str] = []
    for pattern in PATTERNS:
        for match in pattern.finditer(value):
            m = match.group(0)
            if m.casefold() not in {item.casefold() for item in evidence}:
                evidence.append(m)
    return {
        "row_id": str(row.get("Sr. no.") or row.get("row_id") or ""),
        "parser_candidate": "Non-Saleable" if evidence else None,
        "parser_evidence": evidence,
    }


def build_chunk_payload(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for row in chunk:
        sig = candidate(row)
        payload.append({
            "row_id": str(row.get("Sr. no.") or row.get("row_id") or ""),
            "parser_signal": sig["parser_candidate"],
            "combined_description": row.get("Combined Description", ""),
            "secondary_product_description": row.get("Secondary Product Description", ""),
        })
    return payload


def review(row: dict[str, Any], model_result: dict[str, Any]) -> dict[str, Any]:
    model_result = dict(model_result)
    model_result["item_type"] = normalize_item_type(model_result.get("item_type"))
    model = SingleModelDecision.model_validate(model_result)
    sig = candidate(row)
    conflict = sig["parser_candidate"] is not None and sig["parser_candidate"] != model.item_type
    notes = model.decision_notes
    if conflict:
        notes = f"Parser/Gemini conflict. {notes}".strip()

    return ItemTypeDecision(
        row_id=model.row_id,
        item_type=model.item_type,
        parser_candidate=sig["parser_candidate"],
        parser_evidence=sig["parser_evidence"],
        evidence=model.evidence,
        confidence_score=model.confidence_score,
        review_required=model.review_required or conflict,
        decision_notes=notes,
    ).model_dump()


def client() -> Any:
    return gemini.build_client("Item Type classification")


def direct_parallel_classify(rows: list[dict[str, Any]], api: Any, on_progress: Any | None = None) -> list[dict[str, Any]]:
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    schema = inline_schema(BatchItemTypeResponse)

    def _process_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = build_chunk_payload(chunk)
        response = api.models.generate_content(
            model=MODEL_NAME,
            contents=[f"Classify these products:\n{json.dumps(payload, ensure_ascii=False)}"],
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.0,
            },
        )
        data = json.loads(response.text)
        return data.get("items", [])

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_items in executor.map(_process_chunk, chunks):
            results.extend(chunk_items)
            if on_progress:
                on_progress(len(chunk_items))
    return results


def batch_classify(rows: list[dict[str, Any]], api: Any, poll_seconds: float = 10.0) -> list[dict[str, Any]]:
    """Chunked Batch API queue for massive datasets (>200 items)."""
    requests = []
    schema = inline_schema(BatchItemTypeResponse)

    for start in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[start : start + CHUNK_SIZE]
        payload = build_chunk_payload(chunk)
        requests.append(
            json.dumps(
                {
                    "key": f"chunk-{start // CHUNK_SIZE}",
                    "request": {
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"role": "user", "parts": [{"text": f"Classify:\n{json.dumps(payload, ensure_ascii=False)}"}]}],
                        "generation_config": {
                            "response_mime_type": "application/json",
                            "response_schema": schema,
                            "temperature": 0.0,
                        },
                    },
                },
                ensure_ascii=False,
            )
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as file:
        file.write("\n".join(requests) + "\n")
        input_path = Path(file.name)

    try:
        uploaded = api.files.upload(
            file=str(input_path),
            config=types.UploadFileConfig(display_name="gmr-item-type", mime_type="jsonl"),
        )
        job = api.batches.create(
            model=MODEL_NAME,
            src=uploaded.name,
            config={"display_name": "gmr-item-type"},
        )
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while getattr(job, "state", "") not in terminal:
            time.sleep(poll_seconds)
            job = api.batches.get(name=job.name)

        if getattr(job, "state", "") != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Item Type Batch failed: {getattr(job, 'error', '')}")

        res_file = getattr(getattr(job, "dest", None), "file_name", None)
        content = api.files.download(file=res_file).decode("utf-8")

        results = []
        for line in content.splitlines():
            if not line.strip():
                continue
            val = json.loads(line)
            candidates = (val.get("response") or {}).get("candidates") or []
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    data = json.loads(text)
                    results.extend(data.get("items", []))
        return results
    finally:
        input_path.unlink(missing_ok=True)


def classify(rows: list[dict[str, Any]], api: Any | None = None, poll_seconds: float = 10.0, on_progress: Any | None = None) -> list[dict[str, Any]]:
    if not rows:
        return []
    
    service = api or client()

    # Fast path: <200 rows uses direct parallel API (~2 sec)
    if len(rows) <= DIRECT_EXECUTION_THRESHOLD:
        model_results = direct_parallel_classify(rows, service, on_progress=on_progress)
    else:
        # Large files use chunked Batch API
        model_results = batch_classify(rows, service, poll_seconds)

    by_id = {str(item.get("row_id")): item for item in model_results}
    decisions = []
    for row in rows:
        row_id = str(row.get("Sr. no.") or row.get("row_id") or "")
        if row_id not in by_id:
            raise RuntimeError(f"Item Type classification omitted row {row_id}")
        decisions.append(review(row, by_id[row_id]))
    return decisions


def classify_file(file_id: str, poll_seconds: float = 10.0, job_manager: Any | None = None) -> list[dict[str, Any]]:
    rows = database.rows(file_id)
    missing = [row["Sr. no."] for row in rows if not (row.get("Combined Description") or row.get("Secondary Product Description"))]
    if missing:
        raise RuntimeError(f"Separated product contexts are required before Item Type classification: {missing[:10]}")

    total_rows = len(rows)
    _done = {"n": 0}

    def _chunk_progress(count: int) -> None:
        if job_manager:
            _done["n"] += count
            progress = int((_done["n"] / max(1, total_rows)) * 100)
            job_manager.set(
                file_id, "RUNNING", "ITEM_TYPE", progress,
                f"Classifying item types: {_done['n']}/{total_rows}",
                progress_counter=f"{_done['n']}/{total_rows}"
            )

    if job_manager:
        job_manager.set(
            file_id, "RUNNING", "ITEM_TYPE", 0,
            f"Classifying item types: 0/{total_rows}",
            progress_counter=f"0/{total_rows}"
        )

    decisions = classify(rows, poll_seconds=poll_seconds, on_progress=_chunk_progress)

    # The batch API path (large files) does not report per-chunk; snap to 100%.
    if job_manager and total_rows > DIRECT_EXECUTION_THRESHOLD:
        job_manager.set(
            file_id, "RUNNING", "ITEM_TYPE", 100,
            f"Classified {total_rows}/{total_rows} item types",
            progress_counter=f"{total_rows}/{total_rows}"
        )

    database.update_item_types(file_id, decisions)
    return decisions