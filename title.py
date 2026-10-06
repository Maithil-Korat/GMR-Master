from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import database
import logger
from normalize import changed, normalize_title

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 15
DIRECT_EXECUTION_THRESHOLD = 200

GenerationQuality = Literal["CLEAN", "SPARSE_INPUT", "CONFLICT", "AMBIGUOUS"]
AuditQuality = Literal["VERIFIED_CLEAN", "NORMALIZED_FIXED", "SPARSE_INPUT", "UNGROUNDED_FLAG"]

GENERATION_SYSTEM = """You are a senior global product-data specialist for travel retail and multi-brand catalogs.
Generate one grounded Cleaned Title for every row using the supplied product-facing evidence.

Universal Principles:
1. Brand First & Initialism Expansion:
   - Identify the consumer-facing brand from the product context and style code prefixes.
   - Brand must always be the first word in full (expand leading brand initialisms when supported by context). Never leave a 2-letter initialism as the brand.

2. Supplier Jargon & Shorthand Elimination:
   - Strip internal warehouse/manufacturing codes that have no consumer meaning:
     - Plating, buckle hardware, and finishing shorthand (e.g., SPa, BkP, SGo, HSHR, RDOB, REC, MQT, MTB, FI/CS, SFT, SAR, GRN).
     - Database prefixes (e.g., WA_, AP_, FB_, ACC_).
   - Retain recognized consumer collections and descriptors (e.g., Sartorial, Extreme, Soft Grain, Meisterstuck, Classic, 1858, Heritage).

3. Attribute & Color Normalization:
   - Expand color abbreviations to natural Title Case words ('Bk' -> 'Black', 'Eb' -> 'Ebony', 'RdOk' -> 'Red Ochre', 'FrIr' -> 'Forge Iron', 'FrBl' -> 'Forge Blue', 'MdBl' -> 'Medium Blue').
   - Retain standard commercial capacity terms (e.g., preserve '4cc', '6cc', '8cc' for card slots; do not expand to 'Card Compartments').
   - Material deduction: Luxury small leather goods lines are inherently Leather—include 'Leather' as the material descriptor.
   - Watch movement: 'AU-AU' or 'AU' denotes Automatic movement—include 'Automatic' in the title.
   - Unit spacing: '100cl' (no space), '50 ml', '100 g', '42 mm', '35 mm' (with space).
   - Age format: 'N Years Old'.

4. Official Style/Model Numbers:
   - Place official numerical style/model codes at the very end of the title if present.

Category Structural Formats:
- Fashion & Accessories: Brand, Collection/Line, Feature (e.g. Reversible, 3-Compartment), Product Type, Color, Size/Dimensions, Style Code.
- Watches: Brand, Collection/Model, Movement (e.g. Automatic), Gender, Product Type, Dial Color, Strap Material, Case Size, Code.
- Writing & Stationery: Brand, Collection, Descriptor, Product Type, Color/Volume/Size, Code.
- Alcohol/Spirits: Brand, Variant, Descriptor, Country/Style, Age, Pack, Volume.

Return only structured JSON and preserve row_id exactly."""

AUDIT_SYSTEM = """You are the independent title quality auditor for a global travel-retail catalog.
Audit Pass 1 titles against Product Type, Item Type, Combined Description, and Universal Raw Context.

Audit Checks:
1. Brand is full and consumer-facing at the beginning of the title (no raw 2-letter initialisms).
2. Internal manufacturing, hardware, plating, or ERP shorthand tokens (e.g., 'SPa', 'HSHR', 'REC BkP', 'FI/CS', 'WA_') are completely stripped.
3. Colors and specifications are expanded into natural Title Case English.
4. Clean style/model number is preserved cleanly at the end.
5. No unsupported marketing claims.

If Pass 1 retained raw shorthand jargon or failed to expand a brand initialism, correct the title to a clean, natural consumer title and mark NORMALIZED_FIXED.

Return only structured JSON. Preserve row_id exactly."""


class TitleDraft(BaseModel):
    row_id: int
    cleaned_title: str
    quality_flag: GenerationQuality
    evidence: list[str] = Field(default_factory=list)


class TitleAudit(BaseModel):
    row_id: int
    refined_title: str
    refinement_flag: AuditQuality
    review_required: bool
    audit_notes: str


class BatchTitleDraftOutput(BaseModel):
    results: list[TitleDraft]


class BatchTitleAuditOutput(BaseModel):
    results: list[TitleAudit]


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


def source(row: dict[str, Any]) -> dict[str, Any]:
    """Pure, general-purpose extraction with zero vendor or brand assumptions."""
    return {
        "row_id": int(row["Sr. no."]),
        "item_type": row.get("Item Type"),
        "product_type": row.get("Product Type"),
        "combined_description": row.get("Combined Description", ""),
        "secondary_product_description": row.get("Secondary Product Description", ""),
    }


def build_payload(chunk: list[dict[str, Any]], pass2: bool) -> list[dict[str, Any]]:
    if not pass2:
        return [source(row) for row in chunk]
    return [
        {
            **source(row),
            "universal_raw_context_audit_only": row.get("Universal_Raw_Context", ""),
            "draft_title": row.get("Title Pass 1", ""),
            "draft_quality": row.get("Title Pass 1 Quality", ""),
            "draft_evidence": row.get("Title Pass 1 Evidence", []),
        }
        for row in chunk
    ]


def client() -> Any:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for title generation")
    if genai is None:
        raise RuntimeError("google-genai is required for title generation")
    return genai.Client(api_key=key)


def direct_parallel_titles(rows: list[dict[str, Any]], pass2: bool, api: Any) -> dict[int, dict[str, Any]]:
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    schema_model = BatchTitleAuditOutput if pass2 else BatchTitleDraftOutput
    field = "refined_title" if pass2 else "cleaned_title"
    schema = inline_schema(schema_model)

    def _process_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = build_payload(chunk, pass2)
        response = api.models.generate_content(
            model=MODEL_NAME,
            contents=[f"{'Audit' if pass2 else 'Generate'} titles:\n{json.dumps(payload, ensure_ascii=False)}"],
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.0,
            },
        )
        raw = json.loads(response.text)
        for result in raw.get("results", []):
            result[field] = normalize_title(result.get(field, ""))
        return schema_model.model_validate(raw).results

    values: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_results in executor.map(_process_chunk, chunks):
            for res in chunk_results:
                values[res.row_id] = res.model_dump()
    return values


def batch(rows: list[dict[str, Any]], pass2: bool, api: Any | None = None, chunk_size: int = CHUNK_SIZE, poll_seconds: float = 15.0) -> tuple[list[dict[str, Any]], str]:
    stage = "TITLE_AUDIT" if pass2 else "TITLE_GENERATION"
    service = api or client()
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    schema_model = BatchTitleAuditOutput if pass2 else BatchTitleDraftOutput
    field = "refined_title" if pass2 else "cleaned_title"
    schema = inline_schema(schema_model)

    requests = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        payload = build_payload(chunk, pass2)
        requests.append(
            {
                "key": f"chunk-{start // chunk_size}",
                "request": {
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "temperature": 0.0,
                    },
                },
            }
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as file:
        for req in requests:
            file.write(json.dumps(req, ensure_ascii=False) + "\n")
        input_path = Path(file.name)

    try:
        uploaded = service.files.upload(
            file=str(input_path),
            config=types.UploadFileConfig(display_name=stage.casefold(), mime_type="jsonl"),
        )
        job = service.batches.create(model=MODEL_NAME, src=uploaded.name, config={"display_name": stage.casefold()})
        batch_id = job.name

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while getattr(job, "state", "") not in terminal:
            time.sleep(poll_seconds)
            job = service.batches.get(name=batch_id)

        if getattr(job, "state", "") != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"{stage} Batch failed: {getattr(job, 'error', '')}")

        res_file = getattr(getattr(job, "dest", None), "file_name", None)
        content = service.files.download(file=res_file).decode("utf-8")

        values: dict[int, dict[str, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            candidates = (item.get("response") or {}).get("candidates") or []
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    raw = json.loads(text)
                    for r in raw.get("results", []):
                        r[field] = normalize_title(r.get(field, ""))
                    parsed = schema_model.model_validate(raw)
                    for r in parsed.results:
                        values[r.row_id] = r.model_dump()

        expected = {int(row["Sr. no."]) for row in rows}
        missing = sorted(expected - values.keys())
        if missing:
            raise RuntimeError(f"{stage} reconciliation failed: missing={missing[:10]}")

        return [values[idx] for idx in sorted(values)], batch_id
    finally:
        input_path.unlink(missing_ok=True)


def run(file_id: str, api: Any | None = None, row_ids: list[str] | None = None, chunk_size: int = CHUNK_SIZE, poll_seconds: float = 15.0) -> list[dict[str, Any]]:
    rows = database.rows(file_id)
    selected = {str(v) for v in row_ids} if row_ids else None
    rows = [r for r in rows if selected is None or r["Sr. no."] in selected]
    if not rows:
        return []

    missing = [r["Sr. no."] for r in rows if not (r.get("Combined Description") or r.get("Secondary Product Description"))]
    if missing:
        raise RuntimeError(f"Separated product contexts required before title generation: {missing[:10]}")
    missing = [r["Sr. no."] for r in rows if not r.get("Product Type")]
    if missing:
        raise RuntimeError(f"Product Type is required before title generation: {missing[:10]}")

    logger.event("TITLE", "start", file_id=file_id, rows=len(rows))
    service = api or client()

    # Pass 1: Generation
    if len(rows) <= DIRECT_EXECUTION_THRESHOLD:
        drafts_map = direct_parallel_titles(rows, False, service)
        drafts = [drafts_map[int(r["Sr. no."])] for r in rows]
        pass1_batch_id = "direct-execution"
    else:
        drafts, pass1_batch_id = batch(rows, False, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)

    database.update_title_drafts(file_id, pass1_batch_id, drafts)
    logger.event("TITLE", "pass1_stored", file_id=file_id, rows=len(drafts))

    # Pass 2: Audit
    audited_rows = database.rows(file_id)
    audited_rows = [r for r in audited_rows if selected is None or r["Sr. no."] in selected]

    if len(audited_rows) <= DIRECT_EXECUTION_THRESHOLD:
        audits_map = direct_parallel_titles(audited_rows, True, service)
        audits = [audits_map[int(r["Sr. no."])] for r in audited_rows]
        pass2_batch_id = "direct-execution"
    else:
        audits, pass2_batch_id = batch(audited_rows, True, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)

    database.update_title_audits(file_id, pass2_batch_id, audits)
    logger.event("TITLE", "pass2_stored", file_id=file_id, rows=len(audits))
    return audits