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
import logger
from normalize import normalize_invoice, normalize_text

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 15
DIRECT_EXECUTION_THRESHOLD = 200

ContentQuality = Literal["VERIFIED_CLEAN", "NORMALIZED_FIXED", "SPARSE_INPUT", "UNGROUNDED_FLAG"]

GENERATION_SYSTEM = """You are a senior global commercial-content specialist for travel retail and duty-free catalogs.
Generate grounded commercial copy and commercial master data for every product.

Mandatory Master Fields (Must NEVER be 'NA'):
1. HSN Code:
   - MUST be an EXACT 6-DIGIT Harmonized System tariff code (e.g., 220830 for Whiskies, 220860 for Vodka, 220870 for Liqueurs, 220300 for Beer, 420330 for Belts, 420231 for Wallets, 910119 for Watches, 330300 for Perfume, 180632 for Chocolates, 240220 for Cigarettes).
   - If an 8-digit HSN is present in source, extract the first 6 digits. If missing, deduce the exact 6-digit category HSN. It must NEVER be 'NA'.

2. Country of Origin:
   - MUST be a valid Country name (never 'NA').
   - Use row text if stated ('Made in Italy', 'Swiss Made').
   - If not stated, use authoritative brand origin & distillery/manufacture heritage (e.g., Scotland/United Kingdom for Scotch Whisky, Ireland for Baileys, Mexico for Tequila/Corona, Belgium for Stella Artois, Switzerland for Swiss Watches, Germany for Montblanc writing instruments, France for Cognac/Champagne).

3. Brand & Parent Company:
   - Consumer-facing Brand (e.g., 'Don Julio', 'Baileys', 'Budweiser', 'The Singleton', 'Montblanc').
   - Global Parent Company (e.g., 'Diageo', 'AB InBev', 'Richemont', 'LVMH', 'Pernod Ricard').

4. Invoice Name:
   - Uppercase ASCII, max 33 characters. Include Brand, Line, Size/Volume cleanly.

5. Descriptions:
   - Short description (20-30 words).
   - Long description (~300 words) highlighting flavor/craftsmanship, tasting notes, or design specs.

Return only structured JSON and preserve row_id exactly."""

AUDIT_SYSTEM = """You are the independent commercial content auditor.
Audit Pass 1 content. Ensure:
1. HSN is an exact 6-digit code.
2. Country of Origin is a valid Country (NEVER 'NA').
3. Invoice Name is <= 33 characters, uppercase ASCII.
4. Descriptions are rich, accurate, and professional. Short description must be 20-30 words; trim it if longer.

Return corrected structured content. Preserve row_id exactly."""


class ContentFields(BaseModel):
    brand_name: str
    parent_company: str
    product_size: str
    marketplace_title: str
    invoice_name: str
    short_description: str
    long_description: str
    hsn_code: str
    country_of_origin: str
    evidence: list[str] = Field(default_factory=list)


class ContentDraft(BaseModel):
    row_id: int
    content: ContentFields


class ContentAudit(BaseModel):
    row_id: int
    content: ContentFields
    quality_flag: ContentQuality
    review_required: bool
    audit_notes: str


class BatchContentDraftOutput(BaseModel):
    items: list[ContentDraft]


class BatchContentAuditOutput(BaseModel):
    items: list[ContentAudit]


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def _clean(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                return _clean(defs.get(ref_key, {}))
            cleaned = {}
            for k, v in node.items():
                if k in {"additionalProperties", "$defs", "$schema", "title"}:
                    continue
                cleaned[k] = _clean(v)
            return cleaned
        elif isinstance(node, list):
            return [_clean(item) for item in node]
        return node

    return _clean(raw)


def source(row: dict[str, Any], pass2: bool, brand_candidates: list[str]) -> dict[str, Any]:
    value = {
        "row_id": int(row["Sr. no."]),
        "item_type": row.get("Item Type"),
        "product_type": row.get("Product Type"),
        "final_title": row.get("Title Pass 2") or row.get("Title Pass 1"),
        "attributes": row.get("Attributes Pass 2") or row.get("Attributes Pass 1") or {},
        "combined_description": row.get("Combined Description", ""),
        "secondary_product_description": row.get("Secondary Product Description", ""),
        "universal_raw_context": row.get("Universal_Raw_Context", ""),
    }
    if pass2:
        value["pass1_content"] = row.get("Content Pass 1", {})
        value["brand_candidates_in_file"] = brand_candidates
    return value


CATEGORY_FALLBACK_HSN = {
    "Whisky": "220830",
    "Vodka": "220860",
    "Gin": "220850",
    "Rum": "220840",
    "Tequila & Mezcal": "220890",
    "Liqueurs & Cordials": "220870",
    "Wine & Champagne": "220410",
    "Ready to Drink": "220300",
    "Beer": "220300",
    "Fragrance": "330300",
    "Makeup": "330499",
    "Personal Care": "330499",
    "Watches": "910119",
    "Apparel": "420330",
    "Accessories": "420231",
    "Writing Instruments": "960810",
    "Stationery & Gifts": "482010",
    "Chocolates": "180632",
    "Cigarettes": "240220",
    "Cigars": "240210",
}


def sanitize_hsn(value: object, product_type: str = "") -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) >= 6:
        return digits[:6]
    return CATEGORY_FALLBACK_HSN.get(product_type, "220890")


def normalize_content(value: dict[str, Any], product_type: str = "") -> dict[str, Any]:
    content = dict(value)
    for name in ("brand_name", "parent_company", "product_size", "marketplace_title", "short_description", "long_description"):
        content[name] = normalize_text(content.get(name, "")) or "NA"

    content["country_of_origin"] = normalize_text(content.get("country_of_origin", "")) or "Imported"
    if content["country_of_origin"].upper() in {"NA", "N/A", "UNKNOWN", "UNRESOLVED", ""}:
        content["country_of_origin"] = "Imported"

    content["invoice_name"] = normalize_invoice(content.get("invoice_name") or content.get("marketplace_title"))
    content["hsn_code"] = sanitize_hsn(content.get("hsn_code"), product_type)

    raw_ev = content.get("evidence")
    if isinstance(raw_ev, list):
        content["evidence"] = [str(x) for x in raw_ev if str(x).strip()]
    elif isinstance(raw_ev, dict):
        content["evidence"] = [f"{k}: {v}" for k, v in raw_ev.items()]
    else:
        content["evidence"] = []

    return ContentFields.model_validate(content).model_dump()


def unique_invoice(value: str, used: set[str], row_id: str) -> tuple[str, bool]:
    base = normalize_invoice(value) or normalize_invoice(f"ITEM ROW {row_id}")
    candidate = base
    if candidate not in used:
        used.add(candidate)
        return candidate, False
    counter = 1
    while True:
        suffix = f"-{counter:02d}"
        prefix = base[:33 - len(suffix)].rstrip()
        candidate = f"{prefix}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate, True
        counter += 1


def brand_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def client() -> Any:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for commercial content")
    if genai is None:
        raise RuntimeError("google-genai is required for commercial content")
    return genai.Client(api_key=key)


def direct_parallel_content(
    rows: list[dict[str, Any]],
    pass2: bool,
    brand_candidates: list[str],
    api: Any,
) -> dict[int, dict[str, Any]]:
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    schema_model = BatchContentAuditOutput if pass2 else BatchContentDraftOutput
    schema = inline_schema(schema_model)

    def _process_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = [source(r, pass2, brand_candidates) for r in chunk]
        pt_map = {int(r["Sr. no."]): str(r.get("Product Type") or "") for r in chunk}
        response = api.models.generate_content(
            model=MODEL_NAME,
            contents=[f"{'Audit' if pass2 else 'Generate'} commercial content:\n{json.dumps(payload, ensure_ascii=False)}"],
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.0,
            },
        )
        raw = json.loads(response.text)
        for item in raw.get("items", []):
            rid = int(item["row_id"])
            item["content"] = normalize_content(item.get("content", {}), pt_map.get(rid, ""))
        return schema_model.model_validate(raw).items

    values: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_results in executor.map(_process_chunk, chunks):
            for res in chunk_results:
                values[res.row_id] = res.model_dump()
    return values


def batch(
    rows: list[dict[str, Any]],
    pass2: bool,
    brand_candidates: list[str],
    api: Any | None = None,
    chunk_size: int = CHUNK_SIZE,
    poll_seconds: float = 15.0,
) -> tuple[list[dict[str, Any]], str]:
    stage = "CONTENT_AUDIT" if pass2 else "CONTENT_GENERATION"
    service = api or client()
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    schema_model = BatchContentAuditOutput if pass2 else BatchContentDraftOutput
    schema = inline_schema(schema_model)

    requests = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        payload = [source(r, pass2, brand_candidates) for r in chunk]
        requests.append({
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
        })

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

        pt_map = {int(r["Sr. no."]): str(r.get("Product Type") or "") for r in rows}
        result: dict[int, dict[str, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            val = json.loads(line)
            candidates = (val.get("response") or {}).get("candidates") or []
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    raw = json.loads(text)
                    for item in raw.get("items", []):
                        rid = int(item["row_id"])
                        item["content"] = normalize_content(item.get("content", {}), pt_map.get(rid, ""))
                    parsed = schema_model.model_validate(raw)
                    for item in parsed.items:
                        result[item.row_id] = item.model_dump()

        expected = {int(row["Sr. no."]) for row in rows}
        missing = sorted(expected - result.keys())
        if missing:
            raise RuntimeError(f"{stage} reconciliation failed: missing={missing[:10]}")

        return [result[idx] for idx in sorted(result)], batch_id
    finally:
        input_path.unlink(missing_ok=True)


def run(
    file_id: str,
    api: Any | None = None,
    row_ids: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    poll_seconds: float = 15.0,
    job_manager: Any | None = None,
) -> list[dict[str, Any]]:
    rows = database.rows(file_id)
    selected = {str(v) for v in row_ids} if row_ids else None
    rows = [r for r in rows if selected is None or r["Sr. no."] in selected]
    if not rows:
        return []

    service = api or client()
    total_rows = len(rows)
    
    def check_cancelled() -> bool:
        if job_manager and job_manager.is_cancelled(file_id):
            logger.event("CONTENT", "cancelled", file_id=file_id)
            return True
        return False
    
    def report_progress(processed: int, stage: str) -> None:
        if job_manager:
            progress = int((processed / total_rows) * 100)
            job_manager.set(
                file_id, "RUNNING", "CONTENT", progress,
                f"{stage}: {processed}/{total_rows}",
                progress_counter=f"{processed}/{total_rows}"
            )

    # Pass 1: Generation
    report_progress(0, "Pass 1")
    if check_cancelled():
        return []
    
    if len(rows) <= DIRECT_EXECUTION_THRESHOLD:
        drafts_map = direct_parallel_content(rows, False, [], service)
        drafts = [drafts_map[int(r["Sr. no."])] for r in rows]
        pass1_batch_id = "direct-execution"
    else:
        drafts, pass1_batch_id = batch(rows, False, [], api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)
        
        if job_manager:
            job_manager.set(file_id, "RUNNING", "CONTENT", 25, "Pass 1 batch submitted", batch_job_id=pass1_batch_id)
        
        while True:
            if check_cancelled():
                try:
                    service.batches.cancel(name=pass1_batch_id)
                    logger.event("CONTENT", "batch_cancelled", file_id=file_id, batch_id=pass1_batch_id)
                except Exception:
                    pass
                return []
            
            job = service.batches.get(name=pass1_batch_id)
            state = getattr(job, "state", "")
            if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                break
            time.sleep(poll_seconds)

    database.update_content_drafts(file_id, pass1_batch_id, drafts)
    report_progress(len(drafts), "Pass 1 complete")

    if check_cancelled():
        return []

    # Pass 2: Audit
    all_rows = database.rows(file_id)
    brand_candidates = sorted({
        str(r.get("Content Pass 1", {}).get("brand_name", ""))
        for r in all_rows if r.get("Content Pass 1", {}).get("brand_name")
    })
    audited_rows = [r for r in all_rows if selected is None or r["Sr. no."] in selected]
    report_progress(len(drafts), "Pass 2")

    if len(audited_rows) <= DIRECT_EXECUTION_THRESHOLD:
        audits_map = direct_parallel_content(audited_rows, True, brand_candidates, service)
        audits = [audits_map[int(r["Sr. no."])] for r in audited_rows]
        pass2_batch_id = "direct-execution"
    else:
        audits, pass2_batch_id = batch(audited_rows, True, brand_candidates, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)
        
        if job_manager:
            job_manager.set(file_id, "RUNNING", "CONTENT", 75, "Pass 2 batch submitted", batch_job_id=pass2_batch_id)
        
        while True:
            if check_cancelled():
                try:
                    service.batches.cancel(name=pass2_batch_id)
                    logger.event("CONTENT", "batch_cancelled", file_id=file_id, batch_id=pass2_batch_id)
                except Exception:
                    pass
                return []
            
            job = service.batches.get(name=pass2_batch_id)
            state = getattr(job, "state", "")
            if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                break
            time.sleep(poll_seconds)

    # Invoices uniqueness & Parent company consistency
    used = database.invoice_names(exclude_file_id=file_id)
    if selected is not None:
        used.update(str(r["Invoice Name"]) for r in all_rows if r["Sr. no."] not in selected and r.get("Invoice Name"))

    brand_parents: dict[str, str] = {}
    if selected is not None:
        for r in all_rows:
            if r["Sr. no."] in selected:
                continue
            existing = r.get("Content Pass 2", {})
            if existing.get("brand_name") and existing.get("parent_company"):
                brand_parents[brand_key(str(existing["brand_name"]))] = str(existing["parent_company"])

    for decision in audits:
        c = decision["content"]
        invoice, _ = unique_invoice(c["invoice_name"], used, str(decision["row_id"]))
        c["invoice_name"] = invoice

        bkey = brand_key(c["brand_name"])
        parent = c["parent_company"]
        if bkey in brand_parents and brand_parents[bkey] != parent:
            decision["review_required"] = True
            decision["quality_flag"] = "UNGROUNDED_FLAG"
            decision["audit_notes"] = f"Brand parent conflict: {brand_parents[bkey]} versus {parent}.".strip()
        else:
            brand_parents[bkey] = parent

    database.update_content_audits(file_id, pass2_batch_id, audits)
    report_progress(total_rows, "Complete")
    
    return audits