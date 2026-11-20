from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import database
import logger
import gemini
from attribute_schema import descriptions, load, names
from normalize import is_liquor_product, normalize_attribute

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 15
DIRECT_EXECUTION_THRESHOLD = 200

GENERATION_SYSTEM = """You are a senior master-data attribute specialist for travel-retail and FMCG products.
Extract and deduce all defined attributes for the supplied Product Type using the product title, context, and standard retail domain knowledge.

Guidelines:
1. Combine explicit text evidence with standard catalog intelligence:
   - For Spirits (Whisky, Rum, Tequila, Gin, Vodka): Deduce Whisky_Type (Single Malt / Blended / Bourbon), Region (Speyside / Islay / Highland / Jalisco), Volume_cl, and Standard ABV % when typical for the expression.
   - For Liqueurs & RTD: Deduce Liqueur_Type (Cream / Herbal / Fruit), Packaging_Type (Can / Bottle), and RTD_Type (Beer / Cocktail).
   - For Luxury Accessories & Apparel: Deduce Material (Leather for luxury SLGs / Sartorial / Soft Grain lines), Gender (Men / Women / Unisex), Dimensions_cm, Closure_Type.
   - For Watches: Deduce Movement_Type (Automatic for AU-AU / mechanical models), Gender, Case_Diameter_mm, Strap_Material.
   - For Writing Instruments: Deduce Product_Type (Ink Bottle, Fountain Pen, Ballpoint), Ink_Volume_ml, Color, Collection_Series.

2. Formatting Conventions:
   - Centiliters: format as Ncl (e.g., '100cl', '70cl', '35.5cl').
   - Milliliters, grams, mm: format with space (e.g., '50 ml', '100 g', '42 mm', '35 mm').
   - Percentages: include % symbol (e.g., '40%', '5%').

3. Fill every key with the most accurate deduced value. Use 'NA' ONLY when completely unknowable or not applicable. Keep all values as strings. Preserve row_id exactly."""

AUDIT_SYSTEM = """You are the independent product-attribute quality auditor.
Audit Pass 1 attributes against the Product Type schema and product context.
Ensure attributes are rich, formatted according to unit standards, and logically grounded.

Return corrected structured response. Preserve row_id exactly."""


def build_schema(names_list: list[str], pass2: bool) -> dict[str, Any]:
    attributes_props = {name: {"type": "string"} for name in names_list}
    evidence_props = {name: {"type": "array", "items": {"type": "string"}} for name in names_list}

    item = {
        "type": "object",
        "properties": {
            "row_id": {"type": "integer"},
            "attributes": {
                "type": "object",
                "properties": attributes_props,
                "required": names_list,
            },
            "evidence": {
                "type": "object",
                "properties": evidence_props,
                "required": names_list,
            },
        },
        "required": ["row_id", "attributes", "evidence"],
    }

    if pass2:
        item["properties"].update(
            {
                "quality_flag": {
                    "type": "string",
                    "enum": ["VERIFIED_CLEAN", "NORMALIZED_FIXED", "SPARSE_INPUT", "UNGROUNDED_FLAG"],
                },
                "review_required": {"type": "boolean"},
                "audit_notes": {"type": "string"},
            }
        )
        item["required"].extend(["quality_flag", "review_required", "audit_notes"])

    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": item,
            }
        },
        "required": ["items"],
    }


def source(row: dict[str, Any], pass2: bool, attribute_descriptions: dict[str, str]) -> dict[str, Any]:
    value = {
        "row_id": int(row["Sr. no."]),
        "item_type": row.get("Item Type"),
        "product_type": row.get("Product Type"),
        "final_title": row.get("Title Pass 2") or row.get("Title Pass 1"),
        "combined_description": row.get("Combined Description", ""),
        "secondary_product_description": row.get("Secondary Product Description", ""),
        "universal_raw_context": row.get("Universal_Raw_Context", ""),
        "attribute_definitions": attribute_descriptions,
    }
    if pass2:
        value["pass1_attributes"] = row.get("Attributes Pass 1", {})
        value["pass1_evidence"] = row.get("Attributes Pass 1 Evidence", {})
    return value


def client() -> Any:
    return gemini.build_client("Attribute generation")


def process_item_attributes(
    value: dict[str, Any],
    category_names: list[str],
    product_type: str = "",
) -> tuple[dict[str, str], dict[str, list[str]]]:
    raw_attributes = value.get("attributes") or {}
    raw_evidence = value.get("evidence") or {}
    normalized = {
        name: normalize_attribute(
            name,
            raw_attributes.get(name, "NA"),
            liquor=is_liquor_product(product_type),
        )
        for name in category_names
    }
    evidence = {name: [str(item) for item in raw_evidence.get(name, [])] for name in category_names}
    return normalized, evidence


def direct_parallel_attributes(
    rows: list[dict[str, Any]],
    schema_map: dict[str, Any],
    pass2: bool,
    api: Any,
    on_progress: Any | None = None,
) -> dict[int, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["Product Type"]), []).append(row)

    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    tasks = []

    for category, category_rows in groups.items():
        category_names = names(schema_map, category)
        category_descriptions = descriptions(schema_map, category)
        schema = build_schema(category_names, pass2)

        for start in range(0, len(category_rows), CHUNK_SIZE):
            chunk = category_rows[start : start + CHUNK_SIZE]
            payload = [source(r, pass2, category_descriptions) for r in chunk]
            tasks.append((category, category_names, schema, payload))

    def _execute_task(task_args: tuple[str, list[str], dict[str, Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        category, category_names, schema, payload = task_args
        response = api.models.generate_content(
            model=MODEL_NAME,
            contents=[f"Extract attributes for {category}:\n{json.dumps(payload, ensure_ascii=False)}"],
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.0,
            },
        )
        data = json.loads(response.text)
        results = []
        for item in data.get("items", []):
            row_id = int(item["row_id"])
            norm, ev = process_item_attributes(item, category_names, category)
            res = {"row_id": str(row_id), "attributes": norm, "evidence": ev}
            if pass2:
                res.update({
                    "quality_flag": item.get("quality_flag", "VERIFIED_CLEAN"),
                    "review_required": bool(item.get("review_required", False)),
                    "audit_notes": item.get("audit_notes", ""),
                })
            results.append(res)
        return results

    values: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_res in executor.map(_execute_task, tasks):
            for res in chunk_res:
                values[int(res["row_id"])] = res
            if on_progress:
                on_progress(len(chunk_res))
    return values


def batch(
    rows: list[dict[str, Any]],
    schema_map: dict[str, Any],
    pass2: bool,
    api: Any | None = None,
    chunk_size: int = CHUNK_SIZE,
    poll_seconds: float = 15.0,
) -> tuple[list[dict[str, Any]], str]:
    stage = "ATTRIBUTE_AUDIT" if pass2 else "ATTRIBUTE_GENERATION"
    service = api or client()
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["Product Type"]), []).append(row)

    requests = []
    key_categories = {}

    for category, category_rows in groups.items():
        category_names = names(schema_map, category)
        category_descriptions = descriptions(schema_map, category)
        schema = build_schema(category_names, pass2)

        for start in range(0, len(category_rows), chunk_size):
            chunk = category_rows[start : start + chunk_size]
            key = f"{category}-{start // chunk_size}"
            key_categories[key] = category
            requests.append(
                {
                    "key": key,
                    "request": {
                        "system_instruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": json.dumps([source(r, pass2, category_descriptions) for r in chunk], ensure_ascii=False)}]}],
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
        row_product_types = {
            int(row["Sr. no."]): str(row.get("Product Type") or "")
            for row in rows
        }

        result: dict[int, dict[str, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            val = json.loads(line)
            key = val.get("key", "")
            if key not in key_categories:
                continue

            category = key_categories[key]
            category_names = names(schema_map, category)
            candidates = (val.get("response") or {}).get("candidates") or []
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    payload = json.loads(text)
                    for item in payload.get("items", []):
                        row_id = int(item["row_id"])
                        norm, ev = process_item_attributes(item, category_names, row_product_types[row_id])
                        res = {"row_id": str(row_id), "attributes": norm, "evidence": ev}
                        if pass2:
                            res.update({
                                "quality_flag": item.get("quality_flag", "VERIFIED_CLEAN"),
                                "review_required": bool(item.get("review_required", False)),
                                "audit_notes": item.get("audit_notes", ""),
                            })
                        result[row_id] = res

        expected = {int(row["Sr. no."]) for row in rows}
        missing = sorted(expected - result.keys())
        if missing:
            raise RuntimeError(f"{stage} row reconciliation failed: missing={missing[:10]}")

        return [result[idx] for idx in sorted(result)], batch_id
    finally:
        input_path.unlink(missing_ok=True)


def run(
    file_id: str,
    schema_path: str | Path = "attributes.json",
    api: Any | None = None,
    row_ids: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    poll_seconds: float = 15.0,
    job_manager: Any | None = None,
) -> list[dict[str, Any]]:
    schema_map = load(schema_path)
    rows = database.rows(file_id)
    selected = {str(value) for value in row_ids} if row_ids else None
    rows = [row for row in rows if selected is None or row["Sr. no."] in selected]
    if not rows:
        return []

    service = api or client()
    total_rows = len(rows)
    
    def check_cancelled() -> bool:
        if job_manager and job_manager.is_cancelled(file_id):
            logger.event("ATTRIBUTES", "cancelled", file_id=file_id)
            return True
        return False
    
    def report_progress(processed: int, stage: str) -> None:
        if job_manager:
            span = max(1, total_rows * 2)
            progress = int((processed / span) * 100)
            shown = min(processed, total_rows) if processed <= total_rows else min(processed - total_rows, total_rows)
            job_manager.set(
                file_id, "RUNNING", "ATTRIBUTES", progress,
                f"{stage}: {shown}/{total_rows}",
                progress_counter=f"{shown}/{total_rows}"
            )

    _done = {"n": 0}

    def _chunk_progress(count: int, stage: str) -> None:
        _done["n"] += count
        report_progress(_done["n"], stage)

    report_progress(0, "Pass 1")
    if check_cancelled():
        return []
    
    if len(rows) <= DIRECT_EXECUTION_THRESHOLD:
        drafts_map = direct_parallel_attributes(rows, schema_map, False, service, on_progress=lambda c: _chunk_progress(c, "Pass 1"))
        drafts = [drafts_map[int(r["Sr. no."])] for r in rows]
        pass1_batch_id = "direct-execution"
    else:
        drafts, pass1_batch_id = batch(rows, schema_map, False, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)
        
        if job_manager:
            job_manager.set(file_id, "RUNNING", "ATTRIBUTES", 25, "Pass 1 batch submitted", batch_job_id=pass1_batch_id)
        
        while True:
            if check_cancelled():
                try:
                    service.batches.cancel(name=pass1_batch_id)
                    logger.event("ATTRIBUTES", "batch_cancelled", file_id=file_id, batch_id=pass1_batch_id)
                except Exception:
                    pass
                return []
            
            job = service.batches.get(name=pass1_batch_id)
            state = getattr(job, "state", "")
            if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                break
            time.sleep(poll_seconds)

    database.update_attribute_drafts(file_id, pass1_batch_id, drafts)
    _done["n"] = total_rows
    report_progress(_done["n"], "Pass 1 complete")

    if check_cancelled():
        return []

    # Pass 2: Audit
    audited_rows = database.rows(file_id)
    audited_rows = [row for row in audited_rows if selected is None or row["Sr. no."] in selected]
    report_progress(_done["n"], "Pass 2")

    if len(audited_rows) <= DIRECT_EXECUTION_THRESHOLD:
        audits_map = direct_parallel_attributes(audited_rows, schema_map, True, service, on_progress=lambda c: _chunk_progress(c, "Pass 2"))
        audits = [audits_map[int(r["Sr. no."])] for r in audited_rows]
        pass2_batch_id = "direct-execution"
    else:
        audits, pass2_batch_id = batch(audited_rows, schema_map, True, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)
        
        if job_manager:
            job_manager.set(file_id, "RUNNING", "ATTRIBUTES", 75, "Pass 2 batch submitted", batch_job_id=pass2_batch_id)
        
        while True:
            if check_cancelled():
                try:
                    service.batches.cancel(name=pass2_batch_id)
                    logger.event("ATTRIBUTES", "batch_cancelled", file_id=file_id, batch_id=pass2_batch_id)
                except Exception:
                    pass
                return []
            
            job = service.batches.get(name=pass2_batch_id)
            state = getattr(job, "state", "")
            if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                break
            time.sleep(poll_seconds)

    database.update_attribute_audits(file_id, pass2_batch_id, audits)
    report_progress(total_rows * 2, "Complete")
    
    return audits