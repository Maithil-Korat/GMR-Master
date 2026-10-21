from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import pandas as pd

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field as PydanticField

import attribute
import content
import context
import database
import export
import ingest
import item_number
import item_type
import logger
import product_type
import taxonomy_v2
import title
import fields

if not os.environ.get("GEMINI_API_KEY"):
    logger.event("WARNING", "missing_env_key", message="GEMINI_API_KEY is not set in environment or .env file")

StageType = Literal[
    "FIELD_MAP", "ITEM_TYPE", "PRODUCT_TYPE", "TITLE",
    "ATTRIBUTES", "CONTENT", "TAXONOMY", "UNICODE", "AUTO_PILOT"
]


class StageRunRequest(BaseModel):
    stage: StageType
    check_master: bool = False
    selected_row_ids: list[str] = PydanticField(default_factory=list)
    operator_hint: str = ""


class RowPatchRequest(BaseModel):
    cleaned_title: str | None = None
    brand_name: str | None = None
    parent_company: str | None = None
    invoice_name: str | None = None
    product_type: str | None = None
    item_type: str | None = None
    gmr_taxonomy: str | None = None
    hsn_code: str | None = None
    country_of_origin: str | None = None
    
class ColumnMappingUpdate(BaseModel):
    mappings: list[dict[str, Any]]

class SmartPropagateRequest(BaseModel):
    field_name: str
    target_value: str
    matching_text: str
    match_source_field: str = "Universal_Raw_Context"


@dataclass
class JobState:
    status: str = "IDLE"
    active_stage: str = "INGEST"
    progress_percent: int = 100
    progress_counter: str = ""
    message: str = "Ready"
    error: str = ""
    error_stage: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_data: dict[str, Any] = field(default_factory=dict)
    batch_job_ids: dict[str, str] = field(default_factory=dict)

    def update(
        self,
        status: str,
        stage: str,
        progress: int,
        message: str,
        error: str = "",
        progress_counter: str = "",
        metrics: dict[str, Any] | None = None,
        checkpoint_data: dict[str, Any] | None = None,
        batch_job_id: str | None = None,
    ) -> None:
        self.status = status
        self.active_stage = stage
        self.progress_percent = progress
        self.progress_counter = progress_counter
        self.message = message
        self.error = error
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if metrics is not None:
            self.metrics = metrics
        if checkpoint_data is not None:
            self.checkpoint_data = checkpoint_data
        if batch_job_id is not None:
            self.batch_job_ids[stage] = batch_job_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "active_stage": self.active_stage,
            "progress_percent": self.progress_percent,
            "progress_counter": self.progress_counter,
            "message": self.message,
            "error": self.error,
            "error_stage": self.error_stage,
            "updated_at": self.updated_at,
            "metrics": self.metrics,
            "checkpoint_data": self.checkpoint_data,
        }


class PipelineJobManager:
    STAGE_WEIGHTS = {
        "FIELD_MAP": 1, "ITEM_TYPE": 1, "PRODUCT_TYPE": 1,
        "TITLE": 4, "ATTRIBUTES": 4, "CONTENT": 4, "TAXONOMY": 4,
        "UNICODE": 1,
    }

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._cancelled: set[str] = set()
        self._plans: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def begin_plan(self, file_id: str, stages: list[str]) -> None:
        with self._lock:
            self._plans[file_id] = stages
            self._jobs.setdefault(file_id, JobState()).progress_percent = 0

    def get(self, file_id: str) -> JobState:
        with self._lock:
            if file_id not in self._jobs:
                self._jobs[file_id] = JobState()
            return self._jobs[file_id]

    def set(
        self,
        file_id: str,
        status: str,
        stage: str,
        progress: int,
        message: str,
        error: str = "",
        progress_counter: str = "",
        metrics: dict[str, Any] | None = None,
        checkpoint_data: dict[str, Any] | None = None,
        batch_job_id: str | None = None,
    ) -> None:
        with self._lock:
            if file_id not in self._jobs:
                self._jobs[file_id] = JobState()

            plan = self._plans.get(file_id)
            if status == "RUNNING" and plan and stage in plan:
                w = self.STAGE_WEIGHTS
                total = sum(w.get(s, 1) for s in plan)
                start = sum(w.get(s, 1) for s in plan[:plan.index(stage)])
                local = max(0, min(100, progress)) / 100
                mapped = int((start + local * w.get(stage, 1)) / total * 100)
                progress = max(mapped, self._jobs[file_id].progress_percent)

            self._jobs[file_id].update(
                status, stage, progress, message, error, progress_counter, metrics, checkpoint_data, batch_job_id
            )

    def is_running(self, file_id: str) -> bool:
        with self._lock:
            return self._jobs.get(file_id, JobState()).status == "RUNNING"

    def fail(self, file_id: str, stage: str, message: str) -> None:
        with self._lock:
            job = self._jobs.setdefault(file_id, JobState())
            self._plans.pop(file_id, None)
            job.status = "ERROR"
            job.error_stage = stage
            job.error = message
            job.message = message
            job.updated_at = datetime.now(timezone.utc).isoformat()

    def cancel(self, file_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(file_id)
            if job and job.status == "RUNNING":
                self._cancelled.add(file_id)
                return True
            return False

    def is_cancelled(self, file_id: str) -> bool:
        with self._lock:
            return file_id in self._cancelled

    def clear_cancel(self, file_id: str) -> None:
        with self._lock:
            self._cancelled.discard(file_id)

    def get_batch_job_id(self, file_id: str, stage: str) -> str | None:
        with self._lock:
            job = self._jobs.get(file_id)
            if job:
                return job.batch_job_ids.get(stage)
            return None


class RowQueryService:
    @staticmethod
    def get_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        clean = 0
        review = 0
        blocked = 0
        dupes = 0
        unmapped_base = 0
        parser_conflict = 0
        ungrounded = 0

        for r in rows:
            is_rev = bool(r.get("Title Review Required") or r.get("Attributes Review Required") or r.get("Taxonomy Review Required"))
            base_code = r.get("2 Digit Base Code")
            is_blk = r.get("item_number") == "BLOCKED" or base_code in {"UNMAPPED", None, ""}
            is_dup = r.get("item_number_source") == "INTRA_FILE_REUSE"

            if is_blk:
                blocked += 1
            if is_dup:
                dupes += 1
            if is_rev:
                review += 1
            if not is_rev and not is_blk and r.get("item_number"):
                clean += 1

            if base_code in {"UNMAPPED", None, ""}:
                unmapped_base += 1
            if "Parser/Gemini conflict" in str(r.get("Item Type Notes", "")):
                parser_conflict += 1
            if r.get("Content Quality Flag") == "UNGROUNDED_FLAG":
                ungrounded += 1

        return {
            "total": len(rows),
            "clean": clean,
            "review_required": review,
            "blocked": blocked,
            "duplicates": dupes,
            "issue_clusters": {
                "unmapped_base_code": unmapped_base,
                "parser_conflict": parser_conflict,
                "ungrounded_content": ungrounded,
            },
        }

    @staticmethod
    def filter_and_paginate(
        rows: list[dict[str, Any]],
        page: int,
        limit: int,
        status_filter: str,
        cluster_filter: str | None,
        search: str,
    ) -> tuple[int, list[dict[str, Any]]]:
        search_term = search.strip().casefold()
        filtered = []

        for r in rows:
            is_rev = bool(r.get("Title Review Required") or r.get("Attributes Review Required") or r.get("Taxonomy Review Required"))
            base_code = r.get("2 Digit Base Code")
            is_blk = r.get("item_number") == "BLOCKED" or base_code in {"UNMAPPED", None, ""}
            is_dup = r.get("item_number_source") == "INTRA_FILE_REUSE"

            if status_filter == "REVIEW_REQUIRED" and not is_rev:
                continue
            if status_filter == "BLOCKED" and not is_blk:
                continue
            if status_filter == "DUPLICATES" and not is_dup:
                continue
            if status_filter == "CLEAN" and (is_rev or is_blk):
                continue

            if cluster_filter == "unmapped_base_code" and base_code not in {"UNMAPPED", None, ""}:
                continue
            if cluster_filter == "parser_conflict" and "Parser/Gemini conflict" not in str(r.get("Item Type Notes", "")):
                continue
            if cluster_filter == "ungrounded_content" and r.get("Content Quality Flag") != "UNGROUNDED_FLAG":
                continue

            if search_term:
                haystack = (
                    f"{r.get('Title Pass 2', '')} {r.get('Title Pass 1', '')} "
                    f"{r.get('Universal_Raw_Context', '')} {r.get('Invoice Name', '')} "
                    f"{r.get('item_number', '')} {r.get('GMR Taxonomy', '')}"
                ).casefold()
                if search_term not in haystack:
                    continue

            badge = "CLEAN"
            if is_blk:
                badge = "BLOCKED"
            elif is_dup:
                badge = "DUPLICATE"
            elif is_rev:
                badge = "REVIEW"

            content_p2 = r.get("Content Pass 2") or {}
            attrs = r.get("Attributes Pass 2") or r.get("Attributes Pass 1") or {}

            filtered.append({
                "sr_no": r["Sr. no."],
                "status_badge": badge,
                "universal_raw_context": r.get("Universal_Raw_Context", ""),
                "item_type": r.get("Item Type", ""),
                "product_type": r.get("Product Type", ""),
                "cleaned_title": r.get("Title Pass 2") or r.get("Title Pass 1") or "",
                "brand_name": content_p2.get("brand_name") or "",
                "parent_company": content_p2.get("parent_company") or "",
                "invoice_name": r.get("Invoice Name", ""),
                "gmr_taxonomy": r.get("GMR Taxonomy", ""),
                "base_code": base_code or "",
                "item_number": r.get("item_number", ""),
                "item_number_source": r.get("item_number_source", ""),
                "review_required": is_rev,
                "audit_notes": f"{r.get('Title Audit Notes', '')} {r.get('Taxonomy Audit Notes', '')}".strip(),
                "attributes": attrs,
            })

        total_records = len(filtered)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        return total_records, filtered[start_idx:end_idx]


class PipelineWorker:
    def __init__(self, manager: PipelineJobManager) -> None:
        self.manager = manager

    def _compute_metrics(self, file_id: str) -> dict[str, Any]:
        rows = database.rows(file_id)
        return RowQueryService.get_metrics(rows)

    def run(self, file_id: str, req: StageRunRequest) -> None:
        row_ids = req.selected_row_ids if req.selected_row_ids else None
        steps = {
            "FIELD_MAP": ("Mapping semantic fields...", lambda: context.prepare(file_id)),
            "ITEM_TYPE": ("Classifying item types...", lambda: item_type.classify_file(file_id)),
            "PRODUCT_TYPE": ("Classifying product categories...", lambda: product_type.classify_file(file_id)),
            "TITLE": ("Generating cleaned titles...", lambda: title.run(file_id, row_ids=row_ids, job_manager=self.manager)),
            "ATTRIBUTES": ("Extracting attributes...", lambda: attribute.run(file_id, row_ids=row_ids, job_manager=self.manager)),
            "CONTENT": ("Generating commercial content...", lambda: content.run(file_id, row_ids=row_ids, job_manager=self.manager)),
            "TAXONOMY": ("Resolving GMR taxonomy...", lambda: taxonomy_v2.run(file_id, row_ids=row_ids, job_manager=self.manager)),
            "UNICODE": ("Reconciling item numbers...", lambda: item_number.assign_unicodes(file_id, check_master=req.check_master)),
        }
        plan = list(steps) if req.stage == "AUTO_PILOT" else [req.stage]
        current = plan[0]
        completed: list[str] = []

        try:
            self.manager.clear_cancel(file_id)

            initial_metrics = self._compute_metrics(file_id)
            total_rows = initial_metrics.get("total", 0)
            self.manager.set(
                file_id, "RUNNING", req.stage, 5,
                f"Initializing {req.stage}...",
                progress_counter=f"0/{total_rows}",
                metrics=initial_metrics,
            )

            self.manager.begin_plan(file_id, plan)

            for st_name in plan:
                current = st_name
                if self.manager.is_cancelled(file_id):
                    self.manager.set(file_id, "CANCELLED", st_name, 0, "Job cancelled by user")
                    return
                msg, action = steps[st_name]
                self.manager.set(file_id, "RUNNING", st_name, 0, msg)
                action()
                completed.append(st_name)

            final_metrics = self._compute_metrics(file_id)
            self.manager.set(
                file_id, "COMPLETED", req.stage, 100,
                f"{req.stage} completed successfully.",
                metrics=final_metrics,
            )
        except Exception as exc:
            note = f" Completed: {', '.join(completed)}." if completed else ""
            message = f"{current} failed — {type(exc).__name__}: {exc}.{note}"
            logger.event(
                "API_PIPELINE_ERROR", "failure",
                file_id=file_id, stage=current, error_type=type(exc).__name__, error=str(exc),
            )
            self.manager.fail(file_id, current, message)


job_manager = PipelineJobManager()
worker = PipelineWorker(job_manager)

app = FastAPI(title="GMR Duty-Free Master Catalog API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls", ".xlsb"}:
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls, .xlsb) are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        res = await asyncio.to_thread(ingest.ingest, tmp_path)
        job_manager.set(res.file_id, "IDLE", "INGEST", 100, "Workbook ingested.")
        return {
            "file_id": res.file_id,
            "file_name": res.file_name,
            "total_rows": res.row_count,
            "columns": list(res.columns),
        }
    finally:
        tmp_path.unlink(missing_ok=True)
        
class PasteIngestRequest(BaseModel):
    raw_text: str
    file_name: str = "Pasted_Catalog_Data.xlsx"


@app.post("/api/files/paste")
async def paste_catalog_text(req: PasteIngestRequest):
    if not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="Pasted text cannot be empty.")

    try:
        res = await asyncio.to_thread(ingest.ingest_text, req.raw_text, req.file_name)
        job_manager.set(res.file_id, "IDLE", "INGEST", 100, "Pasted catalog parsed.")
        return {
            "file_id": res.file_id,
            "file_name": res.file_name,
            "total_rows": res.row_count,
            "columns": list(res.columns),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse pasted text: {str(exc)}")
    
@app.post("/api/files/{file_id}/sync-excel")
async def sync_excel_edits(file_id: str, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls", ".xlsb"}:
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls, .xlsb) are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    def _process_sync() -> tuple[int, list[str]]:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(tmp_path, dtype=str, engine="openpyxl" if suffix == ".xlsx" else "xlrd").fillna("")
        else:
            df = pd.read_excel(tmp_path, dtype=str, engine="pyxlsb").fillna("")

        # Normalize column lookup mapping
        col_map = {str(c).strip().casefold(): str(c) for c in df.columns}

        def get_col(*aliases: str) -> str | None:
            for alias in aliases:
                if alias.casefold() in col_map:
                    return col_map[alias.casefold()]
            return None

        sr_col = get_col("Sr. no.", "Sr. No.", "Sr No", "sr_no", "#")
        title_col = get_col("Cleaned Titles", "Cleaned Title", "Title")
        brand_col = get_col("Brand Name", "Brand")
        parent_col = get_col("Parent Company", "Parent")
        invoice_col = get_col("Invoice Name")
        size_col = get_col("Product Size", "Size")
        short_desc_col = get_col("Short Description")
        long_desc_col = get_col("Long Description")
        hsn_col = get_col("HSN Code", "HSN")
        country_col = get_col("Country of Origin", "Country")
        pt_col = get_col("Product Type")
        it_col = get_col("Item Type")
        tax_col = get_col("GMR Taxonomy", "Taxonomy")
        item_no_col = get_col("Generated Item Number", "Item Number", "Item No")
        attrs_given_col = get_col("Attributes Given")

        # Detect Attribute 1 through 10 columns
        attr_cols = [get_col(f"Attribute {i}", f"Attr {i}", f"Attr_{i}") for i in range(1, 11)]

        detected_fields = [
            name for name, col in [
                ("Titles", title_col), ("Brand", brand_col), ("Parent", parent_col),
                ("Invoice", invoice_col), ("HSN", hsn_col), ("Country", country_col),
                ("Taxonomy", tax_col), ("Product Type", pt_col), ("Item Type", it_col),
                ("Item Numbers", item_no_col)
            ] if col is not None
        ]

        updated_count = 0
        with database.connect() as conn:
            database.create_table(conn)

            for idx, row in df.iterrows():
                sr_no_val = str(row[sr_col]).strip() if sr_col else str(idx + 1)
                if not sr_no_val:
                    continue

                # Fetch existing row JSON state
                db_record = conn.execute(
                    """
                    SELECT content_pass2_json, attributes_pass2_json, gmr_taxonomy 
                    FROM raw_rows 
                    WHERE file_id = ? AND sr_no = ?
                    """,
                    (file_id, sr_no_val)
                ).fetchone()

                if not db_record:
                    continue

                content_p2 = json.loads(db_record["content_pass2_json"] or "{}")
                attrs_p2 = json.loads(db_record["attributes_pass2_json"] or "{}")

                # 1. Update Core Content Fields
                if brand_col and str(row[brand_col]).strip():
                    content_p2["brand_name"] = str(row[brand_col]).strip()
                if parent_col and str(row[parent_col]).strip():
                    content_p2["parent_company"] = str(row[parent_col]).strip()
                if size_col and str(row[size_col]).strip():
                    content_p2["product_size"] = str(row[size_col]).strip()
                if short_desc_col and str(row[short_desc_col]).strip():
                    content_p2["short_description"] = str(row[short_desc_col]).strip()
                if long_desc_col and str(row[long_desc_col]).strip():
                    content_p2["long_description"] = str(row[long_desc_col]).strip()
                if hsn_col and str(row[hsn_col]).strip():
                    content_p2["hsn_code"] = str(row[hsn_col]).strip()
                if country_col and str(row[country_col]).strip():
                    content_p2["country_of_origin"] = str(row[country_col]).strip()
                if invoice_col and str(row[invoice_col]).strip():
                    content_p2["invoice_name"] = str(row[invoice_col]).strip().upper()[:33]

                # 2. Update Attributes (from Attribute 1..10 Key:Value format or Attributes Given JSON)
                if attrs_given_col and str(row[attrs_given_col]).strip().startswith("{"):
                    try:
                        attrs_p2.update(json.loads(str(row[attrs_given_col]).strip()))
                    except Exception:
                        pass

                for ac in attr_cols:
                    if ac and str(row[ac]).strip() and ":" in str(row[ac]):
                        parts = str(row[ac]).split(":", 1)
                        k = parts[0].strip()
                        v = parts[1].strip()
                        if k and v:
                            attrs_p2[k] = v

                # 3. Prepare Database SQL Updates
                sql_updates = [
                    "content_pass2_json = ?",
                    "attributes_pass2_json = ?",
                    "title_review_required = 0",
                    "taxonomy_review_required = 0",
                    "attributes_review_required = 0",
                    "title_audit_notes = 'Manual bulk edit imported from Excel'"
                ]
                params = [json.dumps(content_p2, ensure_ascii=False), json.dumps(attrs_p2, ensure_ascii=False)]

                if title_col and str(row[title_col]).strip():
                    sql_updates.append("title_pass2 = ?")
                    params.append(str(row[title_col]).strip())

                if pt_col and str(row[pt_col]).strip():
                    sql_updates.append("product_type = ?")
                    params.append(str(row[pt_col]).strip())

                if it_col and str(row[it_col]).strip():
                    sql_updates.append("item_type = ?")
                    params.append(str(row[it_col]).strip())

                if invoice_col and str(row[invoice_col]).strip():
                    sql_updates.append("invoice_name = ?")
                    params.append(str(row[invoice_col]).strip().upper()[:33])

                if item_no_col and str(row[item_no_col]).strip():
                    sql_updates.append("item_number = ?")
                    sql_updates.append("item_number_source = 'MANUAL_EXCEL_SYNC'")
                    params.extend([str(row[item_no_col]).strip().zfill(10)])

                # 4. Taxonomy Re-Decomposition
                if tax_col and str(row[tax_col]).strip():
                    new_tax = str(row[tax_col]).strip()
                    levels = taxonomy_v2.decompose_taxonomy(new_tax)
                    base_code = taxonomy_v2.lookup_base_code(levels["level1"], levels["level2"])
                    sql_updates.extend(["gmr_taxonomy = ?", "base_code_2digit = ?", "level1 = ?", "level2 = ?", "level3 = ?", "level4 = ?", "level5 = ?"])
                    params.extend([new_tax, base_code, levels["level1"], levels["level2"], levels["level3"], levels["level4"], levels["level5"]])

                params.extend([file_id, sr_no_val])
                conn.execute(f"UPDATE raw_rows SET {', '.join(sql_updates)} WHERE file_id = ? AND sr_no = ?", params)
                updated_count += 1

        return updated_count, detected_fields

    try:
        count, fields_updated = await asyncio.to_thread(_process_sync)
        logger.event("EXCEL_SYNC", "completed", file_id=file_id, rows=count, fields=fields_updated)
        return {
            "message": f"Successfully updated {count} rows across {len(fields_updated)} fields ({', '.join(fields_updated)}).",
            "updated_rows": count,
            "fields_synced": fields_updated
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to sync Excel file: {str(exc)}")
    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/api/files/{file_id}/status")
async def get_status(file_id: str) -> dict[str, Any]:
    job = job_manager.get(file_id)
    
    if job.status == "IDLE" and not job.metrics:
        rows = await asyncio.to_thread(database.rows, file_id)
        if not rows:
            raise HTTPException(status_code=404, detail="File ID not found.")
        metrics = RowQueryService.get_metrics(rows)
        job_manager.set(file_id, "IDLE", job.active_stage, job.progress_percent, job.message, metrics=metrics)
        job = job_manager.get(file_id)
    elif not job.metrics:
        rows = await asyncio.to_thread(database.rows, file_id)
        if not rows:
            raise HTTPException(status_code=404, detail="File ID not found.")
        metrics = RowQueryService.get_metrics(rows)
        job.metrics = metrics
    
    return {
        "file_id": file_id,
        "job": job.to_dict(),
        "metrics": job.metrics,
    }


@app.get("/api/files/{file_id}/rows")
async def get_rows(
    file_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=250),
    status_filter: str = Query("ALL"),
    cluster_filter: str | None = Query(None),
    search: str = Query(""),
) -> dict[str, Any]:
    rows = await asyncio.to_thread(database.rows, file_id)
    if not rows:
        raise HTTPException(status_code=404, detail="File ID not found.")

    total, paginated = RowQueryService.filter_and_paginate(
        rows, page, limit, status_filter, cluster_filter, search
    )
    return {
        "total_records": total,
        "page": page,
        "limit": limit,
        "rows": paginated,
    }


@app.post("/api/files/{file_id}/stages/run", status_code=202)
async def run_stage(file_id: str, req: StageRunRequest, bg_tasks: BackgroundTasks) -> dict[str, Any]:
    if job_manager.is_running(file_id):
        raise HTTPException(status_code=409, detail="A pipeline task is currently in progress. Please wait.")

    rows = await asyncio.to_thread(database.rows, file_id)
    if not rows:
        raise HTTPException(status_code=404, detail="File ID not found.")

    job_manager.set(file_id, "RUNNING", req.stage, 5, f"Initializing {req.stage}...")

    bg_tasks.add_task(worker.run, file_id, req)
    return {"message": f"Stage {req.stage} initiated.", "file_id": file_id}


@app.post("/api/files/{file_id}/cancel")
async def cancel_job(file_id: str) -> dict[str, Any]:
    if not job_manager.is_running(file_id):
        raise HTTPException(status_code=400, detail="No running job to cancel.")
    
    job = job_manager.get(file_id)
    batch_job_id = job_manager.get_batch_job_id(file_id, job.active_stage)
    
    if batch_job_id and batch_job_id != "direct-execution":
        try:
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if key:
                from google import genai
                client = genai.Client(api_key=key)
                client.batches.cancel(name=batch_job_id)
                logger.event("API_CANCEL", "batch_cancelled", file_id=file_id, batch_id=batch_job_id)
        except Exception as e:
            logger.event("API_CANCEL", "batch_cancel_failed", file_id=file_id, error=str(e))
    
    job_manager.cancel(file_id)
    job_manager.set(file_id, "CANCELLED", job.active_stage, job.progress_percent, "Job cancelled by user")
    
    return {"message": "Job cancelled.", "file_id": file_id, "stage": job.active_stage}


@app.patch("/api/files/{file_id}/rows/{row_id}")
async def patch_row(file_id: str, row_id: str, req: RowPatchRequest) -> dict[str, str]:
    def _apply_patch() -> None:
        with database.connect() as conn:
            database.create_table(conn)
            updates: list[str] = []
            params: list[Any] = []

            if req.cleaned_title is not None:
                updates.append("title_pass2 = ?")
                params.append(req.cleaned_title)

            # Update content json if brand/parent/hsn/country are present
            if any(v is not None for v in [req.brand_name, req.parent_company, req.hsn_code, req.country_of_origin]):
                row_record = conn.execute(
                    "SELECT content_pass2_json FROM raw_rows WHERE file_id = ? AND sr_no = ?",
                    (file_id, row_id)
                ).fetchone()
                content_p2 = json.loads(row_record["content_pass2_json"] or "{}") if row_record else {}
                if req.brand_name is not None:
                    content_p2["brand_name"] = req.brand_name
                if req.parent_company is not None:
                    content_p2["parent_company"] = req.parent_company
                if req.hsn_code is not None:
                    content_p2["hsn_code"] = req.hsn_code
                if req.country_of_origin is not None:
                    content_p2["country_of_origin"] = req.country_of_origin
                updates.append("content_pass2_json = ?")
                params.append(json.dumps(content_p2, ensure_ascii=False))

            if req.gmr_taxonomy is not None:
                updates.append("gmr_taxonomy = ?")
                params.append(req.gmr_taxonomy)
                levels = taxonomy_v2.decompose_taxonomy(req.gmr_taxonomy)
                base_code = taxonomy_v2.lookup_base_code(levels["level1"], levels["level2"])
                updates.extend(["base_code_2digit = ?", "level1 = ?", "level2 = ?", "level3 = ?", "level4 = ?", "level5 = ?"])
                params.extend([base_code, levels["level1"], levels["level2"], levels["level3"], levels["level4"], levels["level5"]])

            if req.invoice_name is not None:
                updates.append("invoice_name = ?")
                params.append(req.invoice_name)
            if req.item_type is not None:
                updates.append("item_type = ?")
                params.append(req.item_type)
            if req.product_type is not None:
                updates.append("product_type = ?")
                params.append(req.product_type)

            updates.extend(["title_review_required = 0", "taxonomy_review_required = 0", "attributes_review_required = 0"])
            sql = f"UPDATE raw_rows SET {', '.join(updates)} WHERE file_id = ? AND sr_no = ?"
            params.extend([file_id, row_id])
            conn.execute(sql, params)

    await asyncio.to_thread(_apply_patch)
    return {"message": f"Row {row_id} updated."}


@app.post("/api/files/{file_id}/smart-propagate")
async def smart_propagate(file_id: str, req: SmartPropagateRequest) -> dict[str, Any]:
    def _apply_propagation() -> int:
        rows = database.rows(file_id)
        affected = 0
        match_pattern = req.matching_text.casefold()

        with database.connect() as conn:
            for r in rows:
                source_val = str(r.get(req.match_source_field, "") or r.get("Universal_Raw_Context", ""))
                if match_pattern in source_val.casefold():
                    if req.field_name == "brand_name":
                        content_p2 = r.get("Content Pass 2") or {}
                        content_p2["brand_name"] = req.target_value
                        conn.execute(
                            "UPDATE raw_rows SET content_pass2_json = ? WHERE file_id = ? AND sr_no = ?",
                            (json.dumps(content_p2, ensure_ascii=False), file_id, r["Sr. no."])
                        )
                        affected += 1
        return affected

    count = await asyncio.to_thread(_apply_propagation)
    return {"message": f"Applied change to {count} matching products.", "affected_count": count}


class BulkUnlinkRequest(BaseModel):
    row_ids: list[str]


def _allocate_unlinks(file_id: str, row_ids: list[str], duplicates_only: bool) -> dict[str, str]:
    targets = {r["Sr. no."]: r for r in database.rows(file_id) if r["Sr. no."] in set(row_ids)}
    counters = database.get_live_sequences(file_id)
    allocations: list[dict[str, Any]] = []
    result: dict[str, str] = {}

    for rid in row_ids:
        row = targets.get(rid)
        base_code = str((row or {}).get("2 Digit Base Code") or "").strip()
        is_dup = (row or {}).get("item_number_source") == "INTRA_FILE_REUSE"

        if row is None or len(base_code) != 4 or (duplicates_only and not is_dup):
            if duplicates_only:
                continue
            if row is None:
                raise HTTPException(status_code=404, detail=f"Row {rid} not found.")
            raise HTTPException(status_code=400, detail=f"Row {rid} has no valid 4-digit base code.")

        counters[base_code] = counters.get(base_code, 0) + 1
        item_no = f"{base_code}{counters[base_code]:06d}"
        result[rid] = item_no
        allocations.append({
            "row_id": rid,
            "item_number": item_no,
            "item_number_source": "NEW_ALLOCATION",
            "matched_master_title": str(row.get("Title Pass 2") or row.get("Title Pass 1") or "").strip(),
        })

    if allocations:
        item_number.persist_allocations(file_id, allocations)
    return result


@app.post("/api/files/{file_id}/unlink-duplicate/{row_id}")
async def unlink_duplicate(file_id: str, row_id: str) -> dict[str, str]:
    allocations = await asyncio.to_thread(_allocate_unlinks, file_id, [row_id], False)
    item_no = allocations[row_id]
    return {"message": f"Unlinked duplicate. New Item Number: {item_no}", "item_number": item_no}


@app.post("/api/files/{file_id}/unlink-duplicates")
async def unlink_duplicates(file_id: str, req: BulkUnlinkRequest) -> dict[str, Any]:
    allocations = await asyncio.to_thread(_allocate_unlinks, file_id, req.row_ids, True)
    if not allocations:
        raise HTTPException(status_code=400, detail="No duplicate rows to unlink in selection.")
    return {"message": f"Unlinked {len(allocations)} duplicates with new item numbers.", "allocations": allocations}


@app.post("/api/files/{file_id}/undo")
async def undo_stage(file_id: str) -> dict[str, str]:
    await asyncio.to_thread(item_number.rollback, file_id)
    job_manager.set(file_id, "IDLE", "UNICODE", 0, "Item numbers rolled back.")
    return {"message": f"Allocations for file {file_id} reverted."}


@app.get("/api/files/{file_id}/export")
async def export_file(file_id: str) -> FileResponse:
    output_path = await asyncio.to_thread(export.export_catalog, file_id)
    return FileResponse(
        path=output_path,
        filename=f"Final_Catalog_{file_id[:8]}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
class ColumnConfirmationRequest(BaseModel):
    mappings: list[dict[str, Any]]


@app.get("/api/files/{file_id}/columns")
async def get_column_mappings(file_id: str) -> dict[str, Any]:
    rows = await asyncio.to_thread(database.rows, file_id)
    cols = await asyncio.to_thread(database.columns, file_id)
    if not rows:
        raise HTTPException(status_code=404, detail="File not found.")

    first_field_map = rows[0].get("Field Map") or {}
    mappings = []
    for c in cols:
        item = first_field_map.get(c, {})
        mappings.append({
            "column": c,
            "role": item.get("role", "PRODUCT_DESCRIPTION"),
            "use_in_primary": item.get("use_in_primary", True),
            "use_in_secondary": item.get("use_in_secondary", False),
        })
    return {"mappings": mappings}


@app.post("/api/files/{file_id}/columns")
async def save_column_mappings(file_id: str, req: ColumnMappingUpdate) -> dict[str, str]:
    await asyncio.to_thread(context.apply, file_id, req.mappings, "manual-override")
    return {"message": "Column contexts successfully updated."}


static_path = Path(__file__).parent / "frontend"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="frontend")
    
