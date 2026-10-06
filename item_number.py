from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

import database
import logger
from normalize import normalize_title

MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 15

AllocationSource = Literal[
    "MASTER_REUSE",
    "INTRA_FILE_REUSE",
    "NEW_ALLOCATION",
    "BLOCKED_UNMAPPED_TAXONOMY",
]

TITLE_ALIASES = [
    "product title", "product name", "cleaned title", "item description",
    "description", "marketplace title", "title", "clean title",
]

ITEM_NO_ALIASES = [
    "generated item no", "item no", "item number", "material code",
    "item_number", "generated_item_no", "item_no", "material_code",
    "item code", "sap code", "unicode",
]

SKU_MATCH_SYSTEM = """You are the master SKU reconciliation auditor for an international duty-free catalog.
Compare a TARGET PRODUCT with candidate items from the master catalog. Determine if any candidate represents the EXACT SAME PHYSICAL SKU.

Strict Identity Rules:
1. Brand, Sub-brand/Variant, Physical Volume/Weight/Size, Pack Configuration, and Color/Flavor/Shade must all match.
2. Volume or Size mismatch (e.g. 70cl vs 100cl, 50ml vs 100ml, Single vs Twin Pack) is NEVER a match.
3. If an exact match is found, return the candidate's exact 'item_number'.
4. If no candidate matches or if there are conflicting specifications, return 'NONE'.

Return only the structured JSON response."""


class SingleMatchDecision(BaseModel):
    row_id: int
    matched_item_number: str = Field(description="The exact item_number of the matched master product, or 'NONE'")
    match_confidence: float = Field(ge=0.0, le=1.0)
    decision_reason: str


class BatchMatchResponse(BaseModel):
    matches: list[SingleMatchDecision]


def normalize_lookup_key(title: str) -> str:
    clean = normalize_title(title).casefold()
    clean = re.sub(r"[^\w\s%./+-]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


def extract_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {w for w in words if len(w) >= 2}


class MasterCatalogIndex:

    def __init__(self, records: list[dict[str, str]]):
        self.by_base_code: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.exact_title_map: dict[str, str] = {}

        for r in records:
            item_no = str(r["item_number"]).strip()
            title = str(r.get("original_title") or r.get("normalized_title", "")).strip()
            base_code = str(r.get("base_code_2digit", "")).strip() or item_no[:4]
            norm_title = normalize_lookup_key(title)

            entry = {
                "item_number": item_no,
                "original_title": title,
                "normalized_title": norm_title,
                "base_code_2digit": base_code,
                "tokens": extract_tokens(title),
            }
            self.by_base_code[base_code].append(entry)
            if norm_title:
                self.exact_title_map[norm_title] = item_no

    def find_candidates(self, base_code: str, target_title: str, brand: str = "", max_candidates: int = 5) -> list[dict[str, str]]:
        partition = self.by_base_code.get(base_code, [])
        if not partition:
            return []

        target_tokens = extract_tokens(f"{target_title} {brand}")
        if not target_tokens:
            return []

        scored_candidates: list[tuple[float, dict[str, str]]] = []
        for entry in partition:
            cand_tokens = entry["tokens"]
            common = target_tokens.intersection(cand_tokens)
            if not common:
                continue
            # Jaccard / overlap score
            score = len(common) / len(target_tokens.union(cand_tokens))
            if score > 0.35:  # High token overlap threshold
                scored_candidates.append((score, entry))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_candidates[:max_candidates]]


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Auto-detects Title and Item Number columns from master spreadsheet."""
    cols = {str(c).strip().casefold(): str(c) for c in df.columns}
    title_col, item_col = None, None

    for alias in TITLE_ALIASES:
        if alias in cols:
            title_col = cols[alias]
            break
    for alias in ITEM_NO_ALIASES:
        if alias in cols:
            item_col = cols[alias]
            break

    if not title_col or not item_col:
        raise ValueError(
            f"Could not identify columns in master file. Found columns: {list(df.columns)}. "
            f"Expected Title alias and Item No alias."
        )
    return title_col, item_col


def ingest_master_file(file_path: str | Path, source_label: str = "EXTERNAL_MASTER") -> int:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Master file not found: {path}")

    logger.event("MASTER_INGEST", "reading_file", path=str(path))
    df = pd.read_excel(path, dtype=str) if path.suffix.lower() in {".xlsx", ".xls", ".xlsb"} else pd.read_csv(path, dtype=str)
    title_col, item_col = detect_columns(df)

    entries = []
    for _, row in df.iterrows():
        item_no = str(row.get(item_col, "")).strip().zfill(10)
        title = str(row.get(title_col, "")).strip()
        if len(item_no) == 10 and title:
            base_code = item_no[:4]
            entries.append({
                "item_number": item_no,
                "normalized_title": normalize_lookup_key(title),
                "original_title": title,
                "base_code_2digit": base_code,
                "level1": str(row.get("Level1", row.get("Level 1", ""))).strip(),
                "level2": str(row.get("Level2", row.get("Level 2", ""))).strip(),
            })

    if entries:
        database.update_item_numbers(source_label, [], entries)
    logger.event("MASTER_INGEST", "completed", entries=len(entries))
    return len(entries)


def client() -> Any:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for SKU matching")
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError("google-genai==2.23.0 is required for SKU matching") from error
    return genai.Client(api_key=key)


def state(job: Any) -> str:
    val = getattr(job, "state", "")
    return str(getattr(val, "name", val))


def response_text(value: dict[str, Any]) -> str:
    candidates = (value.get("response") or {}).get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    return parts[0].get("text", "") if parts else ""


def llm_sku_match(matching_tasks: list[dict[str, Any]], poll_seconds: float = 10.0) -> dict[int, str]:
    """Runs Gemini batch reconciliation for rows with high candidate overlap."""
    if not matching_tasks:
        return {}

    service = client()
    requests = []
    for start in range(0, len(matching_tasks), CHUNK_SIZE):
        chunk = matching_tasks[start : start + CHUNK_SIZE]
        payload = [
            {
                "row_id": int(task["row_id"]),
                "target_product": {
                    "title": task["title"],
                    "brand": task["brand"],
                    "item_type": task["item_type"],
                    "product_type": task["product_type"],
                    "attributes": task["attributes"],
                },
                "candidate_master_products": [
                    {"item_number": c["item_number"], "title": c["original_title"]}
                    for c in task["candidates"]
                ],
            }
            for task in chunk
        ]
        requests.append({
            "key": f"chunk-{start // CHUNK_SIZE}",
            "request": {
                "system_instruction": {"parts": [{"text": SKU_MATCH_SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": f"Reconcile items:\n{json.dumps(payload, ensure_ascii=False)}"}]}],
                "generation_config": {
                    "response_mime_type": "application/json",
                    "response_schema": BatchMatchResponse.model_json_schema(),
                    "temperature": 0.0,
                },
            },
        })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
        input_path = Path(f.name)

    try:
        from google.genai import types
        uploaded = service.files.upload(file=str(input_path), config=types.UploadFileConfig(display_name="sku-match", mime_type="jsonl"))
        job = service.batches.create(model=MODEL_NAME, src=uploaded.name, config={"display_name": "sku-match"})

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while state(job) not in terminal:
            time.sleep(poll_seconds)
            job = service.batches.get(name=job.name)

        if state(job) != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"SKU Match Batch failed: {getattr(job, 'error', '')}")

        res_file = getattr(getattr(job, "dest", None), "file_name", None)
        content = service.files.download(file=res_file).decode("utf-8")

        matched_results: dict[int, str] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            val = json.loads(line)
            parsed = BatchMatchResponse.model_validate_json(response_text(val))
            for m in parsed.matches:
                if m.matched_item_number and m.matched_item_number.upper() != "NONE":
                    matched_results[m.row_id] = m.matched_item_number.strip()
        return matched_results
    finally:
        input_path.unlink(missing_ok=True)


def assign_unicodes(
    file_id: str,
    check_master: bool = True,
    poll_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    rows = database.rows(file_id)
    if not rows:
        return []

    logger.event("UNICODE", "start", file_id=file_id, rows=len(rows), check_master=check_master)

    database.rollback_item_numbers(file_id)

    master_records = database.load_master_records()
    master_index = MasterCatalogIndex(master_records)
    counters = database.get_max_sequences()

    intra_file_map: dict[str, str] = {}  # {norm_title: item_number}
    llm_matching_tasks: list[dict[str, Any]] = []

    for row in rows:
        row_id = int(row["Sr. no."])
        title = str(row.get("Title Pass 2") or row.get("Title Pass 1") or "").strip()
        base_code = str(row.get("2 Digit Base Code") or "").strip()
        brand = str(row.get("Content Pass 2", {}).get("brand_name") or "").strip()
        norm_key = normalize_lookup_key(title)

        if not base_code or base_code == "UNMAPPED" or len(base_code) != 4:
            continue

        if check_master:
            if norm_key in master_index.exact_title_map:
                intra_file_map[norm_key] = master_index.exact_title_map[norm_key]
                continue

            cands = master_index.find_candidates(base_code, title, brand)
            if cands:
                llm_matching_tasks.append({
                    "row_id": row_id,
                    "title": title,
                    "brand": brand,
                    "item_type": row.get("Item Type"),
                    "product_type": row.get("Product Type"),
                    "attributes": row.get("Attributes Pass 2") or {},
                    "candidates": cands,
                })

    llm_matches: dict[int, str] = {}
    if llm_matching_tasks:
        logger.event("UNICODE", "llm_matching_start", tasks=len(llm_matching_tasks))
        llm_matches = llm_sku_match(llm_matching_tasks, poll_seconds=poll_seconds)
        logger.event("UNICODE", "llm_matching_done", matches_found=len(llm_matches))

    allocations: list[dict[str, Any]] = []
    new_master_entries: list[dict[str, Any]] = []

    for row in rows:
        row_id_int = int(row["Sr. no."])
        row_id_str = str(row["Sr. no."])
        title = str(row.get("Title Pass 2") or row.get("Title Pass 1") or "").strip()
        base_code = str(row.get("2 Digit Base Code") or "").strip()
        l1 = str(row.get("Level 1") or "").strip()
        l2 = str(row.get("Level 2") or "").strip()
        norm_key = normalize_lookup_key(title)

        if not base_code or base_code == "UNMAPPED" or len(base_code) != 4:
            allocations.append({
                "row_id": row_id_str,
                "item_number": "BLOCKED",
                "item_number_source": "BLOCKED_UNMAPPED_TAXONOMY",
                "matched_master_title": "",
            })
            continue

        if row_id_int in llm_matches:
            matched_code = llm_matches[row_id_int]
            intra_file_map[norm_key] = matched_code
            allocations.append({
                "row_id": row_id_str,
                "item_number": matched_code,
                "item_number_source": "MASTER_REUSE",
                "matched_master_title": title,
            })
            continue

        if norm_key in intra_file_map:
            reused_code = intra_file_map[norm_key]
            # If it's already in the master catalog, source is MASTER_REUSE; otherwise INTRA_FILE_REUSE (purple)
            source_type = "MASTER_REUSE" if reused_code in master_index.exact_title_map.values() else "INTRA_FILE_REUSE"
            allocations.append({
                "row_id": row_id_str,
                "item_number": reused_code,
                "item_number_source": source_type,
                "matched_master_title": title,
            })
            continue

        current_seq = counters.get(base_code, 0) + 1
        counters[base_code] = current_seq
        new_item_no = f"{base_code}{current_seq:06d}"

        intra_file_map[norm_key] = new_item_no

        allocations.append({
            "row_id": row_id_str,
            "item_number": new_item_no,
            "item_number_source": "NEW_ALLOCATION",
            "matched_master_title": "",
        })
        new_master_entries.append({
            "item_number": new_item_no,
            "normalized_title": norm_key,
            "original_title": title,
            "base_code_2digit": base_code,
            "level1": l1,
            "level2": l2,
        })

    # Persist batch updates & sync to master catalog
    database.update_item_numbers(file_id, allocations, new_master_entries)
    logger.event("UNICODE", "complete", file_id=file_id, allocated=len(allocations))
    return allocations


def rollback(file_id: str) -> None:
    database.rollback_item_numbers(file_id)
    logger.event("UNICODE", "rollback_complete", file_id=file_id)