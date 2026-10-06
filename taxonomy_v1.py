from __future__ import annotations

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

import database
import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 15
DIRECT_EXECUTION_THRESHOLD = 200

TaxonomyQuality = Literal["VERIFIED_CLEAN", "NORMALIZED_FIXED", "SPARSE_INPUT", "UNGROUNDED_FLAG", "FLAGGED_NO_TAXONOMY"]

BOOK2_CODES: dict[str, str] = {
    "Branded Locations>Accessories": "1010",
    "Branded Locations>Apparel": "1011",
    "Branded Locations>Eyewear": "1012",
    "Branded Locations>Fragrance": "1013",
    "Branded Locations>Gift With Purchase": "1014",
    "Branded Locations>Home/Car Fragrances": "1015",
    "Branded Locations>Jewellery": "1016",
    "Branded Locations>Makeup/Cosmetics": "1017",
    "Branded Locations>Skincare": "1018",
    "Branded Locations>Toiletries": "1019",
    "Branded Locations>Travel Set": "1020",
    "Branded Locations>Watches": "1021",
    "Branded Locations>Writing Equipments": "1022",
    "Confectionery>Beverages": "1110",
    "Confectionery>Candy": "1111",
    "Confectionery>Chocolates": "1112",
    "Confectionery>Gift With Purchase": "1113",
    "Confectionery>Gums & mints": "1114",
    "Confectionery>Health & Wellness": "1115",
    "Confectionery>Savoury & Sweet Snacks": "1116",
    "Confectionery>Spices & Condiments": "1117",
    "Destination & Souvenirs>Apparel & Accessories": "1210",
    "Destination & Souvenirs>Books, Guides & Media": "1211",
    "Destination & Souvenirs>Gift With Purchase": "1212",
    "Destination & Souvenirs>Handicrafts & Art": "1213",
    "Destination & Souvenirs>Home Décor & Lifestyle": "1214",
    "Destination & Souvenirs>Local Beverages": "1215",
    "Destination & Souvenirs>Local Food & Confectionery": "1216",
    "Destination & Souvenirs>Seasonal & Festival": "1217",
    "Destination & Souvenirs>Spices & Condiments": "1218",
    "Destination & Souvenirs>Stationery & Collectibles": "1219",
    "Destination & Souvenirs>Toys & Games": "1220",
    "Fashion, Lifestyle & Accessories>Accessories": "1310",
    "Fashion, Lifestyle & Accessories>Apparel": "1311",
    "Fashion, Lifestyle & Accessories>Electronics": "1312",
    "Fashion, Lifestyle & Accessories>Eyewear": "1313",
    "Fashion, Lifestyle & Accessories>Gift With Purchase": "1314",
    "Fashion, Lifestyle & Accessories>Jewellery": "1315",
    "Fashion, Lifestyle & Accessories>Stationery & Gifts": "1316",
    "Fashion, Lifestyle & Accessories>Toys & Games": "1317",
    "Fashion, Lifestyle & Accessories>Watches": "1318",
    "Fashion, Lifestyle & Accessories>Writing Equipments": "1319",
    "Household & Cleaning>Cleaning": "1410",
    "Household & Cleaning>Gift With Purchase": "1411",
    "Liquor>American Whiskey": "1510",
    "Liquor>Aperitifs & Vermouth": "1511",
    "Liquor>Bitters": "1512",
    "Liquor>Brandy & Cognac": "1513",
    "Liquor>Gift With Purchase": "1514",
    "Liquor>Gin": "1515",
    "Liquor>Indian Spirits": "1516",
    "Liquor>Irish Whiskey": "1517",
    "Liquor>Japanese Whiskey": "1518",
    "Liquor>Liqueurs & Cordials": "1519",
    "Liquor>Non-Alcoholic (0.0%)": "1520",
    "Liquor>Other Whiskey": "1521",
    "Liquor>Ready to Drink": "1522",
    "Liquor>Rum": "1523",
    "Liquor>Sake": "1524",
    "Liquor>Scotch Whisky": "1525",
    "Liquor>Tequila & Mezcal": "1526",
    "Liquor>Vodka": "1527",
    "Liquor>Wine": "1528",
    "Non-Saleable Items>Carry Bags": "1610",
    "Non-Saleable Items>Display Material": "1611",
    "Non-Saleable Items>Marketing Material": "1612",
    "Non-Saleable Items>Stationery": "1613",
    "Perfumes & Cosmetics>Baby Care": "1710",
    "Perfumes & Cosmetics>Fragrance": "1711",
    "Perfumes & Cosmetics>Gift With Purchase": "1712",
    "Perfumes & Cosmetics>Home/Car Fragrances": "1713",
    "Perfumes & Cosmetics>Makeup/Cosmetics": "1714",
    "Perfumes & Cosmetics>Skincare": "1715",
    "Perfumes & Cosmetics>Toiletries": "1716",
    "Perfumes & Cosmetics>Travel Set": "1717",
    "Tobacco>Cigarettes": "1810",
    "Tobacco>Cigars": "1811",
    "Tobacco>Gift With Purchase": "1812",
    "Tobacco>Heated Tobacco & Vaping": "1813",
    "Tobacco>Loose & Pipe Tobacco": "1814",
    "Tobacco>Shisha & Hookah": "1815",
    "Tobacco>Smokeless Tobacco & Snus": "1816",
    "Tobacco>Smoking Accessories": "1817",
}

GENERATION_SYSTEM = """You are the master GMR Taxonomy Classifier for a duty-free and retail catalog.
Assign the single most accurate GMR Taxonomy Path for each item strictly from the provided candidate list.

Rules:
1. Choose ONLY an exact string from the provided 'candidate_taxonomies'. Never invent or alter paths.
2. If Item Type is Non-Saleable, choose the matching Tester, Blotter, GWP, or Display leaf path.
3. Pick the most specific leaf node matching the product identity, variant, gender, and attributes.
4. Return only the requested structured schema with literal supporting evidence. Preserve row_id exactly."""

AUDIT_SYSTEM = """You are the independent GMR Taxonomy Auditor.
Audit Pass 1 taxonomy selections against the allowed candidates, Item Type, Product Type, final title, attributes, and source context.

Rules:
1. Verify that the selected taxonomy is present in 'candidate_taxonomies'.
2. If Pass 1 made a subcategory error or chose a non-leaf node, replace it with the exact best candidate and flag NORMALIZED_FIXED.
3. If no candidate in the allowed list fits the item, set quality_flag to FLAGGED_NO_TAXONOMY, review_required to true, and explain in audit_notes.
4. Return only the structured schema. Preserve row_id exactly."""


class TaxonomyTree:
    _instance: TaxonomyTree | None = None

    def __init__(self, excel_path: str | Path = "all_categories.xlsx"):
        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy source not found: {path}")
        frame = pd.read_excel(path, header=None, dtype=str, engine="openpyxl")
        self.all_paths: list[str] = frame.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        self.path_set: set[str] = set(self.all_paths)

    @classmethod
    def get_instance(cls, excel_path: str | Path = "all_categories.xlsx") -> TaxonomyTree:
        if cls._instance is None:
            cls._instance = cls(excel_path)
        return cls._instance

    def filter_candidates(self, level1_hint: str | None, product_type: str | None, item_type: str | None) -> list[str]:
        candidates = self.all_paths
        if level1_hint:
            candidates = [p for p in candidates if p.startswith(f"{level1_hint}>") or p == level1_hint]

        if item_type == "Non-Saleable":
            ns_terms = ("tester", "blotter", "gift with purchase", "gwp", "display", "dummy", "carry bags", "marketing material", "sample")
            ns_filtered = [p for p in candidates if any(t in p.casefold() for t in ns_terms)]
            if ns_filtered:
                candidates = ns_filtered
        elif product_type:
            pt_clean = product_type.split("&")[0].strip().casefold()
            pt_filtered = [p for p in candidates if pt_clean in p.casefold()]
            if pt_filtered:
                candidates = pt_filtered

        return candidates[:100]


class TaxonomyDraft(BaseModel):
    row_id: int
    gmr_taxonomy: str
    evidence: list[str] = Field(default_factory=list)


class TaxonomyAudit(BaseModel):
    row_id: int
    gmr_taxonomy: str
    quality_flag: TaxonomyQuality
    review_required: bool
    audit_notes: str


class BatchTaxonomyDraftResponse(BaseModel):
    items: list[TaxonomyDraft]


class BatchTaxonomyAuditResponse(BaseModel):
    items: list[TaxonomyAudit]


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


def parse_json_response(text: str, context: str = "API response") -> dict[str, Any]:
    text = (text or "").strip()

    if not text:
        raise RuntimeError(f"{context} returned an empty response")

    # Remove accidental Markdown code fences without changing valid JSON.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Attempt 1: Standard load with strict=False (allows unescaped control characters)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        original_exc = exc

    # Attempt 2: If truncated or unterminated string inside {"items": [...]},
    # search backwards for the last complete item closing brace and seal JSON
    idx = len(text)
    while idx > 0:
        last_brace = text.rfind("}", 0, idx)
        if last_brace == -1:
            break
        candidate = text[: last_brace + 1].rstrip(", \t\r\n")
        if candidate.startswith("{") and '"items"' in candidate:
            for suffix in ["\n]}", "\n}"]:
                try:
                    data = json.loads(candidate + suffix, strict=False)
                    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                        logger.event("TAXONOMY", "json_repaired_truncated_items", context=context, salvaged=len(data["items"]))
                        return data
                except json.JSONDecodeError:
                    pass
        idx = last_brace

    # Attempt 3: Regex extraction of individual valid item objects
    try:
        salvaged_items = []
        for match in re.finditer(r'\{[^{}]*"row_id"\s*:\s*\d+[^{}]*\}', text):
            try:
                item_obj = json.loads(match.group(0), strict=False)
                if "row_id" in item_obj and "gmr_taxonomy" in item_obj:
                    salvaged_items.append(item_obj)
            except Exception:
                continue
        if salvaged_items:
            logger.event("TAXONOMY", "json_regex_salvaged_items", context=context, salvaged=len(salvaged_items))
            return {"items": salvaged_items}
    except Exception:
        pass

    start = max(0, original_exc.pos - 300)
    end = min(len(text), original_exc.pos + 300)
    snippet = text[start:end]
    raise RuntimeError(
        f"Invalid JSON in {context}. "
        f"Error at character {original_exc.pos}: {original_exc.msg}. "
        f"Nearby response content:\n{snippet}"
    ) from original_exc


def decompose_taxonomy(path: str) -> dict[str, str]:
    parts = [part.strip() for part in path.split(">")]
    return {
        "level1": parts[0] if len(parts) > 0 else "",
        "level2": parts[1] if len(parts) > 1 else "",
        "level3": parts[2] if len(parts) > 2 else "",
        "level4": parts[3] if len(parts) > 3 else "",
        "level5": parts[4] if len(parts) > 4 else "",
    }


def lookup_base_code(level1: str, level2: str) -> str:
    key = f"{level1}>{level2}".strip()
    return BOOK2_CODES.get(key, "UNMAPPED")


def detect_level1_hint(row: dict[str, Any]) -> str | None:
    context = f"{row.get('Combined Description', '')} {row.get('Universal_Raw_Context', '')}".casefold()
    if "destination" in context or "souvenir" in context:
        return "Destination & Souvenirs"
    if "branded location" in context:
        return "Branded Locations"
    return None


def source_payload(row: dict[str, Any], candidates: list[str], pass2: bool) -> dict[str, Any]:
    data = {
        "row_id": int(row["Sr. no."]),
        "item_type": row.get("Item Type"),
        "product_type": row.get("Product Type"),
        "final_title": row.get("Title Pass 2") or row.get("Title Pass 1"),
        "attributes": row.get("Attributes Pass 2") or row.get("Attributes Pass 1") or {},
        "candidate_taxonomies": candidates,
    }
    if pass2:
        data["pass1_taxonomy"] = row.get("Taxonomy Pass 1", "")
        data["universal_raw_context_audit_only"] = row.get("Universal_Raw_Context", "")
    return data


def client() -> Any:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for taxonomy mapping")
    if genai is None:
        raise RuntimeError("google-genai is required for taxonomy mapping")
    return genai.Client(api_key=key)


def build_chunk_payloads(rows: list[dict[str, Any]], tree: TaxonomyTree, pass2: bool) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        l1_hint = detect_level1_hint(row)
        candidates = tree.filter_candidates(l1_hint, row.get("Product Type"), row.get("Item Type"))
        payload.append(source_payload(row, candidates, pass2))
    return payload


def direct_parallel_taxonomy(
    rows: list[dict[str, Any]],
    tree: TaxonomyTree,
    pass2: bool,
    api: Any,
) -> dict[int, dict[str, Any]]:
    """Fast concurrent direct API execution for <= 200 items (~2-3 seconds)."""
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    schema_model = BatchTaxonomyAuditResponse if pass2 else BatchTaxonomyDraftResponse
    schema = inline_schema(schema_model)

    def _process_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = build_chunk_payloads(chunk, tree, pass2)
        try:
            response = api.models.generate_content(
                model=MODEL_NAME,
                contents=[f"{'Audit' if pass2 else 'Classify'} taxonomy:\n{json.dumps(payload, ensure_ascii=False)}"],
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": 0.0,
                },
            )
            raw = parse_json_response(response.text, "direct taxonomy response")
            items = schema_model.model_validate(raw).items
            returned_ids = {it.row_id for it in items}
            expected_ids = {int(r["Sr. no."]) for r in chunk}
            if expected_ids.issubset(returned_ids):
                return items

            # If some items were salvaged from a truncated response, retry only missing rows individually
            missing_ids = expected_ids - returned_ids
            if missing_ids and len(chunk) > 1:
                missing_sub = [r for r in chunk if int(r["Sr. no."]) in missing_ids]
                for single_row in missing_sub:
                    items.extend(_process_chunk([single_row]))
            return items
        except Exception as exc:
            if len(chunk) > 1:
                logger.event("TAXONOMY", "direct_chunk_retry_fallback", size=len(chunk), error=str(exc))
                fallback_items = []
                for single_row in chunk:
                    fallback_items.extend(_process_chunk([single_row]))
                return fallback_items
            logger.event("TAXONOMY", "direct_chunk_error", row_id=chunk[0].get("Sr. no."), error=str(exc))
            raise

    values: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_results in executor.map(_process_chunk, chunks):
            for item in chunk_results:
                tax = item.gmr_taxonomy.strip()
                if pass2:
                    flag = item.quality_flag
                    review = item.review_required
                    notes = item.audit_notes

                    if tax not in tree.path_set:
                        flag = "FLAGGED_NO_TAXONOMY"
                        review = True
                        notes = f"Path not in authoritative tree. {notes}".strip()

                    levels = decompose_taxonomy(tax)
                    base_code = lookup_base_code(levels["level1"], levels["level2"])
                    if base_code == "UNMAPPED":
                        review = True
                        notes = f"Unmapped 2-digit base code for {levels['level1']}>{levels['level2']}. {notes}".strip()

                    values[item.row_id] = {
                        "row_id": str(item.row_id),
                        "gmr_taxonomy": tax,
                        "levels": levels,
                        "base_code_2digit": base_code,
                        "quality_flag": flag,
                        "review_required": review,
                        "audit_notes": notes,
                    }
                else:
                    values[item.row_id] = {
                        "row_id": str(item.row_id),
                        "gmr_taxonomy": tax,
                        "evidence": getattr(item, "evidence", []),
                    }
    return values


def batch(
    rows: list[dict[str, Any]],
    tree: TaxonomyTree,
    pass2: bool,
    api: Any | None = None,
    chunk_size: int = CHUNK_SIZE,
    poll_seconds: float = 15.0,
) -> tuple[list[dict[str, Any]], str]:
    stage = "TAXONOMY_AUDIT" if pass2 else "TAXONOMY_GENERATION"
    service = api or client()
    system = AUDIT_SYSTEM if pass2 else GENERATION_SYSTEM
    schema_model = BatchTaxonomyAuditResponse if pass2 else BatchTaxonomyDraftResponse
    schema = inline_schema(schema_model)

    requests = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        payload = build_chunk_payloads(chunk, tree, pass2)
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

        result: dict[int, dict[str, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                val = json.loads(line)
            except Exception as exc:
                logger.event("TAXONOMY", "batch_line_json_error", error=str(exc))
                continue

            candidates = (val.get("response") or {}).get("candidates") or []
            if candidates:
                cand = candidates[0]
                text = cand.get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    try:
                        raw = parse_json_response(text, f"batch taxonomy response ({val.get('key', 'unknown')})")
                        parsed = schema_model.model_validate(raw)
                    except Exception as exc:
                        logger.event("TAXONOMY", "batch_chunk_parse_error", key=val.get("key"), error=str(exc))
                        continue

                    for item in parsed.items:
                        tax = item.gmr_taxonomy.strip()
                        if pass2:
                            flag = item.quality_flag
                            review = item.review_required
                            notes = item.audit_notes

                            if tax not in tree.path_set:
                                flag = "FLAGGED_NO_TAXONOMY"
                                review = True
                                notes = f"Path not in authoritative tree. {notes}".strip()

                            levels = decompose_taxonomy(tax)
                            base_code = lookup_base_code(levels["level1"], levels["level2"])
                            if base_code == "UNMAPPED":
                                review = True
                                notes = f"Unmapped base code for {levels['level1']}>{levels['level2']}. {notes}".strip()

                            result[item.row_id] = {
                                "row_id": str(item.row_id),
                                "gmr_taxonomy": tax,
                                "levels": levels,
                                "base_code_2digit": base_code,
                                "quality_flag": flag,
                                "review_required": review,
                                "audit_notes": notes,
                            }
                        else:
                            result[item.row_id] = {
                                "row_id": str(item.row_id),
                                "gmr_taxonomy": tax,
                                "evidence": getattr(item, "evidence", []),
                            }

        expected = {int(row["Sr. no."]) for row in rows}
        missing = sorted(expected - result.keys())
        if missing:
            logger.event("TAXONOMY", "batch_reconciliation_retry", stage=stage, missing_count=len(missing))
            row_map = {int(r["Sr. no."]): r for r in rows}
            missing_rows = [row_map[idx] for idx in missing if idx in row_map]
            if missing_rows:
                recovered = direct_parallel_taxonomy(missing_rows, tree, pass2, service)
                result.update(recovered)
            missing = sorted(expected - result.keys())
            if missing:
                raise RuntimeError(f"{stage} row reconciliation failed: missing={missing[:10]}")

        return [result[idx] for idx in sorted(result)], batch_id
    finally:
        input_path.unlink(missing_ok=True)


def run(
    file_id: str,
    excel_path: str | Path = "all_categories.xlsx",
    api: Any | None = None,
    row_ids: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    poll_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    tree = TaxonomyTree.get_instance(excel_path)
    rows = database.rows(file_id)
    selected = {str(value) for value in row_ids} if row_ids else None
    rows = [row for row in rows if selected is None or row["Sr. no."] in selected]
    if not rows:
        return []

    logger.event("TAXONOMY", "start", file_id=file_id, rows=len(rows))
    service = api or client()

    # Pass 1: Generation
    if len(rows) <= DIRECT_EXECUTION_THRESHOLD:
        drafts_map = direct_parallel_taxonomy(rows, tree, False, service)
        drafts = [drafts_map[int(r["Sr. no."])] for r in rows]
        pass1_batch_id = "direct-execution"
    else:
        drafts, pass1_batch_id = batch(rows, tree, False, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)

    database.update_taxonomy_drafts(file_id, pass1_batch_id, drafts)
    logger.event("TAXONOMY", "pass1_stored", file_id=file_id, rows=len(drafts))

    # Pass 2: Audit
    audited_rows = database.rows(file_id)
    audited_rows = [row for row in audited_rows if selected is None or row["Sr. no."] in selected]

    if len(audited_rows) <= DIRECT_EXECUTION_THRESHOLD:
        audits_map = direct_parallel_taxonomy(audited_rows, tree, True, service)
        audits = [audits_map[int(r["Sr. no."])] for r in audited_rows]
        pass2_batch_id = "direct-execution"
    else:
        audits, pass2_batch_id = batch(audited_rows, tree, True, api=service, chunk_size=chunk_size, poll_seconds=poll_seconds)

    database.update_taxonomy_audits(file_id, pass2_batch_id, audits)
    logger.event("TAXONOMY", "pass2_stored", file_id=file_id, rows=len(audits))
    return audits