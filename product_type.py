from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

import database
import logger
import gemini
from normalize import normalize_product_type

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

MainCategory = Literal[
    "Whisky", "Vodka", "Gin", "Rum", "Tequila & Mezcal", "Brandy & Cognac", "Aperitifs & Vermouth",
    "Liqueurs & Cordials", "Wine & Champagne", "Non-Alcoholic (0.0%)", "Ready to Drink", "Tech Accessories",
    "Accessories", "Apparel", "Personal Care", "Fragrance", "Beverages", "Books, Guides & Media", "Candy",
    "Chocolates", "Cigarettes", "Cigars", "Detergents & Liquids", "Eyewear", "Gums & mints",
    "Handicrafts & Art", "Health & Wellness", "Savoury & Sweet Snacks", "Heated Tobacco & Vaping",
    "Home Fragrance", "Home Décor & Lifestyle", "Jewellery", "Local Food & Confectionery", "Loose & Pipe Tobacco",
    "Makeup", "Seasonal & Festival", "Shisha & Hookah", "Smokeless Tobacco & Snus", "Smoking Accessories",
    "Stationery & Gifts", "Toys, Games & Plush", "Watches", "Writing Instruments", "Footwear",
]

MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 30
DIRECT_EXECUTION_THRESHOLD = 200

SYSTEM_PROMPT = """You are an expert master-data taxonomist for retail, luxury, duty-free, and FMCG products.
Classify each item into EXACTLY ONE allowed product_type based on the complete product-facing context.

Category Disambiguation Rules:
- Apparel: Wearable clothing, including Belts (Men/Women/Kids), Ties, Hats, Caps, Scarves, Shawls, Socks, and Outerwear.
- Accessories: Handbags, Backpacks, Luggage/Trolleys, Wallets, Cardholders, Keychains, Travel Pouches, Eyewear Cases, and Umbrellas.
- Writing Instruments: Pens, Inks, Cartridges, Refills, Notebooks, Diaries, Planners, Pen Cases/Pouches, and Desk Sets.
- Personal Care: Skincare, Body wash, Deodorants, Shaving & Grooming, Toiletries, and Toiletry/Dopp Kits.
- Liqueurs & Cordials vs Base Spirits: Sweetened/flavored spirit beverages (e.g. Baileys, Cointreau, Kahlua, honey whiskeys, fruit gin liqueurs) belong to Liqueurs & Cordials.
- Aperitifs & Vermouth: Bitters, Vermouths, Campari, Aperol, Pimm's, and Absinthe.
- Wine & Champagne: Still wines, sparkling wines, Champagne, Prosecco, Cava, Port, and Sherry.
- Ready to Drink: Canned beer, cider, seltzers, and pre-mixed cocktails.
- Cigarettes vs Loose Tobacco: Pre-made cigarette packs/cartons are Cigarettes. Rolling or pipe tobacco sold by weight (e.g. 50g, 250g) is Loose & Pipe Tobacco.

Return only the requested structured schema. Preserve every row_id exactly."""


class SingleProductClassification(BaseModel):
    row_id: int
    product_type: MainCategory


class BatchClassificationResponse(BaseModel):
    items: list[SingleProductClassification]


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


def build_chunk_payload(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": int(row["Sr. no."]),
            "item_type": row.get("Item Type"),
            "text": " ".join(
                value for value in [
                    str(row.get("Combined Description", "") or "").strip(),
                    str(row.get("Secondary Product Description", "") or "").strip(),
                ]
                if value
            ),
        }
        for row in chunk
    ]


def client() -> Any:
    return gemini.build_client("Product Type classification")


def direct_parallel_classify(rows: list[dict[str, Any]], api: Any, on_progress: Any | None = None) -> dict[int, str]:
    """Fast direct parallel execution for <= 200 items (~2 seconds)."""
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    schema = inline_schema(BatchClassificationResponse)

    def _process_chunk(chunk: list[dict[str, Any]]) -> list[SingleProductClassification]:
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
        raw = json.loads(response.text)
        for item in raw.get("items", []):
            item["product_type"] = normalize_product_type(item.get("product_type"))
        return BatchClassificationResponse.model_validate(raw).items

    classifications: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_items in executor.map(_process_chunk, chunks):
            for item in chunk_items:
                classifications[item.row_id] = item.product_type
            if on_progress:
                on_progress(len(chunk_items))
    return classifications


def batch_classify(rows: list[dict[str, Any]], api: Any, chunk_size: int = CHUNK_SIZE, poll_seconds: float = 15.0) -> tuple[dict[int, str], str]:
    requests = []
    schema = inline_schema(BatchClassificationResponse)

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        payload = build_chunk_payload(chunk)
        requests.append(
            {
                "key": f"chunk-{start // chunk_size}",
                "request": {
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": f"Classify:\n{json.dumps(payload, ensure_ascii=False)}"}]}],
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
        uploaded = api.files.upload(
            file=str(input_path),
            config=types.UploadFileConfig(display_name="product_type_batch", mime_type="jsonl"),
        )
        job = api.batches.create(
            model=MODEL_NAME,
            src=uploaded.name,
            config={"display_name": "product-type-job"},
        )
        batch_id = job.name

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while getattr(job, "state", "") not in terminal:
            time.sleep(poll_seconds)
            job = api.batches.get(name=batch_id)

        if getattr(job, "state", "") != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Product Type Batch failed: {getattr(job, 'error', '')}")

        res_file = getattr(getattr(job, "dest", None), "file_name", None)
        content = api.files.download(file=res_file).decode("utf-8")

        classifications: dict[int, str] = {}
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
                        item["product_type"] = normalize_product_type(item.get("product_type"))
                    parsed = BatchClassificationResponse.model_validate(raw)
                    for item in parsed.items:
                        classifications[item.row_id] = item.product_type
        return classifications, batch_id
    finally:
        input_path.unlink(missing_ok=True)


def classify_file(file_id: str, api: Any | None = None, chunk_size: int = CHUNK_SIZE, poll_seconds: float = 15.0, job_manager: Any | None = None) -> list[dict[str, Any]]:
    rows = database.rows(file_id)
    missing = [row["Sr. no."] for row in rows if not (row.get("Combined Description") or row.get("Secondary Product Description"))]
    if missing:
        raise RuntimeError(f"Separated product contexts are required before Product Type classification: {missing[:10]}")

    logger.event("PRODUCT_TYPE", "file_loaded", file_id=file_id, rows=len(rows))
    service = api or client()
    total_rows = len(rows)

    _done = {"n": 0}

    def _chunk_progress(count: int) -> None:
        if job_manager:
            _done["n"] += count
            progress = int((_done["n"] / max(1, total_rows)) * 100)
            job_manager.set(
                file_id, "RUNNING", "PRODUCT_TYPE", progress,
                f"Classifying product types: {_done['n']}/{total_rows}",
                progress_counter=f"{_done['n']}/{total_rows}"
            )

    if job_manager:
        job_manager.set(
            file_id, "RUNNING", "PRODUCT_TYPE", 0,
            f"Classifying product types: 0/{total_rows}",
            progress_counter=f"0/{total_rows}"
        )

    if len(rows) <= DIRECT_EXECUTION_THRESHOLD:
        classifications = direct_parallel_classify(rows, service, on_progress=_chunk_progress)
        batch_id = "direct-execution"
    else:
        classifications, batch_id = batch_classify(rows, service, chunk_size, poll_seconds)
        if job_manager:
            job_manager.set(
                file_id, "RUNNING", "PRODUCT_TYPE", 100,
                f"Classified {total_rows}/{total_rows} product types",
                progress_counter=f"{total_rows}/{total_rows}"
            )

    expected = {int(row["Sr. no."]) for row in rows}
    missing_ids = sorted(expected - classifications.keys())
    if missing_ids:
        raise RuntimeError(f"Product Type classification omitted row IDs: {missing_ids[:10]}")

    decisions = [{"row_id": str(row["Sr. no."]), "product_type": classifications[int(row["Sr. no."])]} for row in rows]
    database.update_product_types(file_id, batch_id, decisions)
    logger.event("PRODUCT_TYPE", "stored", file_id=file_id, rows=len(decisions), batch_id=batch_id)
    return decisions