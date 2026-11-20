from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

import attribute_schema
import database
import logger
from normalize import is_liquor_product, normalize_attribute, normalize_units

PURPLE_FILL = PatternFill(start_color="E1D5E7", end_color="E1D5E7", fill_type="solid")
FLAG_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)


def export_catalog(
    file_id: str,
    schema_path: str | Path = "attributes.json",
    output_path: str | Path | None = None,
) -> Path:
    rows = database.rows(file_id)
    if not rows:
        raise ValueError(f"No records found for file_id: {file_id}")

    source_columns = database.columns(file_id)
    schema_map = attribute_schema.load(schema_path)
    target_path = Path(output_path or f"catalog_export_{file_id[:8]}.xlsx").resolve()
    logger.event("EXPORT", "start", file_id=file_id, rows=len(rows), target=str(target_path))

    export_data = []
    for r in rows:
        content_p2 = r.get("Content Pass 2") or {}
        attrs_p2 = r.get("Attributes Pass 2") or r.get("Attributes Pass 1") or {}
        pt = str(r.get("Product Type") or "")
        liquor = is_liquor_product(pt)
        if liquor:
            content_p2 = {
                name: normalize_units(value, liquor=True)
                if isinstance(value, str)
                else value
                for name, value in content_p2.items()
            }

        # 1. Always start with Sr. no. as Column 1
        item_row: dict[str, Any] = {
            "Sr. no.": r["Sr. no."]
        }

        # 2. Preserve all original client columns dynamically
        for col in source_columns:
            if col.strip().casefold() not in {"sr. no.", "sr no", "sr.no."}:
                item_row[col] = r.get(col, "")

        # 3. Build Canonical Attributes Dict & Key:Value pairs
        canonical_attr_names = attribute_schema.names(schema_map, pt) if pt in schema_map else []
        canonical_attrs_dict: dict[str, str] = {}
        attr_kv_list: list[str] = []

        for name in canonical_attr_names:
            val = normalize_attribute(name, attrs_p2.get(name, "NA"), liquor=liquor)
            canonical_attrs_dict[name] = val
            attr_kv_list.append(f"{name}: {val}")

        # Pad up to 10 slots with "NA" if category has fewer than 10 attributes
        while len(attr_kv_list) < 10:
            attr_kv_list.append("NA")

        # 4. JSON string of all canonical attributes (even if NA)
        attributes_given_json = json.dumps(canonical_attrs_dict, ensure_ascii=False)

        l1 = str(r.get("Level 1") or "")
        l2 = str(r.get("Level 2") or "")
        l3 = str(r.get("Level 3") or "")
        l4 = str(r.get("Level 4") or "")
        l5 = str(r.get("Level 5") or "")
        composite_l1_l2 = f"{l1}>{l2}".strip(">") if (l1 or l2) else ""

        # 5. Append Enriched Catalog Columns
        item_row.update({
            "Product Type": pt,
            "Item Type": r.get("Item Type", ""),
            "Combined Description": r.get("Combined Description", ""),
            "Combined Barcode Description": r.get("Secondary Product Description", ""),
            "Cleaned Titles": r.get("Title Pass 2") or r.get("Title Pass 1", ""),
            "Attributes Given": attributes_given_json,
            "Product Size": content_p2.get("product_size", ""),
            "Brand Name": content_p2.get("brand_name", ""),
            "Parent Company": content_p2.get("parent_company", ""),
            "Marketplace Title": content_p2.get("marketplace_title", ""),
            "Invoice Name": r.get("Invoice Name", ""),
            "Short Description": content_p2.get("short_description", ""),
            "Long Description": content_p2.get("long_description", ""),
            "HSN Code": content_p2.get("hsn_code", ""),
            "Country of Origin": content_p2.get("country_of_origin", ""),
            "Attribute 1": attr_kv_list[0],
            "Attribute 2": attr_kv_list[1],
            "Attribute 3": attr_kv_list[2],
            "Attribute 4": attr_kv_list[3],
            "Attribute 5": attr_kv_list[4],
            "Attribute 6": attr_kv_list[5],
            "Attribute 7": attr_kv_list[6],
            "Attribute 8": attr_kv_list[7],
            "Attribute 9": attr_kv_list[8],
            "Attribute 10": attr_kv_list[9],
            "Level 1": l1,
            "GMR Taxonomy": r.get("GMR Taxonomy", ""),
            "Unnamed: 33": composite_l1_l2,
            "Unnamed: 34": l1,
            "Unnamed: 35": l2,
            "Unnamed: 36": l3,
            "Unnamed: 37": l4,
            "Unnamed: 38": l5,
            "2 DIGIT CODE": r.get("2 Digit Base Code", ""),
            "Generated Item Number": r.get("item_number", ""),
        })

        export_data.append(item_row)

    df = pd.DataFrame(export_data)
    df.to_excel(target_path, index=False, engine="openpyxl")

    wb = load_workbook(target_path)
    ws = wb.active
    ws.title = "Master Catalog Export"

    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    item_num_col_idx = df.columns.get_loc("Generated Item Number") + 1

    for row_idx, r in enumerate(rows, start=2):
        source = str(r.get("item_number_source", ""))
        item_no_cell = ws.cell(row=row_idx, column=item_num_col_idx)
        if source == "INTRA_FILE_REUSE":
            item_no_cell.fill = PURPLE_FILL
        elif source == "BLOCKED_UNMAPPED_TAXONOMY":
            item_no_cell.fill = FLAG_FILL

        for col_idx in range(1, len(df.columns) + 1):
            ws.cell(row=row_idx, column=col_idx).border = THIN_BORDER

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 50)

    wb.save(target_path)
    logger.event("EXPORT", "completed", path=str(target_path))
    return target_path