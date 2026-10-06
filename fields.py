from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel, Field
import database
import logger

FieldRole = Literal[
    "PRODUCT_DESCRIPTION", "SECONDARY_PRODUCT_DESCRIPTION", "BRAND", "VARIANT",
    "PHYSICAL_SPECIFICATION", "PRODUCT_CODE", "ITEM_TYPE_SIGNAL", "CATEGORY_HINT",
    "VENDOR_METADATA", "IDENTIFIER", "OTHER_METADATA", "UNKNOWN",
]

PRIMARY_KEYWORDS = {
    "desc", "description", "title", "name", "style", "style no", "model",
    "item name", "item_description", "product", "specs", "variant", "color"
}
SECONDARY_KEYWORDS = {"short_desc", "pos_desc", "alt_desc", "secondary"}
METADATA_KEYWORDS = {
    "vendor", "supplier", "hsn", "upc", "ean", "barcode", "location",
    "warehouse", "sku", "id", "created", "date", "quantity", "stock"
}
CATEGORY_KEYWORDS = {"division", "category", "dept", "department", "class", "group"}


class FieldMapping(BaseModel):
    column: str
    role: FieldRole
    use_in_primary: bool
    use_in_secondary: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


def auto_map_column(column_name: str) -> FieldMapping:
    clean = re.sub(r"[^\w\s]", " ", column_name.casefold()).strip()
    words = set(clean.split())

    if words.intersection(CATEGORY_KEYWORDS):
        return FieldMapping(column=column_name, role="CATEGORY_HINT", use_in_primary=False, use_in_secondary=False)
    if words.intersection(METADATA_KEYWORDS) or "use this" in clean or "barcode" in clean:
        return FieldMapping(column=column_name, role="VENDOR_METADATA" if "vendor" in clean or "supplier" in clean else "IDENTIFIER", use_in_primary=False, use_in_secondary=False)
    if words.intersection(PRIMARY_KEYWORDS) or "style" in clean or "desc" in clean:
        return FieldMapping(column=column_name, role="PRODUCT_DESCRIPTION", use_in_primary=True, use_in_secondary=False)
    if words.intersection(SECONDARY_KEYWORDS):
        return FieldMapping(column=column_name, role="SECONDARY_PRODUCT_DESCRIPTION", use_in_primary=False, use_in_secondary=True)

    return FieldMapping(column=column_name, role="UNKNOWN", use_in_primary=False, use_in_secondary=False)


def map_file(file_id: str) -> list[dict[str, Any]]:
    source_columns = database.columns(file_id)
    mappings = [auto_map_column(col).model_dump() for col in source_columns]
    logger.event("FIELD_MAP", "heuristic_complete", file_id=file_id, columns=len(mappings))
    return mappings