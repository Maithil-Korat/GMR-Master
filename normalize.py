from __future__ import annotations

import re
import unicodedata

NA_VALUES = {"", "NA", "N/A", "NOT AVAILABLE", "NONE", "NULL", "UNRESOLVED"}
LIQUOR_PRODUCT_TYPES = {
    "Whisky",
    "Vodka",
    "Gin",
    "Rum",
    "Tequila & Mezcal",
    "Brandy & Cognac",
    "Aperitifs & Vermouth",
    "Liqueurs & Cordials",
    "Wine & Champagne",
    "Non-Alcoholic (0.0%)",
    "Ready to Drink",
}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def is_liquor_product(product_type: object) -> bool:
    text = normalize_text(product_type)
    return text in LIQUOR_PRODUCT_TYPES or text.casefold().startswith("liquor>")


def normalize_units(value: object, liquor: bool = False) -> str:
    text = normalize_text(value)
    if liquor:
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*(?:L|LT|LITRE|LITER|LITRES|LITERS)\b",
            lambda match: f"{float(match.group(1)) * 100:g}CL",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*ML\b",
            lambda match: f"{float(match.group(1)) / 10:g}CL",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(CL|ML|KG|MM|G|OZ)\b", lambda match: f"{match.group(1)}x{match.group(2)}{match.group(3).upper()}", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(CL)\b", r"\1CL", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(ML|KG|MM|G|OZ)\b", lambda match: f"{match.group(1)} {match.group(2).lower()}", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1%", text)
    return normalize_text(text)


def normalize_item_type(value: object) -> str:
    text = normalize_text(value)
    values = {"saleable": "Saleable", "non-saleable": "Non-Saleable"}
    return values.get(text.casefold(), text)


def normalize_product_type(value: object) -> str:
    return normalize_text(value)


def normalize_title(value: object, liquor: bool = False) -> str:
    return normalize_units(value, liquor=liquor)


def ascii_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_invoice(value: object, limit: int = 33, liquor: bool = False) -> str:
    text = ascii_text(normalize_units(value, liquor=liquor)).upper()
    text = re.sub(r"[^A-Z0-9& ]", " ", text)
    text = normalize_text(text)
    return text[:limit].rstrip()


def normalize_attribute(name: str, value: object, liquor: bool = False) -> str:
    text = normalize_text(value)
    if text.upper() in NA_VALUES:
        return "NA"
    text = normalize_units(text, liquor=liquor)
    if "%" in name and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text}%"
    if name.casefold().endswith("_cl") and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text}CL"
    if name.casefold().endswith("_ml") and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text} ml"
    if name.casefold().endswith("_g") and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text} g"
    if name.casefold().endswith("_kg") and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text} kg"
    if name.casefold().endswith("_mm") and re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text} mm"
    return text


def changed(before: object, after: object) -> bool:
    return str(before or "") != str(after or "")
