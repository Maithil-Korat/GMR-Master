from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class AttributeDefinition(BaseModel):
    name: str
    description: str


class CategoryDefinition(BaseModel):
    main_category: str
    attributes: list[AttributeDefinition]


def load(path: str | Path = "attributes.json") -> dict[str, CategoryDefinition]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Attribute schema not found: {file}")
    source = json.loads(file.read_text(encoding="utf-8"))
    definitions = [CategoryDefinition.model_validate(item) for item in source]
    result = {}
    for category in definitions:
        if category.main_category in result:
            raise ValueError(f"Duplicate attribute category: {category.main_category}")
        names = [attribute.name for attribute in category.attributes]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Invalid attributes for category: {category.main_category}")
        result[category.main_category] = category
    return result


def names(schema: dict[str, CategoryDefinition], category: str) -> list[str]:
    if category not in schema:
        raise KeyError(f"No attribute schema for Product Type: {category}")
    return [attribute.name for attribute in schema[category].attributes]


def descriptions(schema: dict[str, CategoryDefinition], category: str) -> dict[str, str]:
    if category not in schema:
        raise KeyError(f"No attribute schema for Product Type: {category}")
    return {attribute.name: attribute.description for attribute in schema[category].attributes}
