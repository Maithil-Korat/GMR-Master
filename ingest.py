from __future__ import annotations

import hashlib
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import database


@dataclass(frozen=True)
class IngestResult:
    file_id: str
    file_name: str
    row_count: int
    columns: tuple[str, ...]


def headers(columns: list[object]) -> list[str]:
    result: list[str] = []
    seen: dict[str, int] = {}
    for index, column in enumerate(columns):
        name = str(column).strip() or f"Unnamed: {index}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}__{seen[name]}"
        result.append(name)
    return result


def read_file(path: str | Path) -> tuple[bytes, str, pd.DataFrame]:
    file_path = Path(path)
    data = file_path.read_bytes()
    suffix = file_path.suffix.lower()
    if suffix not in {".xls", ".xlsx", ".xlsb"}:
        raise ValueError("Supported file types are XLS, XLSX, and XLSB")
    engines = {".xls": "xlrd", ".xlsx": "openpyxl", ".xlsb": "pyxlsb"}
    table = pd.read_excel(
        io.BytesIO(data),
        engine=engines[suffix],
        dtype=str,
        keep_default_na=False,
    )
    return data, file_path.name, table


def aggregate(table: pd.DataFrame) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    frame = table.copy()
    frame.columns = headers(list(frame.columns))
    frame = frame.fillna("").astype(str)
    columns = tuple(frame.columns)
    rows = []
    for number, values in enumerate(frame.to_dict(orient="records"), start=1):
        context = " ".join(f"[{column}: {value}]" for column, value in values.items() if value != "")
        rows.append(
            {
                "source_row": values,
                "universal_raw_context": context,
                "Sr. no.": str(number),
            }
        )
    return rows, columns


def parse_text_to_dataframe(raw_text: str) -> pd.DataFrame:
    text = raw_text.strip()
    if not text:
        raise ValueError("Pasted text is empty.")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No valid text lines found.")

    if "\t" in lines[0]:
        return pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False).fillna("")

    if "," in lines[0] and len(lines[0].split(",")) > 2:
        return pd.read_csv(io.StringIO(text), sep=",", dtype=str, keep_default_na=False).fillna("")


    split_index = None
    date_or_code_pattern = re.compile(r"^(\d{2}[-/]\d{2}[-/]\d{4}|\d{6,14})$")
    
    for idx, line in enumerate(lines):
        if idx > 0 and date_or_code_pattern.match(line):
            split_index = idx
            break

    if split_index is not None and split_index > 1:
        col_headers = lines[:split_index]
        data_values = lines[split_index:]
        num_cols = len(col_headers)

        # Chunk the remaining values into rows of size num_cols
        rows = []
        for i in range(0, len(data_values), num_cols):
            chunk = data_values[i : i + num_cols]
            if len(chunk) < num_cols:
                # Pad with empty strings if trailing row is incomplete
                chunk += [""] * (num_cols - len(chunk))
            rows.append(chunk)

        return pd.DataFrame(rows, columns=col_headers)

    return pd.DataFrame({"Description": lines})


def ingest_text(raw_text: str, file_name: str = "Pasted_Catalog_Data.xlsx") -> IngestResult:
    table = parse_text_to_dataframe(raw_text)
    
    # Generate in-memory Excel bytes for archive
    excel_buf = io.BytesIO()
    table.to_excel(excel_buf, index=False, engine="openpyxl")
    data = excel_buf.getvalue()

    rows, cols = aggregate(table)
    file_id = uuid.uuid4().hex

    database.save(
        file_id=file_id,
        file_name=file_name,
        file_hash=hashlib.sha256(data).hexdigest(),
        file_bytes=data,
        source_columns=cols,
        rows=rows,
    )
    return IngestResult(file_id, file_name, len(rows), cols)


def ingest(path: str | Path) -> IngestResult:
    data, file_name, table = read_file(path)
    rows, columns = aggregate(table)
    file_id = uuid.uuid4().hex
    database.save(
        file_id=file_id,
        file_name=file_name,
        file_hash=hashlib.sha256(data).hexdigest(),
        file_bytes=data,
        source_columns=columns,
        rows=rows,
    )
    return IngestResult(file_id, file_name, len(rows), columns)
