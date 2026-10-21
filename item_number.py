from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

import database
import logger
from normalize import normalize_title

AllocationSource = Literal[
    "MASTER_REUSE",
    "INTRA_FILE_REUSE",
    "NEW_ALLOCATION",
    "BLOCKED_UNMAPPED_TAXONOMY",
]

MASTER_TITLE_COL = "Product Name(Cleaned Title)"
MASTER_ITEM_COL = "Generated Item Number"


def normalize_key(title: str) -> str:
    clean = normalize_title(title).casefold()
    clean = re.sub(r"[^\w\s%./+-]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


def load_master(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[MASTER_TITLE_COL, MASTER_ITEM_COL])
    df = pd.read_excel(p, dtype=str) if p.suffix.lower() in {".xlsx", ".xls", ".xlsb"} else pd.read_csv(p, dtype=str)
    df.columns = df.columns.str.strip()
    return df


def get_counters(df: pd.DataFrame) -> dict[str, int]:
    counters: dict[str, int] = {}
    for val in df[MASTER_ITEM_COL].dropna():
        item_no = str(val).strip().zfill(10)
        if len(item_no) != 10:
            continue
        base_code = item_no[:4]
        try:
            seq = int(item_no[4:])
            counters[base_code] = max(counters.get(base_code, 0), seq)
        except ValueError:
            continue
    return counters


def build_title_map(df: pd.DataFrame) -> dict[str, str]:
    title_map: dict[str, str] = {}
    for _, row in df.iterrows():
        title = str(row.get(MASTER_TITLE_COL, "")).strip()
        item_no = str(row.get(MASTER_ITEM_COL, "")).strip().zfill(10)
        if not title or len(item_no) != 10:
            continue
        norm_key = normalize_key(title)
        if norm_key:
            title_map[norm_key] = item_no
    return title_map


def assign_unicodes(
    file_id: str,
    check_master: bool = False,
    master_path: str = "master_latest.xlsx",
) -> list[dict[str, Any]]:
    rows = database.rows(file_id)
    if not rows:
        return []

    logger.event("UNICODE", "start", file_id=file_id, rows=len(rows), check_master=check_master)

    database.rollback_item_numbers(file_id)

    master_df = load_master(master_path)
    counters = get_counters(master_df)
    master_title_map = build_title_map(master_df) if check_master else {}

    intra_file_map: dict[str, str] = {}
    allocations: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row["Sr. no."])
        title = str(row.get("Title Pass 2") or row.get("Title Pass 1") or "").strip()
        base_code = str(row.get("2 Digit Base Code") or "").strip()
        norm_key = normalize_key(title)

        if not base_code or base_code == "UNMAPPED" or len(base_code) != 4:
            allocations.append({
                "row_id": row_id,
                "item_number": "BLOCKED",
                "item_number_source": "BLOCKED_UNMAPPED_TAXONOMY",
                "matched_master_title": "",
            })
            continue

        if norm_key in master_title_map:
            matched = master_title_map[norm_key]
            intra_file_map[norm_key] = matched
            allocations.append({
                "row_id": row_id,
                "item_number": matched,
                "item_number_source": "MASTER_REUSE",
                "matched_master_title": title,
            })
            continue

        if check_master and norm_key in intra_file_map:
            allocations.append({
                "row_id": row_id,
                "item_number": intra_file_map[norm_key],
                "item_number_source": "INTRA_FILE_REUSE",
                "matched_master_title": title,
            })
            continue

        seq = counters.get(base_code, 0) + 1
        counters[base_code] = seq
        new_item_no = f"{base_code}{seq:06d}"

        intra_file_map[norm_key] = new_item_no

        allocations.append({
            "row_id": row_id,
            "item_number": new_item_no,
            "item_number_source": "NEW_ALLOCATION",
            "matched_master_title": title,
        })

    persist_allocations(file_id, allocations, master_path)

    logger.event("UNICODE", "complete", file_id=file_id, allocated=len(allocations))
    return allocations


def persist_allocations(
    file_id: str,
    allocations: list[dict[str, Any]],
    master_path: str = "master_latest.xlsx",
) -> None:
    non_blocked = [a for a in allocations if a["item_number"] not in ("", "BLOCKED")]
    newly_minted = [a for a in non_blocked if a["item_number_source"] == "NEW_ALLOCATION"]

    if newly_minted:
        master_df = load_master(master_path)
        new_df = pd.DataFrame({
            MASTER_TITLE_COL: [a.get("matched_master_title", "") for a in newly_minted],
            MASTER_ITEM_COL: [a["item_number"] for a in newly_minted],
        })
        pd.concat([master_df, new_df], ignore_index=True).to_excel(master_path, index=False)

    database.update_item_numbers(file_id, allocations, [
        {
            "item_number": a["item_number"],
            "normalized_title": normalize_key(a.get("matched_master_title", "")),
            "original_title": a.get("matched_master_title", ""),
            "base_code_2digit": a["item_number"][:4],
            "level1": "",
            "level2": "",
        }
        for a in non_blocked
    ])


def rollback(file_id: str) -> None:
    database.rollback_item_numbers(file_id)
    logger.event("UNICODE", "rollback_complete", file_id=file_id)