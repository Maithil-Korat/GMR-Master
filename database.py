from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path("catalog.db")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def create_table(connection: sqlite3.Connection) -> None:
    # 1. Main Batch Rows Table
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_rows (
            file_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            source_file_bytes BLOB,
            source_columns_json TEXT NOT NULL,
            sr_no INTEGER NOT NULL,
            source_row_json TEXT NOT NULL,
            universal_raw_context TEXT NOT NULL,
            parser_candidate TEXT,
            parser_evidence_json TEXT NOT NULL DEFAULT '[]',
            item_type TEXT,
            item_type_evidence_json TEXT NOT NULL DEFAULT '[]',
            item_type_confidence REAL,
            item_type_review_required INTEGER NOT NULL DEFAULT 0,
            item_type_notes TEXT NOT NULL DEFAULT '',
            product_type TEXT,
            product_type_review_required INTEGER NOT NULL DEFAULT 0,
            product_type_batch_id TEXT,
            field_map_json TEXT NOT NULL DEFAULT '{}',
            field_map_batch_id TEXT,
            combined_description TEXT NOT NULL DEFAULT '',
            secondary_product_description TEXT NOT NULL DEFAULT '',
            title_pass1 TEXT,
            title_pass1_batch_id TEXT,
            title_pass1_quality TEXT,
            title_pass1_evidence_json TEXT NOT NULL DEFAULT '[]',
            title_pass2 TEXT,
            title_pass2_batch_id TEXT,
            title_quality_flag TEXT,
            title_review_required INTEGER NOT NULL DEFAULT 0,
            title_audit_notes TEXT NOT NULL DEFAULT '',
            attributes_pass1_json TEXT NOT NULL DEFAULT '{}',
            attributes_pass1_evidence_json TEXT NOT NULL DEFAULT '{}',
            attributes_pass1_batch_id TEXT,
            attributes_pass2_json TEXT NOT NULL DEFAULT '{}',
            attributes_pass2_batch_id TEXT,
            attributes_quality_flag TEXT,
            attributes_review_required INTEGER NOT NULL DEFAULT 0,
            attributes_audit_notes TEXT NOT NULL DEFAULT '',
            content_pass1_json TEXT NOT NULL DEFAULT '{}',
            content_pass1_batch_id TEXT,
            content_pass2_json TEXT NOT NULL DEFAULT '{}',
            content_pass2_batch_id TEXT,
            content_quality_flag TEXT,
            content_review_required INTEGER NOT NULL DEFAULT 0,
            content_audit_notes TEXT NOT NULL DEFAULT '',
            invoice_name TEXT,
            taxonomy_pass1 TEXT,
            taxonomy_pass1_batch_id TEXT,
            taxonomy_pass1_quality TEXT,
            taxonomy_pass1_evidence_json TEXT NOT NULL DEFAULT '[]',
            taxonomy_pass2 TEXT,
            taxonomy_pass2_batch_id TEXT,
            taxonomy_quality_flag TEXT,
            taxonomy_review_required INTEGER NOT NULL DEFAULT 0,
            taxonomy_audit_notes TEXT NOT NULL DEFAULT '',
            gmr_taxonomy TEXT,
            base_code_2digit TEXT,
            level1 TEXT,
            level2 TEXT,
            level3 TEXT,
            level4 TEXT,
            level5 TEXT,
            item_number TEXT,
            item_number_source TEXT,
            matched_master_title TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (file_id, sr_no)
        )
        """
    )

    # 2. Master Catalog Table (Single-statement executions)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS master_catalog (
            item_number TEXT PRIMARY KEY,
            normalized_title TEXT NOT NULL,
            original_title TEXT NOT NULL,
            base_code_2digit TEXT NOT NULL,
            level1 TEXT,
            level2 TEXT,
            source_file_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_master_base_code ON master_catalog(base_code_2digit)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_master_norm_title ON master_catalog(normalized_title)")

    # 3. Dynamic schema migration for existing SQLite databases
    columns = {row[1] for row in connection.execute("PRAGMA table_info(raw_rows)")}
    additions = {
        "parser_candidate": "TEXT",
        "parser_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "item_type": "TEXT",
        "item_type_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "item_type_confidence": "REAL",
        "item_type_review_required": "INTEGER NOT NULL DEFAULT 0",
        "item_type_notes": "TEXT NOT NULL DEFAULT ''",
        "product_type": "TEXT",
        "product_type_review_required": "INTEGER NOT NULL DEFAULT 0",
        "product_type_batch_id": "TEXT",
        "field_map_json": "TEXT NOT NULL DEFAULT '{}'",
        "field_map_batch_id": "TEXT",
        "combined_description": "TEXT NOT NULL DEFAULT ''",
        "secondary_product_description": "TEXT NOT NULL DEFAULT ''",
        "title_pass1": "TEXT",
        "title_pass1_batch_id": "TEXT",
        "title_pass1_quality": "TEXT",
        "title_pass1_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "title_pass2": "TEXT",
        "title_pass2_batch_id": "TEXT",
        "title_quality_flag": "TEXT",
        "title_review_required": "INTEGER NOT NULL DEFAULT 0",
        "title_audit_notes": "TEXT NOT NULL DEFAULT ''",
        "attributes_pass1_json": "TEXT NOT NULL DEFAULT '{}'",
        "attributes_pass1_evidence_json": "TEXT NOT NULL DEFAULT '{}'",
        "attributes_pass1_batch_id": "TEXT",
        "attributes_pass2_json": "TEXT NOT NULL DEFAULT '{}'",
        "attributes_pass2_batch_id": "TEXT",
        "attributes_quality_flag": "TEXT",
        "attributes_review_required": "INTEGER NOT NULL DEFAULT 0",
        "attributes_audit_notes": "TEXT NOT NULL DEFAULT ''",
        "content_pass1_json": "TEXT NOT NULL DEFAULT '{}'",
        "content_pass1_batch_id": "TEXT",
        "content_pass2_json": "TEXT NOT NULL DEFAULT '{}'",
        "content_pass2_batch_id": "TEXT",
        "content_quality_flag": "TEXT",
        "content_review_required": "INTEGER NOT NULL DEFAULT 0",
        "content_audit_notes": "TEXT NOT NULL DEFAULT ''",
        "invoice_name": "TEXT",
        "taxonomy_pass1": "TEXT",
        "taxonomy_pass1_batch_id": "TEXT",
        "taxonomy_pass1_quality": "TEXT",
        "taxonomy_pass1_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "taxonomy_pass2": "TEXT",
        "taxonomy_pass2_batch_id": "TEXT",
        "taxonomy_quality_flag": "TEXT",
        "taxonomy_review_required": "INTEGER NOT NULL DEFAULT 0",
        "taxonomy_audit_notes": "TEXT NOT NULL DEFAULT ''",
        "gmr_taxonomy": "TEXT",
        "base_code_2digit": "TEXT",
        "level1": "TEXT",
        "level2": "TEXT",
        "level3": "TEXT",
        "level4": "TEXT",
        "level5": "TEXT",
        "item_number": "TEXT",
        "item_number_source": "TEXT",
        "matched_master_title": "TEXT",
    }
    for name, definition in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE raw_rows ADD COLUMN {name} {definition}")


def save(
    file_id: str,
    file_name: str,
    file_hash: str,
    file_bytes: bytes,
    source_columns: Iterable[str],
    rows: Iterable[dict[str, str]],
) -> None:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = json.dumps(list(source_columns), ensure_ascii=False)
    values = list(rows)
    if not values:
        raise ValueError("The workbook contains no data rows")
    records = [
        (
            file_id,
            file_name,
            file_hash,
            file_bytes if index == 0 else None,
            columns,
            index + 1,
            json.dumps(row["source_row"], ensure_ascii=False),
            row["universal_raw_context"],
            created_at,
        )
        for index, row in enumerate(values)
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            INSERT INTO raw_rows (
                file_id, file_name, file_hash, source_file_bytes, source_columns_json,
                sr_no, source_row_json, universal_raw_context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def rows(file_id: str) -> list[dict[str, str]]:
    with connect() as connection:
        records = connection.execute(
            """
            SELECT sr_no, source_row_json, universal_raw_context, parser_candidate,
                   parser_evidence_json, item_type, item_type_evidence_json,
                   item_type_confidence, item_type_review_required, item_type_notes,
                   product_type, product_type_review_required, product_type_batch_id,
                   field_map_json, field_map_batch_id, combined_description, secondary_product_description,
                   title_pass1, title_pass1_batch_id, title_pass1_quality, title_pass1_evidence_json,
                   title_pass2, title_pass2_batch_id, title_quality_flag, title_review_required, title_audit_notes,
                   attributes_pass1_json, attributes_pass1_evidence_json, attributes_pass1_batch_id,
                   attributes_pass2_json, attributes_pass2_batch_id, attributes_quality_flag,
                   attributes_review_required, attributes_audit_notes,
                   content_pass1_json, content_pass1_batch_id, content_pass2_json,
                   content_pass2_batch_id, content_quality_flag, content_review_required,
                   content_audit_notes, invoice_name,
                   taxonomy_pass1, taxonomy_pass1_batch_id, taxonomy_pass1_quality, taxonomy_pass1_evidence_json,
                   taxonomy_pass2, taxonomy_pass2_batch_id, taxonomy_quality_flag, taxonomy_review_required,
                   taxonomy_audit_notes, gmr_taxonomy, base_code_2digit, level1, level2, level3, level4, level5,
                   item_number, item_number_source, matched_master_title
            FROM raw_rows
            WHERE file_id = ?
            ORDER BY sr_no
            """,
            (file_id,),
        ).fetchall()
    result = []
    for record in records:
        result.append(
            {
                "Sr. no.": str(record["sr_no"]),
                **json.loads(record["source_row_json"]),
                "Universal_Raw_Context": record["universal_raw_context"],
                "Parser Item Type": record["parser_candidate"],
                "Parser Evidence": json.loads(record["parser_evidence_json"] or "[]"),
                "Item Type": record["item_type"],
                "Item Type Evidence": json.loads(record["item_type_evidence_json"] or "[]"),
                "Item Type Confidence": record["item_type_confidence"],
                "Item Type Review Required": bool(record["item_type_review_required"]),
                "Item Type Notes": record["item_type_notes"],
                "Product Type": record["product_type"],
                "Product Type Review Required": bool(record["product_type_review_required"]),
                "Product Type Batch ID": record["product_type_batch_id"],
                "Field Map": json.loads(record["field_map_json"] or "{}"),
                "Field Map Batch ID": record["field_map_batch_id"],
                "Combined Description": record["combined_description"],
                "Secondary Product Description": record["secondary_product_description"],
                "Title Pass 1": record["title_pass1"],
                "Title Pass 1 Batch ID": record["title_pass1_batch_id"],
                "Title Pass 1 Quality": record["title_pass1_quality"],
                "Title Pass 1 Evidence": json.loads(record["title_pass1_evidence_json"] or "[]"),
                "Title Pass 2": record["title_pass2"],
                "Title Pass 2 Batch ID": record["title_pass2_batch_id"],
                "Title Quality Flag": record["title_quality_flag"],
                "Title Review Required": bool(record["title_review_required"]),
                "Title Audit Notes": record["title_audit_notes"],
                "Attributes Pass 1": json.loads(record["attributes_pass1_json"] or "{}"),
                "Attributes Pass 1 Evidence": json.loads(record["attributes_pass1_evidence_json"] or "{}"),
                "Attributes Pass 1 Batch ID": record["attributes_pass1_batch_id"],
                "Attributes Pass 2": json.loads(record["attributes_pass2_json"] or "{}"),
                "Attributes Pass 2 Batch ID": record["attributes_pass2_batch_id"],
                "Attributes Quality Flag": record["attributes_quality_flag"],
                "Attributes Review Required": bool(record["attributes_review_required"]),
                "Attributes Audit Notes": record["attributes_audit_notes"],
                "Content Pass 1": json.loads(record["content_pass1_json"] or "{}"),
                "Content Pass 1 Batch ID": record["content_pass1_batch_id"],
                "Content Pass 2": json.loads(record["content_pass2_json"] or "{}"),
                "Content Pass 2 Batch ID": record["content_pass2_batch_id"],
                "Content Quality Flag": record["content_quality_flag"],
                "Content Review Required": bool(record["content_review_required"]),
                "Content Audit Notes": record["content_audit_notes"],
                "Invoice Name": record["invoice_name"],
                "Taxonomy Pass 1": record["taxonomy_pass1"],
                "Taxonomy Pass 1 Batch ID": record["taxonomy_pass1_batch_id"],
                "Taxonomy Pass 1 Quality": record["taxonomy_pass1_quality"],
                "Taxonomy Pass 1 Evidence": json.loads(record["taxonomy_pass1_evidence_json"] or "[]"),
                "Taxonomy Pass 2": record["taxonomy_pass2"],
                "Taxonomy Pass 2 Batch ID": record["taxonomy_pass2_batch_id"],
                "Taxonomy Quality Flag": record["taxonomy_quality_flag"],
                "Taxonomy Review Required": bool(record["taxonomy_review_required"]),
                "Taxonomy Audit Notes": record["taxonomy_audit_notes"],
                "GMR Taxonomy": record["gmr_taxonomy"],
                "2 Digit Base Code": record["base_code_2digit"],
                "Level 1": record["level1"],
                "Level 2": record["level2"],
                "Level 3": record["level3"],
                "Level 4": record["level4"],
                "Level 5": record["level5"],
                "item_number": record["item_number"],
                "item_number_source": record["item_number_source"],
                "matched_master_title": record["matched_master_title"],
            }
        )
    return result


def source(file_id: str) -> bytes:
    with connect() as connection:
        record = connection.execute(
            "SELECT source_file_bytes FROM raw_rows WHERE file_id = ? AND source_file_bytes IS NOT NULL LIMIT 1",
            (file_id,),
        ).fetchone()
    if record is None:
        raise KeyError(file_id)
    return bytes(record["source_file_bytes"])


def columns(file_id: str) -> list[str]:
    with connect() as connection:
        record = connection.execute(
            "SELECT source_columns_json FROM raw_rows WHERE file_id = ? LIMIT 1",
            (file_id,),
        ).fetchone()
    if record is None:
        raise KeyError(file_id)
    return json.loads(record["source_columns_json"])


def update_item_types(file_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            decision["parser_candidate"],
            json.dumps(decision["parser_evidence"], ensure_ascii=False),
            decision["item_type"],
            json.dumps(decision["evidence"], ensure_ascii=False),
            decision["confidence_score"],
            int(bool(decision["review_required"])),
            decision["decision_notes"],
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET parser_candidate = ?, parser_evidence_json = ?, item_type = ?,
                item_type_evidence_json = ?, item_type_confidence = ?,
                item_type_review_required = ?, item_type_notes = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_product_types(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            decision["product_type"],
            int(bool(decision.get("review_required", False))),
            batch_id,
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET product_type = ?, product_type_review_required = ?, product_type_batch_id = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_contexts(
    file_id: str,
    batch_id: str,
    field_map: dict[str, object],
    contexts: Iterable[dict[str, object]],
) -> None:
    values = [
        (
            json.dumps(field_map, ensure_ascii=False),
            batch_id,
            str(context["combined_description"]),
            str(context["secondary_product_description"]),
            file_id,
            str(context["row_id"]),
        )
        for context in contexts
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET field_map_json = ?, field_map_batch_id = ?, combined_description = ?,
                secondary_product_description = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_title_drafts(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            decision["cleaned_title"],
            batch_id,
            decision["quality_flag"],
            json.dumps(decision["evidence"], ensure_ascii=False),
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET title_pass1 = ?, title_pass1_batch_id = ?, title_pass1_quality = ?,
                title_pass1_evidence_json = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_title_audits(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            decision["refined_title"],
            batch_id,
            decision["refinement_flag"],
            int(bool(decision["review_required"])),
            decision["audit_notes"],
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET title_pass2 = ?, title_pass2_batch_id = ?, title_quality_flag = ?,
                title_review_required = ?, title_audit_notes = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_attribute_drafts(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            json.dumps(decision["attributes"], ensure_ascii=False),
            json.dumps(decision["evidence"], ensure_ascii=False),
            batch_id,
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET attributes_pass1_json = ?, attributes_pass1_evidence_json = ?, attributes_pass1_batch_id = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_attribute_audits(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            json.dumps(decision["attributes"], ensure_ascii=False),
            batch_id,
            decision["quality_flag"],
            int(bool(decision["review_required"])),
            decision["audit_notes"],
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET attributes_pass2_json = ?, attributes_pass2_batch_id = ?, attributes_quality_flag = ?,
                attributes_review_required = ?, attributes_audit_notes = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def invoice_names(exclude_file_id: str | None = None) -> set[str]:
    query = "SELECT invoice_name FROM raw_rows WHERE invoice_name IS NOT NULL AND invoice_name != ''"
    values: tuple[str, ...] = ()
    if exclude_file_id is not None:
        query += " AND file_id != ?"
        values = (exclude_file_id,)
    with connect() as connection:
        return {str(row["invoice_name"]) for row in connection.execute(query, values).fetchall()}


def update_content_drafts(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (json.dumps(decision, ensure_ascii=False), batch_id, file_id, str(decision["row_id"]))
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            "UPDATE raw_rows SET content_pass1_json = ?, content_pass1_batch_id = ? WHERE file_id = ? AND sr_no = ?",
            values,
        )


def update_content_audits(file_id: str, batch_id: str, decisions: Iterable[dict[str, object]]) -> None:
    values = [
        (
            json.dumps(decision["content"], ensure_ascii=False),
            batch_id,
            decision["quality_flag"],
            int(bool(decision["review_required"])),
            decision["audit_notes"],
            decision["content"]["invoice_name"],
            file_id,
            str(decision["row_id"]),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET content_pass2_json = ?, content_pass2_batch_id = ?, content_quality_flag = ?,
                content_review_required = ?, content_audit_notes = ?, invoice_name = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_taxonomy_drafts(file_id: str, batch_id: str, decisions: Iterable[dict[str, Any]]) -> None:
    values = [
        (
            str(decision.get("gmr_taxonomy", "")),
            batch_id,
            str(decision.get("quality_flag", "CLEAN")),
            json.dumps(decision.get("evidence", []), ensure_ascii=False),
            file_id,
            str(decision.get("row_id", "")),
        )
        for decision in decisions
    ]
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET taxonomy_pass1 = ?, taxonomy_pass1_batch_id = ?, taxonomy_pass1_quality = ?,
                taxonomy_pass1_evidence_json = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def update_taxonomy_audits(file_id: str, batch_id: str, decisions: Iterable[dict[str, Any]]) -> None:
    values = []
    for decision in decisions:
        raw_levels = decision.get("levels")
        levels: dict[str, Any] = raw_levels if isinstance(raw_levels, dict) else {}
        values.append(
            (
                str(decision.get("gmr_taxonomy", "")),
                batch_id,
                str(decision.get("quality_flag", "")),
                int(bool(decision.get("review_required", False))),
                str(decision.get("audit_notes", "")),
                str(decision.get("gmr_taxonomy", "")),
                str(decision.get("base_code_2digit", "")),
                str(levels.get("level1", "")),
                str(levels.get("level2", "")),
                str(levels.get("level3", "")),
                str(levels.get("level4", "")),
                str(levels.get("level5", "")),
                file_id,
                str(decision.get("row_id", "")),
            )
        )
    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET taxonomy_pass2 = ?, taxonomy_pass2_batch_id = ?, taxonomy_quality_flag = ?,
                taxonomy_review_required = ?, taxonomy_audit_notes = ?,
                gmr_taxonomy = ?, base_code_2digit = ?,
                level1 = ?, level2 = ?, level3 = ?, level4 = ?, level5 = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            values,
        )


def load_master_records(base_code_filter: str | None = None) -> list[dict[str, str]]:
    query = "SELECT item_number, normalized_title, original_title, base_code_2digit, level1, level2 FROM master_catalog"
    params: tuple[Any, ...] = ()
    if base_code_filter:
        query += " WHERE base_code_2digit = ?"
        params = (base_code_filter,)
    with connect() as connection:
        create_table(connection)
        records = connection.execute(query, params).fetchall()
        return [dict(r) for r in records]


def get_max_sequences() -> dict[str, int]:
    with connect() as connection:
        create_table(connection)
        records = connection.execute(
            """
            SELECT base_code_2digit, item_number 
            FROM master_catalog 
            WHERE LENGTH(item_number) = 10
            """
        ).fetchall()

    counters: dict[str, int] = {}
    for r in records:
        prefix = str(r["base_code_2digit"]).strip()
        num_str = str(r["item_number"]).strip()
        if len(num_str) == 10 and num_str.startswith(prefix):
            try:
                seq = int(num_str[len(prefix):])
                counters[prefix] = max(counters.get(prefix, 0), seq)
            except ValueError:
                continue
    return counters


def get_live_sequences(file_id: str) -> dict[str, int]:
    counters = get_max_sequences()

    with connect() as connection:
        records = connection.execute(
            """
            SELECT base_code_2digit AS prefix, item_number
            FROM raw_rows
            WHERE file_id = ? AND LENGTH(item_number) = 10
            """,
            (file_id,),
        ).fetchall()

    for r in records:
        prefix = str(r["prefix"]).strip()
        num_str = str(r["item_number"]).strip()
        if num_str.startswith(prefix):
            try:
                seq = int(num_str[len(prefix):])
                counters[prefix] = max(counters.get(prefix, 0), seq)
            except ValueError:
                continue
    return counters


def update_item_numbers(
    file_id: str,
    allocations: Iterable[dict[str, Any]],
    new_master_entries: Iterable[dict[str, Any]],
) -> None:
    row_updates = [
        (
            item["item_number"],
            item["item_number_source"],
            item.get("matched_master_title", ""),
            file_id,
            str(item["row_id"]),
        )
        for item in allocations
    ]
    master_records = [
        (
            entry["item_number"],
            entry["normalized_title"],
            entry["original_title"],
            entry["base_code_2digit"],
            entry.get("level1", ""),
            entry.get("level2", ""),
            file_id,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        for entry in new_master_entries
    ]

    with connect() as connection:
        create_table(connection)
        connection.executemany(
            """
            UPDATE raw_rows
            SET item_number = ?, item_number_source = ?, matched_master_title = ?
            WHERE file_id = ? AND sr_no = ?
            """,
            row_updates,
        )
        if master_records:
            connection.executemany(
                """
                INSERT OR REPLACE INTO master_catalog (
                    item_number, normalized_title, original_title, base_code_2digit,
                    level1, level2, source_file_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                master_records,
            )


def rollback_item_numbers(file_id: str) -> None:
    with connect() as connection:
        create_table(connection)
        connection.execute("DELETE FROM master_catalog WHERE source_file_id = ?", (file_id,))
        connection.execute(
            """
            UPDATE raw_rows
            SET item_number = NULL, item_number_source = NULL, matched_master_title = NULL
            WHERE file_id = ?
            """,
            (file_id,),
        )