from __future__ import annotations

import os
from typing import Any, Callable

import logger

try:
    from google import genai
except ImportError:
    genai = None


class ProgressReporter:
    def __init__(self, file_id: str, stage: str, total: int, manager: Any):
        self.file_id = file_id
        self.stage = stage
        self.total = total
        self.manager = manager
        self.processed = 0
        self._batch_job_id: str | None = None

    def update(self, count: int, message: str = "") -> None:
        self.processed = count
        progress = int((count / self.total) * 100) if self.total > 0 else 0
        progress_counter = f"{count}/{self.total}"
        self.manager.set(
            self.file_id, "RUNNING", self.stage, progress, 
            message or f"Processing {count}/{self.total}",
            progress_counter=progress_counter
        )

    def report_batch_job(self, batch_job_id: str) -> None:
        self._batch_job_id = batch_job_id
        self.manager.set(
            self.file_id, "RUNNING", self.stage,
            self.manager.get(self.file_id).progress_percent,
            f"Batch job {batch_job_id} submitted",
            batch_job_id=batch_job_id
        )

    def is_cancelled(self) -> bool:
        return self.manager.is_cancelled(self.file_id)

    def cancel_batch_if_needed(self, api: Any) -> None:
        if self._batch_job_id and self._batch_job_id != "direct-execution":
            try:
                api.batches.cancel(name=self._batch_job_id)
                logger.event("STAGE_CANCEL", "batch_cancelled", file_id=self.file_id, stage=self.stage, batch_id=self._batch_job_id)
            except Exception as e:
                logger.event("STAGE_CANCEL", "batch_cancel_failed", file_id=self.file_id, stage=self.stage, error=str(e))


def check_batch_cancelled(api: Any, batch_id: str, manager: Any, file_id: str, stage: str) -> bool:
    if manager.is_cancelled(file_id):
        if batch_id and batch_id != "direct-execution":
            try:
                api.batches.cancel(name=batch_id)
                logger.event("BATCH_CANCEL", "cancelled", file_id=file_id, stage=stage, batch_id=batch_id)
            except Exception as e:
                logger.event("BATCH_CANCEL", "cancel_failed", file_id=file_id, stage=stage, error=str(e))
        return True
    return False


def get_gemini_client() -> Any:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required")
    if genai is None:
        raise RuntimeError("google-genai is required")
    return genai.Client(api_key=key)
