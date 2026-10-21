from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAME = "gmr.catalog"
LOG_PATH = Path("catalog.log")
CURRENT_PATH: Path | None = None

NOISY_PATHS = ("/status", "/favicon.ico")


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        path = str(args[2])
        if not any(p in path for p in NOISY_PATHS):
            return True
        status = int(args[4]) if len(args) >= 5 and str(args[4]).isdigit() else 200
        return status >= 400


def configure(path: str | Path | None = None) -> logging.Logger:
    global CURRENT_PATH
    logger = logging.getLogger(LOGGER_NAME)
    target = Path(path or CURRENT_PATH or LOG_PATH).resolve()
    file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    if file_handlers and Path(file_handlers[0].baseFilename).resolve() == target:
        CURRENT_PATH = target
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").addFilter(AccessLogFilter())
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file = RotatingFileHandler(target, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file)
    logger.propagate = False
    CURRENT_PATH = target
    return logger


def event(_stage: str, _name: str, level: int = logging.INFO, **fields: Any) -> None:
    logger = configure()
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.log(level, "stage=%s event=%s%s", _stage, _name, f" {values}" if values else "")
