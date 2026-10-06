from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER_NAME = "gmr.catalog"
LOG_PATH = Path("catalog.log")
CURRENT_PATH: Path | None = None


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
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file = logging.FileHandler(target, encoding="utf-8")
    file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file)
    logger.propagate = False
    CURRENT_PATH = target
    return logger


def event(stage: str, name: str, **fields: Any) -> None:
    logger = configure()
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("stage=%s event=%s%s", stage, name, f" {values}" if values else "")
