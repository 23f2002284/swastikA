import logging
import json
from typing import Any


def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_structure(logger: logging.Logger, event: str, **fields: Any):
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))