"""
Centralized structured logger.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger(level: str) -> None:
    """Configure the root logger once at startup."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())


@lru_cache(maxsize=None)
def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger, creating it once and caching it."""
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level.upper())
    return logger


def setup_logging(level: str = "info") -> None:
    """Call once from main.py to initialise the logging subsystem."""
    _configure_root_logger(level)
    logger = get_logger("ai_service")
    logger.info("Logging subsystem initialised at level=%s", level.upper())
