"""Logging configuration utilities."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logger(
    logger_name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger with file and stderr handlers.

    Args:
        logger_name: Logger name
        log_file: Path to log file (optional; if None, only stderr is used)
        level: Logging level (default INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Stderr handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(sh)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=5
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)

    return logger
