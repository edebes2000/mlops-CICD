# src/logger.py
"""
Purpose
- Configure logging once so every module gets consistent, timestamped, leveled logs
- Log to console (developer visibility) and to file (post-run audit trail)
"""

import logging
import sys
from pathlib import Path


def configure_logging(
    *,
    log_level: str,
    log_file: Path,
) -> None:
    """
    Configure root logging once for the whole process

    Inputs
    - log_level: "DEBUG", "INFO", "WARNING", "ERROR"
    - log_file: absolute path to the log file
    """
    numeric_level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        filename=str(log_file),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[console_handler, file_handler],
        force=True,
    )