"""
logger.py

Small helper to get a console + file logger for every pipeline stage.
Logs are written to logs/<stage_name>.log so each DVC stage keeps its
own trail, which is handy while debugging the pipeline.
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both console and a per-stage log file."""

    logger = logging.getLogger(name)

    # avoid attaching duplicate handlers if get_logger() is called twice
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = os.path.join(LOG_DIR, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
