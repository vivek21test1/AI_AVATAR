"""
Centralised logging configuration.
Call configure_logging() once at app startup (inside create_app).
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when called multiple times (e.g. --reload)
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers = [handler]

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
