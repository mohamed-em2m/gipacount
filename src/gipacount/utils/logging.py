"""Logging setup.  Library code never calls ``print``."""

from __future__ import annotations

import logging
import sys
from typing import Any

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger once per process.

    Args:
        level: Root log level (name or numeric).
        log_file: Optional path that receives the same records.
        force: Re-configure even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    # Third-party noise control.
    for noisy in ("PIL", "matplotlib", "urllib3", "filelock", "timm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str, **kwargs: Any) -> logging.Logger:
    """Return a module logger, configuring the root logger on first use."""
    configure_logging()
    return logging.getLogger(name)
