#!/usr/bin/env python3
"""Compatibility entry point for running the project from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcriber.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
