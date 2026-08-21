#!/usr/bin/env python3
"""Freeze one already-working external separator runtime into a lock."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli.freeze_separator_runtime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
