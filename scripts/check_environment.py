"""Standalone environment check: `python scripts/check_environment.py`.

Prints the same diagnostics as `echotwin doctor` without requiring an
installed package (adds src/ to sys.path). Exit 0 when core is healthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echotwin.check_env import doctor_report  # noqa: E402


def main() -> int:
    report = doctor_report()
    for line in report["lines"]:
        print(line)
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
