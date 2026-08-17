"""Helpers for exercising repository-owned CI scripts as subprocesses."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_ci_script(
    script: str,
    *args: str | Path,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    """Run one checked-in CI script with the repository's test defaults."""

    return subprocess.run(
        [sys.executable, str(ROOT / ".github" / "scripts" / script), *map(str, args)],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        input=input_text,
        text=True,
        timeout=timeout,
    )
