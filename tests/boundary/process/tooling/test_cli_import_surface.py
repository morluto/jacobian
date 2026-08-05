from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("surface", ["help", "version"])
def test_cheap_cli_surfaces_do_not_import_runtime_or_math_backends(
    surface: str,
) -> None:
    probe = """
import json
import sys

if sys.argv[1] == "help":
    from jacobian.cli import app
    app(args=["--help"], prog_name="jacobian", standalone_mode=False)
else:
    import jacobian
    assert jacobian.__version__

forbidden = (
    "jacobian.adapters.mcp.server",
    "jacobian.domains",
    "jacobian.lean_frontend.service",
    "sympy",
    "cvc5",
    "flint",
    "z3",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print("JACOBIAN_IMPORT_SURFACE=" + json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, surface],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("JACOBIAN_IMPORT_SURFACE=")
    )

    assert json.loads(marker.partition("=")[2]) == []
