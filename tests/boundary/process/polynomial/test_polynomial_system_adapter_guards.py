from __future__ import annotations

import subprocess
import sys


def test_solution_adapter_rejects_missing_checker_under_optimized_python() -> None:
    """An optimized interpreter must not erase checker-authorization guards."""

    script = """
from jacobian.polynomial_system_capabilities import (
    PolynomialSystemInstallation,
    PolynomialSystemResources,
    PolynomialSystemSolutionAdapter,
)

installation = PolynomialSystemInstallation(
    semantics_uri="semantics://test",
    system_schema_uri="schema://system",
    assignment_schema_uri="schema://assignment",
    claim_schema_uri="schema://claim",
    certificate_schema_uri="schema://certificate",
    checker_id=None,
)
resources = PolynomialSystemResources(
    store=None,
    artifacts=None,
    verification=None,
    installation=installation,
)
try:
    PolynomialSystemSolutionAdapter(resources)
except RuntimeError as exc:
    if "authorized checker" not in str(exc):
        raise
else:
    raise SystemExit("missing checker did not prevent adapter construction")
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
