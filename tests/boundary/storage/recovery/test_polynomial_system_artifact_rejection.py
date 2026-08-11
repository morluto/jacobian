from __future__ import annotations

import sqlite3
from typing import Any

from tests.support.polynomials import univariate_term as _term

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus


def _input(value: int) -> dict[str, Any]:
    return {
        "system": {
            "system_schema_version": "1",
            "domain": "QQ",
            "variables": ["x"],
            "equations": [{"terms": [_term(1, 2), _term(-4, 0)]}],
            "inequations": [{"terms": [_term(1, 1)]}],
        },
        "assignment": [{"num": str(value), "den": "1"}],
    }


def test_solution_capability_rejects_dimension_mismatch_before_artifact_writes(
    authorized_complete_runtime,
) -> None:
    connection = sqlite3.connect(authorized_complete_runtime.core.store.db_path)
    try:
        before = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    invalid = _input(2)
    invalid["assignment"].append({"num": "3", "den": "1"})

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=invalid,
        )
    )

    connection = sqlite3.connect(authorized_complete_runtime.core.store.db_path)
    try:
        after = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_POLYNOMIAL_SYSTEM_SOLUTION_REQUEST"
    assert result.diagnostics[0].stage == "request_validation"
    assert before == after
