from __future__ import annotations

import pytest
from tests.boundary.providers.external_sat.external_sat_support import (
    open_cadical_services,
)
from tests.support.provider_external_sat import cadical_runtime_available

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.providers.external_solver_runtime import (
    CADICAL_VERSION,
    cadical_provider_runtime,
)

pytestmark = [
    pytest.mark.skipif(
        not cadical_runtime_available(),
        reason="the pinned CaDiCaL runtime is unavailable",
    ),
]


def test_pinned_cadical_produces_a_model_and_text_drat_proof(
    tmp_path,
) -> None:
    provider_runtime = cadical_provider_runtime()
    if provider_runtime.version != CADICAL_VERSION:
        pytest.skip(f"requires pinned CaDiCaL {CADICAL_VERSION}")
    with open_cadical_services(tmp_path / "state") as runtime:
        operation_ids = {
            descriptor.operation_id
            for descriptor in runtime.core.operations.snapshot().operations
        }
        assert {"sat.model.find", "sat.unsat_proof.find"}.issubset(operation_ids)

        satisfiable = runtime.core.sat.put_cnf(
            variable_names=("x", "y"),
            clauses=((1,), (-2,)),
        )
        model = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="sat.model.find",
                input={
                    "cnf_uri": satisfiable.artifact_uri,
                    "resource_budget": {"wall_seconds": 5},
                },
            )
        )
        assert model.execution.status is ExecutionStatus.COMPLETED
        assert model.output["status"] == "ASSIGNMENT_PRODUCED"
        assert "conclusion" not in model.output

        unsatisfiable = runtime.core.sat.put_cnf(
            variable_names=("x",),
            clauses=((1,), (-1,)),
        )
        proof = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="sat.unsat_proof.find",
                input={
                    "cnf_uri": unsatisfiable.artifact_uri,
                    "resource_budget": {"wall_seconds": 5},
                },
            )
        )
        assert proof.execution.status is ExecutionStatus.COMPLETED
        assert proof.output["status"] == "PROOF_PRODUCED"
        assert "conclusion" not in proof.output
