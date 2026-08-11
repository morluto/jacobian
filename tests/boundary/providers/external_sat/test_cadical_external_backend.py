from __future__ import annotations

import pytest
from tests.support.provider_external_sat import cadical_runtime_available

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.providers.external_solver_runtime import (
    CADICAL_VERSION,
)

pytestmark = [
    pytest.mark.skipif(
        not cadical_runtime_available(),
        reason="the pinned CaDiCaL runtime is unavailable",
    ),
]


def test_pinned_cadical_produces_a_model_and_text_drat_proof(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    provider_runtime = runtime.portfolio.cadical_runtime
    assert provider_runtime is not None
    if provider_runtime.version != CADICAL_VERSION:
        pytest.skip(f"requires pinned CaDiCaL {CADICAL_VERSION}")
    capability_ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
    assert {"sat.model.find", "sat.unsat_proof.find"}.issubset(capability_ids)

    satisfiable = runtime.core.sat.put_cnf(
        variable_names=("x", "y"),
        clauses=((1,), (-2,)),
    )
    model = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.model.find",
            input={
                "cnf_uri": satisfiable.artifact_uri,
                "resource_budget": {"wall_seconds": 5},
            },
        )
    )
    assert model.execution.status is ExecutionStatus.COMPLETED
    assert model.output["status"] == "ASSIGNMENT_PRODUCED"
    assert model.output["conclusion"] == "UNKNOWN"

    unsatisfiable = runtime.core.sat.put_cnf(
        variable_names=("x",),
        clauses=((1,), (-1,)),
    )
    proof = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.unsat_proof.find",
            input={
                "cnf_uri": unsatisfiable.artifact_uri,
                "resource_budget": {"wall_seconds": 5},
            },
        )
    )
    assert proof.execution.status is ExecutionStatus.COMPLETED
    assert proof.output["status"] == "PROOF_PRODUCED"
    assert proof.output["conclusion"] == "UNKNOWN"
