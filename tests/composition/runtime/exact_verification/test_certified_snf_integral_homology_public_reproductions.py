from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRODUCTIONS = (
    PROJECT_ROOT
    / "benchmarks"
    / "reproduction_cases"
    / "certified_snf_integral_homology_public.json"
)


def _suite() -> dict[str, Any]:
    suite = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    assert suite["scored"] is False
    assert suite["held_out_evaluation"]["status"] == "READY_NOT_RUN"
    return suite


def test_public_certified_smith_cases_reach_checker_bound_results(
    authorized_complete_runtime,
) -> None:
    for case in _suite()["smith_cases"]:
        computed = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="matrix.normal_form.smith.certified.compute",
                input={"matrix": case["matrix"]},
            )
        )
        certificate = computed.output["result"]["certificate"]
        assert certificate["rank"] == case["expected_rank"]
        assert certificate["invariant_factors"] == case["expected_invariant_factors"]
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="matrix.normal_form.smith.certified.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_public_integral_homology_cases_bind_generators_and_torsion(
    authorized_complete_runtime,
) -> None:
    for case in _suite()["homology_cases"]:
        materialized = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_complex.materialize",
                input=case["presentation"],
            )
        )
        computed = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_homology.integral.compute",
                input={
                    "complex": materialized.output["result"]["complex"],
                    "convention": case["convention"],
                },
            )
        )
        groups = computed.output["result"]["groups"]
        assert [group["betti_number"] for group in groups] == (
            case["expected_free_ranks"]
        )
        assert [group["torsion_coefficients"] for group in groups] == (
            case["expected_torsion"]
        )
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_homology.integral.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
