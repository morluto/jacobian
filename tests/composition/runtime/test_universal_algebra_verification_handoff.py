"""Composition-owned universal-algebra verification handoff workflows."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import Conclusion

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "AUTHORITY"


def _variable(name: str) -> dict[str, object]:
    return {"kind": "VARIABLE", "variable": name, "left": None, "right": None}


def _product(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    return {"kind": "PRODUCT", "variable": None, "left": left, "right": right}


def _left_projection_problem() -> dict[str, object]:
    x = _variable("x")
    y = _variable("y")
    z = _variable("z")
    return {
        "problem_schema_version": "1",
        "structure": {
            "structure_schema_version": "1",
            "operation": "binary",
            "order": 2,
            "table": [[0, 0], [1, 1]],
        },
        "laws": [
            {
                "law_id": "associative",
                "variables": ["x", "y", "z"],
                "left": _product(_product(x, y), z),
                "right": _product(x, _product(y, z)),
            },
            {
                "law_id": "commutative",
                "variables": ["x", "y"],
                "left": _product(x, y),
                "right": _product(y, x),
            },
        ],
    }


def test_evaluate_laws_handoff_composes_with_certificate_verify(
    authorized_complete_runtime,
) -> None:
    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={"problem": _left_projection_problem()},
        )
    )
    handoff = result.output["verification_handoff"]
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=handoff["capability_id"],
            input=handoff["payload"],
        )
    )

    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.TRUE.value
    assert verified.output["verification_record_uri"]


def test_countermodel_search_composes_with_independent_law_replay(
    authorized_complete_runtime,
) -> None:
    laws = _left_projection_problem()["laws"]

    search = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.search.countermodel",
            input={
                "order": 2,
                "source_laws": [laws[0]],
                "target_law": laws[1],
            },
        )
    )

    assert search.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert search.output["status"] == "WITNESS_FOUND"
    assert search.output["verification"] == "UNVERIFIED"
    assert search.output["target_record"]["holds"] is False
    assert all(record["holds"] for record in search.output["source_records"])

    evaluation = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={
                "problem": {
                    "problem_schema_version": "1",
                    "structure": search.output["structure"],
                    "laws": laws,
                }
            },
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            input={
                "certificate_uri": evaluation.output["certificate_uri"],
                "checker_id": evaluation.output["checker_id"],
            },
        )
    )

    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.TRUE.value
