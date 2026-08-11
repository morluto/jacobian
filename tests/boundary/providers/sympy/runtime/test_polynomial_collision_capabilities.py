"""Polynomial-map collision and witness boundary tests."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest
from sympy import Poly, symbols
from tests.boundary.providers.sympy.runtime.polynomial_capabilities_support import (
    jacobian_counterexample_map as _jacobian_counterexample_map,
)
from tests.boundary.providers.sympy.runtime.polynomial_capabilities_support import (
    point as _point,
)
from tests.boundary.providers.sympy.runtime.polynomial_capabilities_support import (
    poly_payload as _poly_payload,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope, WitnessRole
from jacobian.contracts.results import Conclusion, InputStatus, Verification


def test_collision_checker_rejects_a_forged_image(authorized_complete_runtime) -> None:
    runtime = authorized_complete_runtime
    polynomial_map = _jacobian_counterexample_map()
    first_evaluation = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={
                "map": polynomial_map,
                "point": _point(0, 0, Fraction(-1, 4)),
            },
        )
    )
    second_evaluation = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={
                "map": polynomial_map,
                "point": _point(1, Fraction(-3, 2), Fraction(13, 2)),
            },
        )
    )
    collision = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": second_evaluation.output["evaluation_uri"],
            },
        )
    )
    claim_uri = collision.output["claim_uri"]
    candidate_uri = collision.output["candidate_uri"]
    witness_uri = collision.output["witness_uri"]
    assert witness_uri is not None
    witness_artifact = runtime.core.store.get(witness_uri)
    original = WitnessEnvelope.model_validate(witness_artifact.payload)
    forged_payload = deepcopy(original.payload)
    forged_payload["image"][0] = {"num": "0", "den": "1"}
    forged = WitnessEnvelope(
        witness_format=original.witness_format,
        format_version=original.format_version,
        role=WitnessRole.REFUTES_CLAIM,
        bindings=EvidenceBindings.model_validate(
            original.bindings.model_dump(mode="json")
        ),
        payload=forged_payload,
    )
    forged_artifact = runtime.core.store.put(
        schema_uri=runtime.portfolio.polynomial.witness_schema_uri,
        semantics_uri=runtime.portfolio.polynomial.semantics_uri,
        payload=forged.model_dump(mode="json"),
        parents=(claim_uri, candidate_uri),
        summary="forged collision witness",
    )

    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="witness.verify",
            input={
                "claim_uri": claim_uri,
                "candidate_uri": candidate_uri,
                "witness_uri": forged_artifact.artifact_uri,
                "checker_id": runtime.portfolio.polynomial.collision_checker_id,
            },
        )
    )

    assert rejected.output["input"]["status"] == InputStatus.REJECTED.value
    assert rejected.output["conclusion"] == Conclusion.UNKNOWN.value
    assert rejected.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert rejected.output["verification_record_uri"] is None


def test_collision_comparison_does_not_promote_forged_evaluations(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }
    first_evaluation = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(0)},
        )
    )
    map_uri = first_evaluation.output["map_uri"]
    forged_evaluation = runtime.core.artifacts.put(
        schema_uri=runtime.portfolio.polynomial.evaluation_schema_uri,
        semantics_uri=runtime.portfolio.polynomial.semantics_uri,
        payload={
            "evaluation_schema_version": "1",
            "map_uri": map_uri,
            "point": {"values": _point(1)},
            "image": _point(0),
            "backend": "sympy",
            "backend_version": "forged",
        },
        parents=(map_uri,),
    )

    candidate = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": forged_evaluation.artifact_uri,
            },
        )
    )

    assert candidate.output["candidate_collision"] is True
    assert candidate.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert candidate.output["verification"] == Verification.UNVERIFIED.value

    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="witness.verify",
            input={
                "claim_uri": candidate.output["claim_uri"],
                "candidate_uri": candidate.output["candidate_uri"],
                "witness_uri": candidate.output["witness_uri"],
                "checker_id": candidate.output["checker_id"],
            },
        )
    )

    assert rejected.output["input"]["status"] == InputStatus.REJECTED.value
    assert rejected.output["conclusion"] == Conclusion.UNKNOWN.value
    assert rejected.assurance.level is CapabilityAssuranceLevel.HEURISTIC


def test_noncollision_is_computed_evidence_without_witness_or_conclusion(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }

    first_evaluation = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(0)},
        )
    )
    second_evaluation = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": second_evaluation.output["evaluation_uri"],
            },
        )
    )

    assert result.output["candidate_collision"] is False
    assert result.output["witness_uri"] is None
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert "conclusion" not in result.output


def test_collision_rejects_evaluations_from_different_maps(
    attached_complete_runtime,
) -> None:
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }
    square_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x**2, x, domain="QQ"))],
    }
    first_evaluation = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    second_evaluation = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": square_map, "point": _point(1)},
        )
    )

    result = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": second_evaluation.output["evaluation_uri"],
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "POLYNOMIAL_EVALUATION_MAP_MISMATCH"


def test_collision_validates_evaluation_dimensions_before_artifact_writes(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }
    first_evaluation = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    map_uri = first_evaluation.output["map_uri"]
    incompatible_evaluation = attached_complete_runtime.core.artifacts.put(
        schema_uri=attached_complete_runtime.portfolio.polynomial.evaluation_schema_uri,
        semantics_uri=attached_complete_runtime.portfolio.polynomial.semantics_uri,
        payload={
            "evaluation_schema_version": "1",
            "map_uri": map_uri,
            "point": {"values": _point(1, 1)},
            "image": _point(1, 1),
            "backend": "sympy",
            "backend_version": "test",
        },
        parents=(map_uri,),
    )
    artifact_put_calls = 0
    original_put = attached_complete_runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(attached_complete_runtime.core.artifacts, "put", recording_put)

    result = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": incompatible_evaluation.artifact_uri,
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "POLYNOMIAL_EVALUATION_DIMENSION_MISMATCH"
    assert artifact_put_calls == 0
