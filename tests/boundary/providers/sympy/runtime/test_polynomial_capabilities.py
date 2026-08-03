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
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope, WitnessRole
from jacobian.contracts.results import Conclusion, InputStatus, Verification


def test_jacobian_canonically_omits_zero_partial_derivatives(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    x, y = symbols("x y")
    polynomial_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x", "y"],
        "coordinates": [
            _poly_payload(Poly(x + y**2, x, y, domain="QQ")),
            _poly_payload(Poly(y, x, y, domain="QQ")),
        ],
    }

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.compute_jacobian",
            input={"map": polynomial_map},
        )
    )

    assert result.execution.status.value == "COMPLETED"
    assert result.output["matrix"][1][0] == {"terms": []}
    assert result.output["determinant"] == {
        "terms": [
            {
                "coefficient": {"num": "1", "den": "1"},
                "exponents": [0, 0],
            }
        ]
    }


def test_jacobian_represents_derived_exponents_above_the_source_limit(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    x, y = symbols("x y")
    polynomial_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x", "y"],
        "coordinates": [
            _poly_payload(Poly(x**32, x, y, domain="QQ")),
            _poly_payload(Poly(x**32 * y, x, y, domain="QQ")),
        ],
    }

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.compute_jacobian",
            input={"map": polynomial_map},
        )
    )

    assert result.execution.status.value == "COMPLETED"
    assert result.output["determinant"] == {
        "terms": [
            {
                "coefficient": {"num": "32", "den": "1"},
                "exponents": [63, 0],
            }
        ]
    }
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "certificate_uri": result.output["certificate_uri"],
                "checker_id": result.output["checker_id"],
            },
        )
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_polynomial_jacobian_and_collision_reproduce_public_counterexample(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    polynomial_map = _jacobian_counterexample_map()

    jacobian = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.compute_jacobian",
            input={"map": polynomial_map},
        )
    )

    assert jacobian.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert jacobian.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert jacobian.output["determinant"] == {
        "terms": [
            {
                "coefficient": {"num": "-2", "den": "1"},
                "exponents": [0, 0, 0],
            }
        ]
    }
    assert jacobian.output["backend"] == "sympy"
    assert jacobian.output["backend_version"]
    assert jacobian.output["certificate_uri"] in jacobian.artifact_uris
    assert (
        jacobian.output["checker_id"]
        == runtime.portfolio.polynomial.jacobian_checker_id
    )
    assert "conclusion" not in jacobian.output

    verified_jacobian = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "certificate_uri": jacobian.output["certificate_uri"],
                "checker_id": jacobian.output["checker_id"],
            },
        )
    )

    assert verified_jacobian.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified_jacobian.output["conclusion"] == Conclusion.TRUE.value
    assert verified_jacobian.output["verification_record_uri"]

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

    assert collision.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert (
        collision.output["first_evaluation_uri"]
        == first_evaluation.output["evaluation_uri"]
    )
    assert (
        collision.output["second_evaluation_uri"]
        == second_evaluation.output["evaluation_uri"]
    )
    assert collision.output["candidate_uri"] == first_evaluation.output["map_uri"]
    assert collision.output["candidate_collision"] is True
    assert (
        collision.output["first_image"]
        == collision.output["second_image"]
        == [
            {"num": "-1", "den": "4"},
            {"num": "0", "den": "1"},
            {"num": "0", "den": "1"},
        ]
    )
    assert collision.output["witness_uri"] in collision.artifact_uris
    assert (
        collision.output["checker_id"]
        == runtime.portfolio.polynomial.collision_checker_id
    )
    assert "conclusion" not in collision.output

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="witness.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "claim_uri": collision.output["claim_uri"],
                "candidate_uri": collision.output["candidate_uri"],
                "witness_uri": collision.output["witness_uri"],
                "checker_id": collision.output["checker_id"],
            },
        )
    )

    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.FALSE.value
    assert verified.output["assurance"]["verification"] == Verification.VERIFIED.value
    assert verified.output["verification_record_uri"]


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
            mode=CapabilityMode.VERIFY,
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
            mode=CapabilityMode.VERIFY,
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
    fresh_complete_runtime,
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
    first_evaluation = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    second_evaluation = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": square_map, "point": _point(1)},
        )
    )

    result = fresh_complete_runtime.core.capabilities.invoke(
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
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }
    first_evaluation = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    map_uri = first_evaluation.output["map_uri"]
    incompatible_evaluation = fresh_complete_runtime.core.artifacts.put(
        schema_uri=fresh_complete_runtime.portfolio.polynomial.evaluation_schema_uri,
        semantics_uri=fresh_complete_runtime.portfolio.polynomial.semantics_uri,
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
    original_put = fresh_complete_runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(fresh_complete_runtime.core.artifacts, "put", recording_put)

    result = fresh_complete_runtime.core.capabilities.invoke(
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


def test_polynomial_map_evaluation_is_exact_and_materialized(
    fresh_complete_runtime,
) -> None:
    polynomial_map = _jacobian_counterexample_map()

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.evaluate",
            input={
                "map": polynomial_map,
                "point": _point(-1, Fraction(3, 2), Fraction(13, 2)),
            },
        )
    )

    assert result.output["image"] == [
        {"num": "-1", "den": "4"},
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
    ]
    assert result.output["map_uri"] in result.artifact_uris
    assert result.output["evaluation_uri"] in result.artifact_uris
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE


@pytest.mark.parametrize(
    ("capability_id", "payload", "diagnostic_code"),
    [
        (
            "polynomial.map.evaluate",
            {
                "map": {
                    "map_schema_version": "1",
                    "domain": "QQ",
                    "variables": ["x"],
                    "coordinates": [
                        {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                }
                            ]
                        }
                    ],
                },
                "point": [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                ],
            },
            "INVALID_POLYNOMIAL_EVALUATION_REQUEST",
        ),
        (
            "polynomial.map.compute_jacobian",
            {
                "map": {
                    "map_schema_version": "1",
                    "domain": "QQ",
                    "variables": ["x", "y"],
                    "coordinates": [{"terms": []}],
                }
            },
            "INVALID_POLYNOMIAL_JACOBIAN_REQUEST",
        ),
        (
            "polynomial.map.collision_witness",
            {
                "map": {
                    "map_schema_version": "1",
                    "domain": "QQ",
                    "variables": ["x"],
                    "coordinates": [
                        {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                }
                            ]
                        }
                    ],
                },
                "first_point": [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                ],
                "second_point": [{"num": "0", "den": "1"}],
            },
            "INVALID_REQUEST",
        ),
        (
            "polynomial.map.collision_witness",
            {
                "first_evaluation_uri": "artifact://sha256/" + "a" * 64,
                "second_evaluation_uri": "artifact://sha256/" + "a" * 64,
            },
            "INVALID_POLYNOMIAL_COLLISION_REQUEST",
        ),
    ],
)
def test_complete_request_validation_precedes_artifact_writes(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
    capability_id: str,
    payload: dict[str, Any],
    diagnostic_code: str,
) -> None:
    artifact_put_calls = 0
    original_put = fresh_complete_runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(fresh_complete_runtime.core.artifacts, "put", recording_put)

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == diagnostic_code
    assert artifact_put_calls == 0
