"""Polynomial-map collision and witness boundary tests."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest
from sympy import Poly, symbols
from tests.component.providers.polynomial.polynomial_operations_support import (
    jacobian_counterexample_map as _jacobian_counterexample_map,
)
from tests.component.providers.polynomial.polynomial_operations_support import (
    point as _point,
)
from tests.component.providers.polynomial.polynomial_operations_support import (
    poly_payload as _poly_payload,
)

from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope, WitnessRole
from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import Conclusion, InputStatus


def test_collision_checker_rejects_a_forged_image(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    polynomial_map = _jacobian_counterexample_map()
    first_evaluation = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={
                "map": polynomial_map,
                "point": _point(0, 0, Fraction(-1, 4)),
            },
        )
    )
    second_evaluation = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={
                "map": polynomial_map,
                "point": _point(1, Fraction(-3, 2), Fraction(13, 2)),
            },
        )
    )
    collision = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_witness",
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
        schema_uri=runtime.polynomial.witness_schema_uri,
        semantics_uri=runtime.polynomial.semantics_uri,
        payload=forged.model_dump(mode="json"),
        parents=(claim_uri, candidate_uri),
        summary="forged collision witness",
    )

    rejected = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_evidence.verify",
            input={
                "claim_uri": claim_uri,
                "candidate_uri": candidate_uri,
                "witness_uri": forged_artifact.artifact_uri,
            },
        )
    )

    assert rejected.output["input"]["status"] == InputStatus.REJECTED.value
    assert rejected.output["conclusion"] == Conclusion.UNKNOWN.value
    assert rejected.output["verification_record_uri"] is None


def test_collision_comparison_does_not_promote_forged_evaluations(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }
    first_evaluation = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(0)},
        )
    )
    map_uri = first_evaluation.output["map_uri"]
    forged_evaluation = runtime.core.artifacts.put(
        schema_uri=runtime.polynomial.evaluation_schema_uri,
        semantics_uri=runtime.polynomial.semantics_uri,
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

    candidate = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": forged_evaluation.artifact_uri,
            },
        )
    )

    assert candidate.output["candidate_collision"] is True

    rejected = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_evidence.verify",
            input={
                "claim_uri": candidate.output["claim_uri"],
                "candidate_uri": candidate.output["candidate_uri"],
                "witness_uri": candidate.output["witness_uri"],
            },
        )
    )

    assert rejected.output["input"]["status"] == InputStatus.REJECTED.value
    assert rejected.output["conclusion"] == Conclusion.UNKNOWN.value


def test_noncollision_is_computed_evidence_without_witness_or_conclusion(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }

    first_evaluation = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(0)},
        )
    )
    second_evaluation = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": second_evaluation.output["evaluation_uri"],
            },
        )
    )

    assert result.output["candidate_collision"] is False
    assert result.output["witness_uri"] is None
    assert "conclusion" not in result.output


def test_collision_rejects_evaluations_from_different_maps(
    polynomial_services,
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
    first_evaluation = polynomial_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    second_evaluation = polynomial_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={"map": square_map, "point": _point(1)},
        )
    )

    result = polynomial_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": second_evaluation.output["evaluation_uri"],
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "POLYNOMIAL_EVALUATION_MAP_MISMATCH"


def test_collision_validates_evaluation_dimensions_before_artifact_writes(
    polynomial_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = symbols("x")
    identity_map = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [_poly_payload(Poly(x, x, domain="QQ"))],
    }
    first_evaluation = polynomial_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.evaluate",
            input={"map": identity_map, "point": _point(1)},
        )
    )
    map_uri = first_evaluation.output["map_uri"]
    incompatible_evaluation = polynomial_services.core.artifacts.put(
        schema_uri=polynomial_services.polynomial.evaluation_schema_uri,
        semantics_uri=polynomial_services.polynomial.semantics_uri,
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
    original_put = polynomial_services.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(polynomial_services.core.artifacts, "put", recording_put)

    result = polynomial_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision_witness",
            input={
                "first_evaluation_uri": first_evaluation.output["evaluation_uri"],
                "second_evaluation_uri": incompatible_evaluation.artifact_uri,
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "POLYNOMIAL_EVALUATION_DIMENSION_MISMATCH"
    assert artifact_put_calls == 0
