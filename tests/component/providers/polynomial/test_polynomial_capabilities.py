from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest
from sympy import Poly, symbols
from tests.component.providers.polynomial.polynomial_capabilities_support import (
    jacobian_counterexample_map as _jacobian_counterexample_map,
)
from tests.component.providers.polynomial.polynomial_capabilities_support import (
    point as _point,
)
from tests.component.providers.polynomial.polynomial_capabilities_support import (
    poly_payload as _poly_payload,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import Conclusion, Verification


def test_jacobian_canonically_omits_zero_partial_derivatives(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
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
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
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
            input={
                "certificate_uri": result.output["certificate_uri"],
                "checker_id": result.output["checker_id"],
            },
        )
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_polynomial_jacobian_and_collision_reproduce_public_counterexample(
    authorized_polynomial_services,
) -> None:
    runtime = authorized_polynomial_services
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
    assert jacobian.output["checker_id"] == runtime.polynomial.jacobian_checker_id
    assert "conclusion" not in jacobian.output

    verified_jacobian = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
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
    assert collision.output["checker_id"] == runtime.polynomial.collision_checker_id
    assert "conclusion" not in collision.output

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="witness.verify",
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


def test_polynomial_map_evaluation_is_exact_and_materialized(
    polynomial_services,
) -> None:
    polynomial_map = _jacobian_counterexample_map()

    result = polynomial_services.core.capabilities.invoke(
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
    polynomial_services,
    monkeypatch: pytest.MonkeyPatch,
    capability_id: str,
    payload: dict[str, Any],
    diagnostic_code: str,
) -> None:
    artifact_put_calls = 0
    original_put = polynomial_services.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(polynomial_services.core.artifacts, "put", recording_put)

    result = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == diagnostic_code
    assert artifact_put_calls == 0
