from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.polynomials import (
    PolynomialMapInverseSynthesisArtifact,
    PolynomialMapInverseVerifyOutput,
)

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64
_OTHER_ARTIFACT_URI = "artifact://sha256/" + "b" * 64
_CHECKER_URI = "checker://sha256/" + "c" * 64


def _verification_output(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "inverse_verified": True,
        "conclusion": "TRUE",
        "forward_map_uri": _ARTIFACT_URI,
        "inverse_map_uri": _ARTIFACT_URI,
        "residuals_uri": _ARTIFACT_URI,
        "claim_uri": _ARTIFACT_URI,
        "certificate_uri": _OTHER_ARTIFACT_URI,
        "inverse_after_forward_checker_records": [_ARTIFACT_URI],
        "forward_after_inverse_checker_records": [_ARTIFACT_URI],
        "verification_record_uri": _ARTIFACT_URI,
        "checker_id": _CHECKER_URI,
        "source_variables": ["x"],
        "target_variables": ["u"],
    }
    payload.update(updates)
    return payload


def _identity_map(variable: str) -> dict[str, object]:
    return {
        "domain": "QQ",
        "variables": [variable],
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
    }


def _synthesis_artifact(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "FOUND",
        "forward_map": _identity_map("x"),
        "ansatz": {
            "support_mode": "FULL_TOTAL_DEGREE",
            "inverse_degree_bound": 1,
            "source_variables": ["x"],
            "target_variables": ["u"],
            "coordinate_supports": [[[1]]],
            "coefficient_symbols": [["c_0_0"]],
        },
        "coefficient_equations": [],
        "solver_provenance": {
            "solver": "sympy.solve",
            "backend_version": "test",
            "timeout_ms": 1000,
            "unknown_count": 1,
            "equation_count": 1,
            "residual_term_count": 0,
            "elapsed_ms": 1,
        },
        "candidate_inverse_map": _identity_map("u"),
        "inverse_after_forward": [{"terms": []}],
        "forward_after_inverse": [{"terms": []}],
        "verification_output": _verification_output(),
        "verification_artifact_uri": _ARTIFACT_URI,
    }
    payload.update(updates)
    return payload


def test_inverse_verification_record_exists_exactly_for_decisive_conclusions() -> None:
    for invalid in (
        _verification_output(verification_record_uri=None),
        _verification_output(
            inverse_verified=None,
            conclusion="UNKNOWN",
        ),
    ):
        with pytest.raises(ValidationError, match="verification record"):
            PolynomialMapInverseVerifyOutput.model_validate(invalid)


def test_synthesis_binds_typed_verification_output_and_artifact() -> None:
    artifact = PolynomialMapInverseSynthesisArtifact.model_validate(
        _synthesis_artifact()
    )
    assert artifact.verification_output is not None
    assert artifact.verification_output.inverse_verified is True

    with pytest.raises(ValidationError, match="must bind"):
        PolynomialMapInverseSynthesisArtifact.model_validate(
            _synthesis_artifact(verification_artifact_uri=_OTHER_ARTIFACT_URI)
        )
    with pytest.raises(ValidationError, match="must be absent"):
        PolynomialMapInverseSynthesisArtifact.model_validate(
            _synthesis_artifact(verification_failure="contradictory failure")
        )
