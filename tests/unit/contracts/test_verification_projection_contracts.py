from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.finite_coverage import FiniteCoverageVerifyOutput
from jacobian.contracts.nullstellensatz import NullstellensatzVerificationOutput
from jacobian.contracts.polynomials import (
    PolynomialCollisionVerifyOutput,
    PolynomialIdentityOutput,
    PolynomialKellerConditionVerifyOutput,
    PolynomialMapInverseCollisionVerifyOutput,
    RationalFunctionIdentityOutput,
)


def _uri(fill: str) -> str:
    return "artifact://sha256/" + fill * 64


def _checker(fill: str = "a") -> str:
    return "checker://sha256/" + fill * 64


@pytest.mark.parametrize(
    "model",
    [PolynomialIdentityOutput, RationalFunctionIdentityOutput],
)
def test_decisive_identity_projection_requires_a_record(model: type) -> None:
    with pytest.raises(ValidationError, match="decisive identity"):
        model.model_validate(
            {
                "identical": True,
                "conclusion": "TRUE",
                "left_uri": _uri("1"),
                "right_uri": _uri("2"),
                "claim_uri": _uri("3"),
                "certificate_uri": _uri("4"),
                "checker_id": _checker(),
            }
        )


def test_polynomial_collision_projection_requires_accepted_recorded_replay() -> None:
    with pytest.raises(ValidationError, match="verified collision requires"):
        PolynomialCollisionVerifyOutput(
            collision_verified=True,
            conclusion="FALSE",
            verification_input={"status": "ACCEPTED"},
            map_uri=_uri("1"),
            claim_uri=_uri("2"),
            witness_uri=_uri("3"),
            checker_id=_checker(),
            first_point=(),
            second_point=(),
            claimed_image=(),
        )


def test_other_polynomial_projections_cannot_be_decisive_without_records() -> None:
    with pytest.raises(ValidationError, match="Keller-condition"):
        PolynomialKellerConditionVerifyOutput(
            keller_condition_verified=True,
            conclusion="TRUE",
            map_uri=_uri("1"),
            jacobian_uri=_uri("2"),
            claim_uri=_uri("3"),
            certificate_uri=_uri("4"),
            determinant={"terms": []},
            checker_id=_checker(),
        )
    with pytest.raises(ValidationError, match="non-invertibility"):
        PolynomialMapInverseCollisionVerifyOutput(
            noninvertibility_verified=True,
            conclusion="TRUE",
            verification_input={"status": "ACCEPTED"},
            map_uri=_uri("1"),
            claim_uri=_uri("2"),
            witness_uri=_uri("3"),
            checker_id=_checker(),
            first_point=(),
            second_point=(),
            claimed_image=(),
        )


def test_nullstellensatz_and_finite_coverage_bind_records_to_true_results() -> None:
    with pytest.raises(ValidationError, match="checker-backed record"):
        NullstellensatzVerificationOutput(
            system_uri=_uri("1"),
            certificate_bundle_uri=_uri("2"),
            evidence_uri=_uri("3"),
            checker_id=_checker(),
            conclusion="TRUE",
            checked_chart_count=12,
        )

    with pytest.raises(ValidationError, match="unknown coverage"):
        FiniteCoverageVerifyOutput(
            coverage_status="INVALID",
            conclusion="UNKNOWN",
            canonicalizer_id="finite.integer.decimal@1",
            canonicalizer_uri=_uri("1"),
            scope_uri=_uri("2"),
            archive_uri=_uri("3"),
            page_uris=(),
            claim_uri=_uri("4"),
            certificate_uri=_uri("5"),
            verification_record_uri=_uri("6"),
            diagnostics={},
            scope_keys_digest="sha256:" + "7" * 64,
            archive_digest="sha256:" + "8" * 64,
            checker_id=_checker(),
            detail="invalid coverage",
        )
