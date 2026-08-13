from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.math.probability import FiniteJointTable, mutual_information
from jacobian.math.probability.mutual_information import (
    MutualInformationCertificate,
    MutualInformationResult,
    MutualInformationTerm,
)

_PAYLOAD = {
    "row_labels": ["0", "1"],
    "column_labels": ["0", "1"],
    "probabilities": [
        [_q(1, 2), _q(0)],
        [_q(0), _q(1, 2)],
    ],
    "log_base": 2,
}


def _verify(probability_services, payload, candidate):
    return probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.verify",
            input={"input": payload, "candidate": candidate},
        )
    )


def test_native_mutual_information_composes_with_fractions() -> None:
    result = mutual_information(
        FiniteJointTable(
            row_labels=("0", "1"),
            column_labels=("0", "1"),
            probabilities=(
                (Fraction(1, 2), Fraction()),
                (Fraction(), Fraction(1, 2)),
            ),
            log_base=16,
        )
    )

    assert result.row_marginals == (Fraction(1, 2), Fraction(1, 2))
    assert result.column_marginals == (Fraction(1, 2), Fraction(1, 2))
    assert result.certificate.scale == 2
    assert result.certificate.product == 4
    assert result.exact_value == Fraction(1, 4)


def test_native_table_rejects_oversized_probability_components() -> None:
    oversized = Fraction(1, 10**256 + 1)

    with pytest.raises(ValueError, match="256-digit bound"):
        FiniteJointTable(
            row_labels=("rare", "common"),
            column_labels=("only",),
            probabilities=((oversized,), (1 - oversized,)),
        )


def test_native_result_rejects_support_that_does_not_reconstruct_marginals() -> None:
    with pytest.raises(ValueError, match="does not reconstruct row marginals"):
        MutualInformationResult(
            row_marginals=(Fraction(1),),
            column_marginals=(Fraction(1),),
            positive_support=(
                MutualInformationTerm(
                    row_index=0,
                    column_index=0,
                    probability=Fraction(2),
                    row_marginal=Fraction(1),
                    column_marginal=Fraction(1),
                    likelihood_ratio=Fraction(2),
                ),
            ),
            log_base=2,
            certificate=MutualInformationCertificate(scale=1, product=Fraction(4)),
            exact_value=Fraction(2),
            sign="POSITIVE",
        )


def test_finite_joint_mutual_information_is_exact_and_verified(
    probability_services,
) -> None:
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=_PAYLOAD,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    assert result["row_marginals"] == [_q(1, 2), _q(1, 2)]
    assert result["column_marginals"] == [_q(1, 2), _q(1, 2)]
    assert result["exact_value"] == _q(1)
    assert result["sign"] == "POSITIVE"
    assert result["log_product_certificate"] == {
        "scale": "2",
        "product": _q(4),
        "identity": "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT",
    }
    assert [
        (item["row_index"], item["column_index"]) for item in result["positive_support"]
    ] == [(0, 0), (1, 1)]
    assert computed.artifact_uris == ()

    verified = _verify(probability_services, _PAYLOAD, result)

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_composite_log_base_preserves_available_exact_value(
    probability_services,
) -> None:
    payload = {**_PAYLOAD, "log_base": 16}
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=payload,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    assert result["log_product_certificate"]["scale"] == "2"
    assert result["log_product_certificate"]["product"] == _q(4)
    assert result["exact_value"] == _q(1, 4)

    verified = _verify(probability_services, payload, result)
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_checker_formats_large_valid_log_product_without_decimal_limit_failure(
    probability_services,
) -> None:
    payload = {
        "row_labels": ["rare", "common"],
        "column_labels": ["rare", "common"],
        "probabilities": [
            [_q(1, 1400), _q(0)],
            [_q(0), _q(1399, 1400)],
        ],
        "log_base": 2,
    }
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=payload,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    product = result["log_product_certificate"]["product"]
    assert len(product["num"]) > 4_300
    assert len(product["den"]) > 4_300

    verified = _verify(probability_services, payload, result)
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_unit_likelihood_ratios_do_not_consume_power_cost_budget(
    probability_services,
) -> None:
    payload = {
        "row_labels": ["rare", "common"],
        "column_labels": ["only"],
        "probabilities": [[_q(1, 100_000)], [_q(99_999, 100_000)]],
        "log_base": 2,
    }
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=payload,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    assert result["sign"] == "ZERO"
    assert result["exact_value"] == _q(0)
    assert result["log_product_certificate"]["scale"] == "100000"
    assert result["log_product_certificate"]["product"] == _q(1)

    verified = _verify(probability_services, payload, result)
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_mutual_information_checker_rejects_tampered_ratio(
    probability_services,
) -> None:
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=_PAYLOAD,
        )
    )
    candidate = deepcopy(computed.output["result"])
    candidate["positive_support"][0]["likelihood_ratio"] = _q(3)

    rejected = _verify(probability_services, _PAYLOAD, candidate)

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.verification_record_uri is None
