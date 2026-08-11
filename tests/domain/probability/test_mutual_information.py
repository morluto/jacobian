from __future__ import annotations

from copy import deepcopy

import pytest
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.domains.probability.operations import (
    _require_bounded_mutual_information_product,
)

_PX = {
    "row_labels": ["0", "1", "2"],
    "column_labels": ["0", "1", "2"],
    "probabilities": [
        [_q(0), _q(1, 4), _q(1, 4)],
        [_q(1, 4), _q(0), _q(0)],
        [_q(1, 4), _q(0), _q(0)],
    ],
    "log_base": 2,
}

_PZ = {
    "row_labels": ["0", "1", "2"],
    "column_labels": ["0", "1", "2"],
    "probabilities": [
        [_q(1, 6), _q(1, 6), _q(1, 12)],
        [_q(1, 6), _q(1, 6), _q(1, 12)],
        [_q(1, 12), _q(1, 12), _q(0)],
    ],
    "log_base": 2,
}


def test_qutrit_tables_produce_exact_log_product_certificates(
    probability_services,
) -> None:
    px = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=_PX,
        )
    ).output["result"]
    pz = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=_PZ,
        )
    ).output["result"]

    assert px["row_marginals"] == [_q(1, 2), _q(1, 4), _q(1, 4)]
    assert {entry["likelihood_ratio"]["num"] for entry in px["positive_support"]} == {
        "2"
    }
    assert px["log_product_certificate"] == {
        "scale": "4",
        "product": _q(16),
        "identity": "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT",
    }
    assert px["exact_value"] == _q(1)
    assert pz["row_marginals"] == [_q(5, 12), _q(5, 12), _q(1, 6)]
    assert pz["log_product_certificate"] == {
        "scale": "12",
        "product": _q(3456**4, 3125**4),
        "identity": "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT",
    }
    assert pz["exact_value"] is None
    assert pz["sign"] == "POSITIVE"


def test_zero_cells_are_omitted_without_division(
    probability_services,
) -> None:
    result = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input={
                "row_labels": ["0", "1"],
                "column_labels": ["0", "1"],
                "probabilities": [[_q(1), _q(0)], [_q(0), _q(0)]],
                "log_base": 2,
            },
        )
    ).output["result"]

    assert len(result["positive_support"]) == 1
    assert result["log_product_certificate"]["product"] == _q(1)
    assert result["sign"] == "ZERO"


def test_certificate_cost_is_rejected_before_exponentiation() -> None:
    class _Rational:
        def __init__(self, numerator: int, denominator: int) -> None:
            self._numerator = numerator
            self._denominator = denominator

        def numer(self) -> int:
            return self._numerator

        def denom(self) -> int:
            return self._denominator

    probability = _Rational(1, 1)
    ratio = _Rational(2, 1)

    with pytest.raises(ValueError, match="scale exceeds the replay bound"):
        _require_bounded_mutual_information_product(1 << 1_024, [(probability, ratio)])

    with pytest.raises(ValueError, match="product exceeds the output-cost bound"):
        _require_bounded_mutual_information_product(32_768, [(probability, ratio)])


def test_independent_checker_rejects_tampered_ratio(
    probability_services,
) -> None:
    computed = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.compute",
            input=_PX,
        )
    )
    candidate = deepcopy(computed.output["result"])
    candidate["positive_support"][0]["likelihood_ratio"] = _q(3)

    rejected = probability_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.joint.mutual_information.verify",
            input={"input": _PX, "candidate": candidate},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
