from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from tests.component.checkers.exact_domain_checker_support import _request
from tests.support.artifacts import canonical_digest
from tests.support.rationals import rational_payload as q

from jacobian_checkers.exact_arithmetic import (
    check_rational_difference,
    check_rational_product,
    check_rational_quotient,
    check_rational_sum,
)

_Case = tuple[
    Callable[[object], dict[str, Any]],
    str,
    str,
    dict[str, str],
]

_CASES: tuple[_Case, ...] = (
    (
        check_rational_sum,
        "rational.compute.sum",
        "rational.sum.flint-replay",
        q(5, 6),
    ),
    (
        check_rational_difference,
        "rational.compute.difference",
        "rational.difference.flint-replay",
        q(1, 6),
    ),
    (
        check_rational_product,
        "rational.compute.product",
        "rational.product.flint-replay",
        q(1, 6),
    ),
    (
        check_rational_quotient,
        "rational.compute.quotient",
        "rational.quotient.flint-replay",
        q(3, 2),
    ),
)


def _checker_request(
    operation_id: str,
    witness_format: str,
    result: dict[str, str],
) -> dict[str, Any]:
    return _request(
        operation_id,
        witness_format,
        {"left": q(1, 2), "right": q(1, 3)},
        {"value": result},
    )


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "result"), _CASES
)
def test_rational_arithmetic_checker_accepts_independent_exact_replay(
    checker: Callable[[object], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    result: dict[str, str],
) -> None:
    decision = checker(_checker_request(operation_id, witness_format, result))

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "result"), _CASES
)
def test_rational_arithmetic_checker_rejects_forged_freshly_bound_result(
    checker: Callable[[object], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    result: dict[str, str],
) -> None:
    request = _checker_request(operation_id, witness_format, result)
    request["candidate"]["payload"]["value"] = q(7, 9)
    request["candidate"]["payload_digest"] = canonical_digest(
        request["candidate"]["payload"]
    )

    decision = checker(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_rational_arithmetic_checker_rejects_cross_operation_witness() -> None:
    request = _checker_request(
        "rational.compute.sum",
        "rational.sum.flint-replay",
        q(5, 6),
    )
    request["witness"]["payload"]["payload"]["operation_id"] = (
        "rational.compute.product"
    )
    request["witness"]["payload_digest"] = canonical_digest(
        request["witness"]["payload"]
    )

    decision = check_rational_sum(copy.deepcopy(request))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
