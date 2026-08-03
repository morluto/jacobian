from __future__ import annotations

import copy
from typing import Any

from tests.component.checkers.exact_domain_checker_support import (
    _MATRIX_CASES,
    _qq,
)
from tests.support.rationals import rational_payload as _q
from tests.unit.contracts.artifacts import canonical_digest as _digest

from jacobian_checkers.exact_domain_operations import (
    check_matrix_nullspace,
    check_matrix_product,
)


def _matrix_product_checker_request() -> dict[str, Any]:
    return copy.deepcopy(
        next(
            checker_request
            for checker, checker_request in _MATRIX_CASES
            if checker is check_matrix_product
        )
    )


def test_matrix_nullspace_checker_rejects_wrong_rank() -> None:
    checker_request = copy.deepcopy(
        next(
            checker_request
            for checker, checker_request in _MATRIX_CASES
            if checker is check_matrix_nullspace
        )
    )
    checker_request["candidate"]["payload"]["rank"] = 2
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_matrix_nullspace(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_matrix_product_checker_rejects_wrong_entry() -> None:
    checker_request = _matrix_product_checker_request()
    checker_request["candidate"]["payload"]["product"]["entries"][0][0] = _q(2)
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_matrix_product(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_matrix_product_checker_rejects_wrong_shape_binding() -> None:
    checker_request = _matrix_product_checker_request()
    checker_request["candidate"]["payload"]["inner_dimension"] = 2
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_matrix_product(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_matrix_product_checker_rejects_oversized_source() -> None:
    checker_request = _matrix_product_checker_request()
    checker_request["claim"]["payload"]["left"] = _qq([[1]] * 33)
    checker_request["claim"]["payload_digest"] = _digest(
        checker_request["claim"]["payload"]
    )

    decision = check_matrix_product(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
