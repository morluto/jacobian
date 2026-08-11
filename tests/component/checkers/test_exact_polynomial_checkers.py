from __future__ import annotations

import copy
from typing import Any

from tests.component.checkers.exact_domain_checker_support import (
    _POLY_CASES,
    _poly,
    _rational_poly,
    _request,
)
from tests.support.artifacts import canonical_digest as _digest
from tests.support.rationals import rational_payload as _q

from jacobian_checkers.exact_domain_operations import (
    check_polynomial_factorization,
    check_polynomial_square_free,
)


def test_square_free_checker_normalizes_flint_factors_to_monic_contract() -> None:
    checker_request = _request(
        "polynomial.compute.square_free_decomposition",
        "polynomial.square-free.flint-replay",
        {"polynomial": _poly(2, 10, 16, 8)},
        {
            "coefficient": _q(8),
            "factors": [
                {"factor": _poly(1, 1), "multiplicity": 1},
                {
                    "factor": _rational_poly((1, 2), (1, 1)),
                    "multiplicity": 2,
                },
            ],
            "reconstructed": _poly(2, 10, 16, 8),
            "normalization": "MONIC_FACTORS",
        },
    )

    assert check_polynomial_square_free(checker_request)["accepted"] is True


def test_factorization_checker_accepts_content_and_monic_irreducibles() -> None:
    checker_request = _request(
        "polynomial.factor.compute",
        "polynomial.factorization.flint-replay",
        {"polynomial": _rational_poly((-3, 2), (0, 1), (-3, 2))},
        {
            "coefficient": _q(-3, 2),
            "factors": [
                {"factor": _poly(1, 0, 1), "multiplicity": 1},
            ],
            "reconstructed": _rational_poly((-3, 2), (0, 1), (-3, 2)),
            "normalization": "CONTENT_AND_MONIC_IRREDUCIBLES",
            "irreducibility_assurance": "UNVERIFIED",
            "product_reconstruction": "EXACT",
        },
    )

    assert check_polynomial_factorization(checker_request)["accepted"] is True


def test_polynomial_checker_rejects_variable_renaming() -> None:
    checker, checker_request = _POLY_CASES[0]
    mutated = copy.deepcopy(checker_request)
    mutated["candidate"]["payload"]["gcd"]["variables"] = ["y"]
    mutated["candidate"]["payload_digest"] = _digest(mutated["candidate"]["payload"])

    decision = checker(mutated)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_polynomial_checker_accepts_consistent_nondefault_variable_name() -> None:
    checker, checker_request = _POLY_CASES[0]
    renamed = copy.deepcopy(checker_request)
    payloads: tuple[dict[str, Any], ...] = (
        renamed["claim"]["payload"]["left"],
        renamed["claim"]["payload"]["right"],
        renamed["candidate"]["payload"]["gcd"],
        renamed["candidate"]["payload"]["bezout"]["left_multiplier"],
        renamed["candidate"]["payload"]["bezout"]["right_multiplier"],
    )
    for polynomial in payloads:
        polynomial["variables"] = ["t"]
    renamed["claim"]["payload_digest"] = _digest(renamed["claim"]["payload"])
    renamed["candidate"]["payload_digest"] = _digest(renamed["candidate"]["payload"])

    assert checker(renamed)["accepted"] is True
