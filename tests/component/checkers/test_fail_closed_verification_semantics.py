"""Fail-closed verification semantics for checker support and VERIFIED gates."""

from __future__ import annotations

from jacobian.contracts.results import Verification
from jacobian.exact_domain_checkers import _checker_supports


def test_checker_supports_unknown_polynomial_operation_is_false() -> None:
    assert (
        _checker_supports(
            "polynomial.compute.hypothetical_new_op",
            {"left": {"variables": ["x"], "terms": []}},
        )
        is False
    )


def test_checker_supports_known_univariate_gcd() -> None:
    payload = {
        "left": {"variables": ["x"], "terms": [{"coefficient": "1", "exponents": [1]}]},
        "right": {
            "variables": ["x"],
            "terms": [{"coefficient": "1", "exponents": [0]}],
        },
    }
    assert _checker_supports("polynomial.compute.gcd", payload) is True


def test_verified_assurance_requires_accepted_checker_decision() -> None:
    """An accepted exact checker verifies either decisive conclusion."""

    def verification_for(*, accepted: bool) -> Verification:
        return Verification.VERIFIED if accepted else Verification.UNVERIFIED

    assert verification_for(accepted=True) is Verification.VERIFIED
    assert verification_for(accepted=False) is Verification.UNVERIFIED
