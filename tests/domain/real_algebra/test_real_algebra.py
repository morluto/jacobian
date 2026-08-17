from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.real_algebra import (
    PolynomialTerm,
    RootCountRequest,
    SturmChainRequest,
    UnivariatePolynomial,
)
from jacobian.domains.real_algebra.operations import (
    compute_root_count,
    compute_sturm_chain,
)

R = CanonicalRational


def _poly(*terms: tuple[str, str, int]) -> UnivariatePolynomial:
    return UnivariatePolynomial(
        terms=tuple(
            PolynomialTerm(coefficient=R(num=num, den=den), exponent=exp)
            for num, den, exp in terms
        )
    )


def test_sturm_chain_cubic() -> None:
    """x^3 - 2x^2 + x - 3 has a Sturm chain of length 4."""
    poly = _poly(("1", "1", 3), ("-2", "1", 2), ("1", "1", 1), ("-3", "1", 0))
    result = compute_sturm_chain(SturmChainRequest(polynomial=poly))
    assert len(result.chain) == 4
    assert result.degree == 3
    assert result.method == "SYMPY_STURM"


def test_sturm_chain_quadratic() -> None:
    """x^2 - 2 has a Sturm chain of length 3."""
    poly = _poly(("1", "1", 2), ("-2", "1", 0))
    result = compute_sturm_chain(SturmChainRequest(polynomial=poly))
    assert len(result.chain) == 3
    assert result.degree == 2


def test_root_count_cubic_has_one_real_root() -> None:
    """x^3 - 2x^2 + x - 3 has exactly 1 real root in [-10, 10]."""
    poly = _poly(("1", "1", 3), ("-2", "1", 2), ("1", "1", 1), ("-3", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="-10", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result.root_count == 1


def test_root_count_x_squared_minus_2() -> None:
    """x^2 - 2 has 2 real roots in [-10, 10] but 0 in [0, 10]... wait sqrt(2) ~1.41.
    Actually x^2-2 has roots at +-sqrt(2). So 2 roots in [-10,10], 1 in [0,10]."""
    poly = _poly(("1", "1", 2), ("-2", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="-10", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result.root_count == 2

    result_pos = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="2", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result_pos.root_count == 0


def test_root_count_no_real_roots() -> None:
    """x^2 + 1 has no real roots."""
    poly = _poly(("1", "1", 2), ("1", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="-10", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result.root_count == 0


def test_root_count_linear() -> None:
    """x - 5 has one root at x=5."""
    poly = _poly(("1", "1", 1), ("-5", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="0", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result.root_count == 1


def test_root_count_quartic_two_roots() -> None:
    """x^4 - 5x^2 + 4 = (x-1)(x+1)(x-2)(x+2) has 4 roots in [-10, 10]."""
    poly = _poly(("1", "1", 4), ("-5", "1", 2), ("4", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="-10", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result.root_count == 4


def test_root_count_empty_interval() -> None:
    """An empty interval has 0 roots."""
    poly = _poly(("1", "1", 2), ("-2", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="0", den="1"),
            upper=R(num="0", den="1"),
        )
    )
    assert result.root_count == 0


def test_contract_rejects_lower_gt_upper() -> None:
    with pytest.raises(ValidationError, match="lower bound"):
        RootCountRequest(
            polynomial=_poly(("1", "1", 1)),
            lower=R(num="10", den="1"),
            upper=R(num="0", den="1"),
        )


def test_contract_rejects_duplicate_exponents() -> None:
    with pytest.raises(ValidationError, match="unique"):
        UnivariatePolynomial(
            terms=(
                PolynomialTerm(coefficient=R(num="1", den="1"), exponent=2),
                PolynomialTerm(coefficient=R(num="1", den="1"), exponent=2),
            )
        )


def test_contract_rejects_zero_coefficient() -> None:
    with pytest.raises(ValidationError, match="zero"):
        UnivariatePolynomial(
            terms=(PolynomialTerm(coefficient=R(num="0", den="1"), exponent=2),)
        )


def test_contract_rejects_oversized_coefficient_digits() -> None:
    """A coefficient at the 32,768-digit wire limit is rejected by the
    operation-specific 256-digit polynomial budget."""
    big_num = "9" * 257
    with pytest.raises(ValidationError, match="256-digit bound"):
        UnivariatePolynomial(
            terms=(
                PolynomialTerm(coefficient=R(num=big_num, den="1"), exponent=2),
                PolynomialTerm(coefficient=R(num="-2", den="1"), exponent=0),
            )
        )


def test_sturm_rejects_constant_polynomial() -> None:
    """A degree-0 polynomial is rejected before execution."""
    with pytest.raises(ValidationError, match="non-constant"):
        SturmChainRequest(
            polynomial=_poly(("5", "1", 0)),
        )
