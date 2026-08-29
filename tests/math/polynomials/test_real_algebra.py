from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._models import (
    PolynomialTerm,
    RootCountRequest,
    SturmChainRequest,
    UnivariatePolynomial,
)
from jacobian.math.polynomials.real_algebra._tools import (
    TOOLS,
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


def _coefficient_map(poly: UnivariatePolynomial) -> dict[int, Fraction]:
    return {term.exponent: term.coefficient.as_fraction() for term in poly.terms}


def test_sturm_metadata_matches_the_integer_euclidean_envelope() -> None:
    declarations = {tool.operation_id: tool for tool in TOOLS}
    sturm = declarations["polynomial.sturm_chain.compute"]
    root_count = declarations["polynomial.root_count.compute"]

    for description in (sturm.description, root_count.description):
        assert "integer coefficients" in description
        assert "degree at most 32" in description
        assert "16 decimal digits" in description
        assert "subresultant" not in description


def test_sturm_chain_cubic_known_answer() -> None:
    """x^3 - 2x^2 + x - 3 has SymPy's documented four-term Sturm chain."""
    poly = _poly(("1", "1", 3), ("-2", "1", 2), ("1", "1", 1), ("-3", "1", 0))
    result = compute_sturm_chain(SturmChainRequest(polynomial=poly))

    assert result.degree == 3
    assert [_coefficient_map(p) for p in result.chain] == [
        {3: Fraction(1), 2: Fraction(-2), 1: Fraction(1), 0: Fraction(-3)},
        {2: Fraction(3), 1: Fraction(-4), 0: Fraction(1)},
        {1: Fraction(2, 9), 0: Fraction(25, 9)},
        {0: Fraction(-2079, 4)},
    ]


def test_sturm_chain_quadratic() -> None:
    """x^2 - 2 has a three-term Sturm chain."""
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
    assert result.source_polynomial == poly
    assert result.root_count == 1


def test_root_count_x_squared_minus_2() -> None:
    """x^2 - 2 has 2 roots in [-10, 10] and 0 in [2, 10]."""
    poly = _poly(("1", "1", 2), ("-2", "1", 0))

    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="-10", den="1"),
            upper=R(num="10", den="1"),
        )
    )
    assert result.source_polynomial == poly
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


def test_root_count_quartic_four_roots() -> None:
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


def test_root_count_rational_endpoints() -> None:
    """Root counting supports rational interval endpoints exactly."""
    poly = _poly(("1", "1", 2), ("-2", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="0", den="1"),
            upper=R(num="3", den="2"),
        )
    )
    assert result.root_count == 1  # sqrt(2) in [0, 1.5], -sqrt(2) excluded


def test_root_count_repeated_roots_are_distinct() -> None:
    """(x-1)^2 counts its single distinct real root once."""
    poly = _poly(("1", "1", 2), ("-2", "1", 1), ("1", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="0", den="1"),
            upper=R(num="2", den="1"),
        )
    )
    assert result.root_count == 1


def test_root_count_counts_lower_endpoint_root() -> None:
    """Review fix: a root exactly at the lower endpoint belongs to [lower, upper]."""
    poly = _poly(("1", "1", 1))

    full = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="0", den="1"),
            upper=R(num="1", den="1"),
        )
    )
    assert full.root_count == 1

    singleton = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="0", den="1"),
            upper=R(num="0", den="1"),
        )
    )
    assert singleton.root_count == 1


def test_root_count_counts_both_endpoint_roots() -> None:
    """x^2 - 1 has roots at both endpoints of [-1, 1]."""
    poly = _poly(("1", "1", 2), ("-1", "1", 0))
    result = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num="-1", den="1"),
            upper=R(num="1", den="1"),
        )
    )
    assert result.root_count == 2


def test_root_count_empty_interval() -> None:
    """A non-root singleton interval has 0 roots."""
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
    request = RootCountRequest(
        polynomial=_poly(("1", "1", 1)),
        lower=R(num="10", den="1"),
        upper=R(num="0", den="1"),
    )
    with pytest.raises(OperationDomainValidationError, match="lower bound"):
        compute_root_count(request)


def test_contract_rejects_duplicate_exponents() -> None:
    with pytest.raises(ValidationError):
        UnivariatePolynomial(
            terms=(
                PolynomialTerm(coefficient=R(num="1", den="1"), exponent=2),
                PolynomialTerm(coefficient=R(num="1", den="1"), exponent=2),
            )
        )


def test_contract_rejects_zero_coefficient() -> None:
    with pytest.raises(ValidationError):
        UnivariatePolynomial(
            terms=(PolynomialTerm(coefficient=R(num="0", den="1"), exponent=2),)
        )


def test_contract_rejects_oversized_coefficient_digits() -> None:
    """Review fix: a 17-digit coefficient exceeds the 16-digit input budget."""
    big_num = "9" * 17
    poly = _poly((big_num, "1", 2), ("-2", "1", 0))
    request = RootCountRequest(
        polynomial=poly,
        lower=R(num="-10", den="1"),
        upper=R(num="10", den="1"),
    )
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficient"):
        compute_root_count(request)


def test_contract_rejects_non_integer_coefficients() -> None:
    """Review fix: rational coefficients would blow up SymPy's plain QQ chain."""
    polynomial = _poly(("1", "2", 2), ("-2", "1", 0))
    root_request = RootCountRequest(
        polynomial=polynomial,
        lower=R(num="-10", den="1"),
        upper=R(num="10", den="1"),
    )
    with pytest.raises(OperationDomainValidationError, match="must be integers"):
        compute_root_count(root_request)
    with pytest.raises(OperationDomainValidationError, match="must be integers"):
        compute_sturm_chain(SturmChainRequest(polynomial=polynomial))


def test_sturm_rejects_constant_polynomial() -> None:
    """Review fix: a degree-0 polynomial is rejected before execution."""
    request = SturmChainRequest(polynomial=_poly(("5", "1", 0)))
    with pytest.raises(OperationDomainValidationError, match="non-constant"):
        compute_sturm_chain(request)


def test_sturm_chain_boundary_sixteen_digit_coefficient() -> None:
    """Review fix: the largest accepted coefficient stays representable.

    For x^2 + N*x + 1 with N = 10^16 - 1, SymPy's chain constant is
    (N^2 - 4)/4, which has about 32 digits -- far below the 32,768-digit wire
    bound that the unbounded request would have blown through.
    """
    n = 10**16 - 1
    poly = _poly(("1", "1", 2), (str(n), "1", 1), ("1", "1", 0))
    result = compute_sturm_chain(SturmChainRequest(polynomial=poly))

    assert result.degree == 2
    assert _coefficient_map(result.chain[-1]) == {0: Fraction(n * n - 4, 4)}

    count = compute_root_count(
        RootCountRequest(
            polynomial=poly,
            lower=R(num=str(-(n + 1)), den="1"),
            upper=R(num=str(n + 1), den="1"),
        )
    )
    assert count.root_count == 2
