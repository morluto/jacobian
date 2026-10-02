"""Bounded sparse division establishes exact reconstruction and normal form."""

from fractions import Fraction
from math import comb
from random import Random

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._division_bounds import _Ledger
from jacobian.math.polynomials.multivariate import _division_bounds as bounds
from jacobian.math.polynomials.multivariate import operations
from jacobian.math.polynomials.multivariate._division import (
    MonomialOrder,
    MultivariateDivisionResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], int | Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponent,
                )
                for exponent, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _terms(value: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in value.polynomial.terms
    }


def _normal_form(result: MultivariateDivisionResult) -> None:
    """Check the defining relation using independent sparse Fraction convolution."""
    reconstructed = _terms(result.remainder)
    for a, ac in _terms(result.quotient).items():
        for b, bc in _terms(result.right).items():
            exponent = tuple(x + y for x, y in zip(a, b, strict=True))
            reconstructed[exponent] = reconstructed.get(exponent, Fraction(0)) + ac * bc
    assert {e: c for e, c in reconstructed.items() if c} == _terms(result.left)

    def key(exponent: tuple[int, ...]) -> tuple[int, ...]:
        if result.monomial_order == "lex":
            return exponent
        if result.monomial_order == "grlex":
            return (sum(exponent), *exponent)
        return (sum(exponent), *(-x for x in reversed(exponent)))

    leading = max(_terms(result.right), key=key)
    assert all(
        any(e < b for e, b in zip(exponent, leading, strict=True))
        for exponent in _terms(result.remainder)
    )
    for output in (result.quotient, result.remainder):
        assert output.variables == result.left.variables
        assert list(_terms(output)) == sorted(_terms(output), reverse=True)


def _geometric(
    variables: tuple[str, ...], degree: int
) -> tuple[
    RationalPolynomial, RationalPolynomial, RationalPolynomial, RationalPolynomial
]:
    def exponent(**powers: int) -> tuple[int, ...]:
        return tuple(powers.get(variable, 0) for variable in variables)

    left = _poly(variables, {exponent(x=degree): 1})
    right = _poly(
        variables,
        {exponent(x=1): 1, exponent(y=1): -1, exponent(z=1): -1, exponent(w=1): -1},
    )
    quotient: dict[tuple[int, ...], int | Fraction] = {}
    remainder: dict[tuple[int, ...], int | Fraction] = {}
    for k in range(degree + 1):
        for y in range(k + 1):
            for z in range(k - y + 1):
                powers = {"y": y, "z": z, "w": k - y - z}
                coefficient = comb(k, y) * comb(k - y, z)
                if k == degree:
                    remainder[exponent(**powers)] = coefficient
                else:
                    quotient[exponent(x=degree - 1 - k, **powers)] = coefficient
    return left, right, _poly(variables, quotient), _poly(variables, remainder)


@pytest.mark.parametrize("degree", [17, 18, 28])
def test_geometric_division_admits_large_canonical_quotients(degree: int) -> None:
    left, right, quotient, remainder = _geometric(("x", "y", "z", "w"), degree)
    result = operations.multivariate_division(left, right)
    assert (result.quotient, result.remainder) == (quotient, remainder)
    assert len(result.quotient.polynomial.terms) == comb(degree + 2, 3)
    _normal_form(result)
    decoded = MultivariateDivisionResult.model_validate_json(result.model_dump_json())
    assert operations.verify_multivariate_division(decoded)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_full_permuted_axes_and_orders_survive_large_result(
    order: MonomialOrder,
) -> None:
    axes = ("u", "x", "z", "v", "w", "r", "y", "t")
    left, right, quotient, remainder = _geometric(axes, 18)
    result = operations.multivariate_division(left, right, order)
    assert (result.quotient, result.remainder) == (quotient, remainder)
    _normal_form(result)


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_independent_small_rational_division_relations(order: MonomialOrder) -> None:
    rng = Random(4364)
    for _ in range(18):
        left = _poly(
            ("x", "y", "z"),
            {
                tuple(rng.randrange(4) for _ in range(3)): Fraction(
                    rng.randrange(-7, 8), rng.randrange(1, 11)
                )
                for _ in range(6)
            },
        )
        right = _poly(
            ("x", "y", "z"),
            {
                tuple(rng.randrange(3) for _ in range(3)): Fraction(
                    rng.randrange(1, 8), rng.randrange(1, 11)
                )
                for _ in range(4)
            },
        )
        _normal_form(operations.multivariate_division(left, right, order))


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
def test_monomial_orders_with_different_leading_terms(order: MonomialOrder) -> None:
    left = _poly(("x", "y"), {(2, 0): 1})
    right = _poly(("x", "y"), {(1, 0): 1, (0, 2): -1})
    result = operations.multivariate_division(left, right, order)
    if order == "lex":
        assert result.quotient == _poly(left.variables, {(1, 0): 1, (0, 2): 1})
        assert result.remainder == _poly(left.variables, {(0, 4): 1})
    else:
        assert not result.quotient.polynomial.terms
        assert result.remainder == left
    _normal_form(result)


def test_merging_paths_accumulate_maximum_depth_before_expansion() -> None:
    # xy -> xz or y^2 yields merging paths of different length from two sources.
    left = _poly(
        ("x", "y", "z"), {(4, 4, 0): Fraction(2, 3), (3, 5, 0): Fraction(5, 7)}
    )
    right = _poly(
        left.variables,
        {
            (1, 1, 0): Fraction(3, 5),
            (1, 0, 1): Fraction(-7, 11),
            (0, 2, 0): Fraction(-13, 17),
        },
    )
    result = operations.multivariate_division(left, right)
    _normal_form(result)
    support = bounds._support(left, right, "lex", _Ledger())
    assert support.depth == 10
    assert len(result.quotient.polynomial.terms) <= support.quotient
    assert len(result.remainder.polynomial.terms) <= support.remainder


@pytest.mark.parametrize("kind", ["zero", "self", "constant", "unit", "monomial"])
def test_degenerate_divisions_remain_exact(kind: str) -> None:
    left = _poly(
        ("x", "y", "z"), {(64, 0, 0): Fraction(7, 11), (0, 1, 0): -1, (0, 0, 1): 1}
    )
    right = _poly(left.variables, {(1, 0, 0): 1, (0, 1, 0): -1})
    if kind == "zero":
        left = _poly(left.variables, {})
    elif kind == "self":
        right = left
    elif kind == "constant":
        right = _poly(left.variables, {(0, 0, 0): Fraction(2, 7)})
    elif kind == "unit":
        right = _poly(left.variables, {(0, 0, 0): 1})
    else:
        right = _poly(left.variables, {(1, 0, 0): Fraction(2, 7)})
    result = operations.multivariate_division(left, right)
    _normal_form(result)
    assert operations.verify_multivariate_division(
        MultivariateDivisionResult.model_validate_json(result.model_dump_json())
    )


@pytest.mark.parametrize("failure", ["support", "coefficient_height", "exponent"])
def test_true_output_overflows_are_rejected_before_backend(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sympy.polys.rings import PolyElement

    if failure == "support":
        # Its positive geometric quotient has C(31,3)=4,495 terms.
        left = _poly(("x", "y", "z", "w"), {(29, 0, 0, 0): 1})
        right = _poly(
            left.variables,
            {(1, 0, 0, 0): 1, (0, 1, 0, 0): -1, (0, 0, 1, 0): -1, (0, 0, 0, 1): -1},
        )
    elif failure == "coefficient_height":
        # A positive rewrite path reaches x*z^99 after148 reductions. The
        # coefficient is at least (10^255)^148, beyond32,768 decimal digits.
        left = _poly(("x", "y", "z"), {(50, 50, 0): 1})
        right = _poly(
            left.variables, {(1, 1, 0): 1, (1, 0, 1): -(10**255), (0, 2, 0): -(10**255)}
        )
    else:
        # Eight y^64 rewrites followed by568 x*z^64 rewrites yield an
        # uncancelled remainder monomial x^56*z^36352.
        left = _poly(("x", "y", "z"), {(64, 64, 0): 1})
        right = _poly(left.variables, {(1, 1, 0): 1, (1, 0, 64): -1, (0, 64, 0): -1})

    def unexpected_backend(*args: object, **kwargs: object) -> None:
        pytest.fail("growth must be admitted before PolyElement.div")

    monkeypatch.setattr(PolyElement, "div", unexpected_backend)
    with pytest.raises(OperationResourceAdmissionError) as error:
        operations.multivariate_division(left, right)
    assert error.value.errors()[0]["type"] == f"polynomial.division.{failure}"


@pytest.mark.parametrize(
    "budget",
    [
        "_MAX_MONOMIAL_WORK",
        "MAX_DIVISION_ALLOCATION_BITS",
        "MAX_DIVISION_RESULT_DIGITS",
    ],
)
def test_accounting_guards_precede_backend(
    budget: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sympy.polys.rings import PolyElement

    left, right, _, _ = _geometric(("x", "y", "z", "w"), 2)
    true_claim = operations.multivariate_division(left, right)
    monkeypatch.setattr(bounds, budget, 1)

    def unexpected_backend(*args: object, **kwargs: object) -> None:
        pytest.fail("work/storage/output envelope must precede the backend")

    monkeypatch.setattr(PolyElement, "div", unexpected_backend)
    with pytest.raises(OperationResourceAdmissionError):
        operations.multivariate_division(left, right)
    with pytest.raises(OperationResourceAdmissionError):
        operations.verify_multivariate_division(true_claim)


def test_zero_divisor_retains_typed_domain_diagnostic() -> None:
    left = _poly(("x", "y"), {(1, 0): 1})
    with pytest.raises(
        OperationDomainValidationError, match="divisor polynomial must be nonzero"
    ):
        operations.multivariate_division(left, _poly(left.variables, {}))


def test_exact_4096_term_quotient_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from sympy.polys.rings import PolyElement

    variables = ("x", "y", "z", "w", "t", "u", "s")
    source: dict[tuple[int, ...], int | Fraction] = {
        (28, 0, 0, 0, 0, 0, 0): 1,
        (5, 0, 0, 0, 1, 0, 0): 1,
        (1, 0, 0, 0, 0, 1, 0): 1,
    }
    right = _poly(
        variables,
        {
            (1, 0, 0, 0, 0, 0, 0): 1,
            (0, 1, 0, 0, 0, 0, 0): -1,
            (0, 0, 1, 0, 0, 0, 0): -1,
            (0, 0, 0, 1, 0, 0, 0): -1,
        },
    )
    result = operations.multivariate_division(_poly(variables, source), right)
    # The three geometric quotients have disjoint t/u supports:4060+35+1.
    assert len(result.quotient.polynomial.terms) == 4096
    assert len(result.remainder.polynomial.terms) == 459
    _normal_form(result)
    decoded = MultivariateDivisionResult.model_validate_json(result.model_dump_json())
    assert operations.verify_multivariate_division(decoded)

    # The new s-tagged x term adds exactly one quotient term without cancellation.
    source[(1, 0, 0, 0, 0, 0, 1)] = 1
    left = _poly(variables, source)

    def unexpected_backend(*args: object, **kwargs: object) -> None:
        pytest.fail("4,097-term quotient must be refused before the backend")

    monkeypatch.setattr(PolyElement, "div", unexpected_backend)
    with pytest.raises(OperationResourceAdmissionError) as error:
        operations.multivariate_division(left, right)
    assert error.value.errors()[0]["type"] == "polynomial.division.support"
