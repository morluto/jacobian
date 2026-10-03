"""Exact sparse growth and pre-kernel refusals for generic invariants."""

from fractions import Fraction
from math import comb, factorial

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials import _invariant_admission
from jacobian.math.polynomials._models import (
    PolynomialDiscriminantResult,
    PolynomialResultantResult,
)
from jacobian.math.polynomials.operations import (
    polynomial_discriminant,
    polynomial_resultant,
    verify_polynomial_discriminant,
    verify_polynomial_resultant,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], Fraction | int]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(
                        Fraction(c).numerator, Fraction(c).denominator
                    ),
                    exponents=e,
                )
                for e, c in sorted(terms.items(), reverse=True)
                if c
            )
        ),
    )


def _values(source: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {t.exponents: t.coefficient.as_fraction() for t in source.polynomial.terms}


def _multinomial(degree: int, scalar: int = 1) -> dict[tuple[int, ...], Fraction]:
    return {
        (a, b, degree - a - b): Fraction(
            scalar * factorial(degree),
            factorial(a) * factorial(b) * factorial(degree - a - b),
        )
        for a in range(degree + 1)
        for b in range(degree - a + 1)
    }


@pytest.mark.parametrize("degree", [43, 44, 89])
@pytest.mark.parametrize("swap", [False, True])
def test_monomial_resultant_matches_independent_multinomial(
    degree: int, swap: bool
) -> None:
    variables = ("x", "y", "z", "w")
    f = _polynomial(
        variables, {(1, 0, 0, 0): 1, (0, 1, 0, 0): 1, (0, 0, 1, 0): 1, (0, 0, 0, 1): 1}
    )
    g = _polynomial(variables, {(degree, 0, 0, 0): 1})
    result = (
        polynomial_resultant(g, f, "x") if swap else polynomial_resultant(f, g, "x")
    )
    assert result.resultant.kind == "POLYNOMIAL"
    expected = _multinomial(degree, 1 if swap else (-1) ** degree)
    assert _values(result.resultant.value) == expected
    assert result.resultant.value.variables == ("y", "z", "w")
    decoded = PolynomialResultantResult.model_validate_json(result.model_dump_json())
    assert verify_polynomial_resultant(decoded)


@pytest.mark.parametrize("degree", [44, 45, 64])
def test_binomial_discriminant_matches_independent_multinomial(degree: int) -> None:
    source = _polynomial(
        ("x", "y", "z", "w"),
        {(degree, 0, 0, 0): 1, (0, 1, 0, 0): 1, (0, 0, 1, 0): 1, (0, 0, 0, 1): 1},
    )
    result = polynomial_discriminant(source, "x")
    assert result.discriminant.kind == "POLYNOMIAL"
    assert _values(result.discriminant.value) == _multinomial(
        degree - 1, (-1) ** (degree * (degree - 1) // 2) * degree**degree
    )
    assert verify_polynomial_discriminant(
        PolynomialDiscriminantResult.model_validate_json(result.model_dump_json())
    )


def test_exact_support_boundary_and_pre_kernel_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("y", "x", "z", "unused")
    f = _polynomial(
        variables,
        {
            (0, 1, 0, 0): 1,
            (1, 0, 1, 0): -1,
            (1, 0, 0, 0): -1,
            (0, 0, 1, 0): -1,
            (0, 0, 0, 0): -1,
        },
    )
    result = polynomial_resultant(f, _polynomial(variables, {(0, 63, 0, 0): 1}), "x")
    assert result.resultant.kind == "POLYNOMIAL"
    assert _values(result.resultant.value) == {
        (a, b, 0): Fraction(comb(63, a) * comb(63, b))
        for a in range(64)
        for b in range(64)
    }
    assert len(result.resultant.value.polynomial.terms) == 4096

    def forbidden(self: _invariant_admission.InvariantPlan) -> RationalPolynomial:
        pytest.fail("backend must not run after a failed admission")

    monkeypatch.setattr(_invariant_admission.InvariantPlan, "execute", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_resultant(f, _polynomial(variables, {(0, 64, 0, 0): 1}), "x")


def test_nonconstant_leading_coefficient_discriminant_and_denominators() -> None:
    # Disc(a*x^2+b*x+c)=b^2-4*a*c; retained inactive axes are not projected away.
    source = _polynomial(
        ("y", "x", "idle"),
        {
            (1, 2, 0): Fraction(2, 3),
            (0, 1, 0): Fraction(5, 7),
            (1, 0, 0): Fraction(-3, 11),
            (0, 0, 0): Fraction(1, 2),
        },
    )
    result = polynomial_discriminant(source, "x")
    assert result.discriminant.kind == "POLYNOMIAL"
    assert result.discriminant.value.variables == ("y", "idle")
    assert _values(result.discriminant.value) == {
        (0, 0): Fraction(25, 49),
        (2, 0): Fraction(8, 11),
        (1, 0): Fraction(-4, 3),
    }


def test_generic_sylvester_resultant_has_caller_orientation() -> None:
    f = _polynomial(("x", "y"), {(2, 0): 1, (1, 1): 1, (0, 0): 1})
    g = _polynomial(("x", "y"), {(1, 1): 1, (0, 0): 2})
    result = polynomial_resultant(f, g, "x")
    assert result.resultant.kind == "POLYNOMIAL"
    assert _values(result.resultant.value) == {(0,): Fraction(4), (2,): Fraction(-1)}


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_degenerate_discriminants(degree: int) -> None:
    source = _polynomial(("x", "y"), {(degree, 1): Fraction(2, 3)})
    result = polynomial_discriminant(source, "x")
    assert result.discriminant.kind == "POLYNOMIAL"
    assert _values(result.discriminant.value) == (
        {(0,): Fraction(1)} if degree == 1 else {}
    )


def test_genuine_scalar_growth_is_refused_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coefficient = 10**256 - 1
    # The middle coefficient of [coefficient*(y+1)]**128 is too large.
    assert comb(128, 64) * coefficient**128 >= 10**32768
    left = _polynomial(("x", "y"), {(0, 1): coefficient, (0, 0): coefficient})
    right = _polynomial(("x", "y"), {(128, 0): 1})

    def forbidden(self: _invariant_admission.InvariantPlan) -> RationalPolynomial:
        pytest.fail("a scalar-growth refusal must precede the kernel")

    monkeypatch.setattr(_invariant_admission.InvariantPlan, "execute", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_resultant(left, right, "x")


def test_generic_workspace_refusal_remains_operational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _polynomial(("x", "y", "z"), {(8, 0, 0): 1, (1, 128, 0): 1, (0, 0, 128): 1})
    right = _polynomial(("x", "y", "z"), {(7, 0, 0): 1, (1, 0, 127): 1, (0, 127, 0): 1})

    def forbidden(self: _invariant_admission.InvariantPlan) -> RationalPolynomial:
        pytest.fail("a workspace refusal must precede the kernel")

    monkeypatch.setattr(_invariant_admission.InvariantPlan, "execute", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_resultant(left, right, "x")
    from jacobian.math.polynomials._models import PolynomialValue

    claim = PolynomialResultantResult(
        left=left,
        right=right,
        elimination_variable="x",
        resultant=PolynomialValue(value=_polynomial(("y", "z"), {})),
    )
    with pytest.raises(OperationResourceAdmissionError):
        verify_polynomial_resultant(claim)


@pytest.mark.parametrize("variables", [("x", "y"), ("y", "x", "z")])
def test_zero_and_constant_resultant_preserve_ring(variables: tuple[str, ...]) -> None:
    zero = _polynomial(variables, {})
    constant = _polynomial(variables, {(0,) * len(variables): Fraction(2, 3)})
    for left, right, expected in [
        (zero, constant, {}),
        (constant, zero, {}),
        (constant, constant, {(0,) * (len(variables) - 1): Fraction(1)}),
    ]:
        result = polynomial_resultant(left, right, "x")
        assert result.resultant.kind == "POLYNOMIAL"
        assert result.resultant.value.variables == tuple(
            v for v in variables if v != "x"
        )
        assert _values(result.resultant.value) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "negative",
        "boolean",
        "missing_exponents",
        "missing_coefficient",
        "zero_denominator",
        "noncanonical",
        "zero_coefficient",
        "duplicate",
        "axis_shape",
    ],
)
def test_native_source_shape_rejected_before_presolving(mutation: str) -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    source = _polynomial(("x", "y"), {(1, 0): 1})
    term = source.polynomial.terms[0]
    if mutation == "negative":
        term = term.model_copy(update={"exponents": (-1, 0)})
    elif mutation == "boolean":
        term = term.model_copy(update={"exponents": (True, 0)})
    elif mutation == "axis_shape":
        term = term.model_copy(update={"exponents": (1,)})
    elif mutation == "missing_exponents":
        term = RationalPolynomialTerm.model_construct(coefficient=term.coefficient)
    elif mutation == "missing_coefficient":
        term = RationalPolynomialTerm.model_construct(exponents=(1, 0))
    elif mutation == "zero_denominator":
        term = term.model_copy(
            update={"coefficient": CanonicalRational.model_construct(num=1, den=0)}
        )
    elif mutation == "noncanonical":
        term = term.model_copy(
            update={"coefficient": CanonicalRational.model_construct(num=2, den=2)}
        )
    elif mutation == "zero_coefficient":
        term = term.model_copy(
            update={"coefficient": CanonicalRational.from_integer_ratio(0, 1)}
        )
    source = source.model_copy(
        update={
            "polynomial": source.polynomial.model_copy(
                update={"terms": (term, term) if mutation == "duplicate" else (term,)}
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        polynomial_discriminant(source, "x")
    assert exc_info.value.errors()[0]["type"] == "polynomial.invariant_source"
    with pytest.raises(OperationDomainValidationError) as exc_info:
        polynomial_resultant(source, _polynomial(("x", "y"), {}), "x")
    assert exc_info.value.errors()[0]["type"] == "polynomial.invariant_source"
