from fractions import Fraction
from itertools import pairwise

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.tropical._models import UnivariateSplitFormRequest
from jacobian.math.polynomials.tropical._tools import (
    TOOLS,
    compute_univariate_split_form,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_SCALAR_DIGITS,
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
)


def _poly(
    convention: str,
    base: str,
    terms: tuple[tuple[int, Fraction], ...],
) -> TropicalPolynomial:
    semiring = TropicalSemiring(
        convention=convention,  # type: ignore[arg-type]
        base=base,  # type: ignore[arg-type]
    )
    return TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(exponent,),
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_fraction(coefficient),
                ),
            )
            for exponent, coefficient in terms
        ),
    )


def _evaluate(poly: TropicalPolynomial, point: Fraction) -> Fraction | None:
    if not poly.terms:
        return None
    values = tuple(
        term.coefficient.value.as_fraction() + term.exponents[0] * point  # type: ignore[union-attr]
        for term in poly.terms
    )
    return min(values) if poly.semiring.convention == "MIN_PLUS" else max(values)


def _active_slope(poly: TropicalPolynomial, point: Fraction) -> int | None:
    if not poly.terms:
        return None
    values = tuple(
        (
            term.coefficient.value.as_fraction()  # type: ignore[union-attr]
            + term.exponents[0] * point,
            term.exponents[0],
        )
        for term in poly.terms
    )
    optimum = (
        min(value for value, _ in values)
        if poly.semiring.convention == "MIN_PLUS"
        else max(value for value, _ in values)
    )
    active = tuple(slope for value, slope in values if value == optimum)
    assert len(set(active)) == 1
    return active[0]


def _exact_function_equality_probes(
    left: TropicalPolynomial, right: TropicalPolynomial
) -> tuple[tuple[Fraction, bool], ...]:
    """Partition the line at every crossing among source and result lines."""
    lines = tuple(
        (
            term.exponents[0],
            term.coefficient.value.as_fraction(),  # type: ignore[union-attr]
        )
        for poly in (left, right)
        for term in poly.terms
    )
    crossings = set()
    for index, (left_slope, left_intercept) in enumerate(lines):
        for right_slope, right_intercept in lines[index + 1 :]:
            if left_slope != right_slope:
                crossings.add(
                    (right_intercept - left_intercept) / (left_slope - right_slope)
                )
    ordered = tuple(sorted(crossings))
    if not ordered:
        return ((Fraction(0), False),)
    probes = {ordered[0] - 1, ordered[-1] + 1, *ordered}
    probes.update((lower + upper) / 2 for lower, upper in pairwise(ordered))
    return tuple((point, point in crossings) for point in sorted(probes))


@pytest.mark.parametrize(
    ("convention", "terms", "expected_coefficients"),
    [
        (
            "MIN_PLUS",
            ((0, Fraction(0)), (2, Fraction(1))),
            (Fraction(0), Fraction(1, 2), Fraction(1)),
        ),
        (
            "MAX_PLUS",
            ((0, Fraction(0)), (1, Fraction(2)), (2, Fraction(0))),
            (Fraction(0), Fraction(2), Fraction(0)),
        ),
        # The middle term is never active; the split form preserves the function.
        (
            "MIN_PLUS",
            ((0, Fraction(0)), (1, Fraction(100)), (2, Fraction(0))),
            (Fraction(0), Fraction(0), Fraction(0)),
        ),
    ],
)
def test_split_form_is_functionally_equal_and_keeps_exact_roots(
    convention: str,
    terms: tuple[tuple[int, Fraction], ...],
    expected_coefficients: tuple[Fraction, ...],
) -> None:
    source = _poly(
        convention, "ZZ" if all(c.denominator == 1 for _, c in terms) else "QQ", terms
    )
    result = compute_univariate_split_form(
        UnivariateSplitFormRequest(polynomial=source)
    )

    first_exponent = terms[0][0]
    assert tuple(term.exponents[0] for term in result.terms) == tuple(
        range(first_exponent, first_exponent + len(expected_coefficients))
    )
    assert (
        tuple(
            term.coefficient.value.as_fraction()  # type: ignore[union-attr]
            for term in result.terms
        )
        == expected_coefficients
    )
    # Pairwise line crossings partition the whole real line into intervals.
    # Compare exact values at every boundary and exact affine slopes and values
    # in each interval, independently of the root and split-form kernels.
    for point, is_boundary in _exact_function_equality_probes(source, result):
        assert _evaluate(result, point) == _evaluate(source, point)
        if not is_boundary:
            assert _active_slope(result, point) == _active_slope(source, point)


def test_integer_input_promotes_when_a_split_coefficient_is_rational() -> None:
    source = _poly("MIN_PLUS", "ZZ", ((0, Fraction(0)), (2, Fraction(1))))
    result = compute_univariate_split_form(
        UnivariateSplitFormRequest(polynomial=source)
    )
    assert result.semiring == TropicalSemiring(convention="MIN_PLUS", base="QQ")
    assert result.terms[1].coefficient.value == CanonicalRational.from_integer_ratio(
        1, 2
    )  # type: ignore[union-attr]


def test_zero_and_monomial_have_their_obvious_split_forms() -> None:
    zero = _poly("MIN_PLUS", "ZZ", ())
    monomial = _poly("MAX_PLUS", "ZZ", ((3, Fraction(-7)),))
    assert (
        compute_univariate_split_form(UnivariateSplitFormRequest(polynomial=zero))
        == zero
    )
    assert (
        compute_univariate_split_form(UnivariateSplitFormRequest(polynomial=monomial))
        == monomial
    )


def test_support_growth_is_rejected_before_root_computation(monkeypatch) -> None:
    source = _poly("MIN_PLUS", "ZZ", ((0, Fraction(0)), (512, Fraction(0))))

    def unexpected_root_computation(_poly: TropicalPolynomial):
        raise AssertionError("root computation must follow split-form admission")

    monkeypatch.setattr(
        "jacobian.math.polynomials.tropical.operations.tropical_polynomial_univariate_roots",
        unexpected_root_computation,
    )
    with pytest.raises(OperationResourceAdmissionError, match="consecutive split form"):
        compute_univariate_split_form(UnivariateSplitFormRequest(polynomial=source))


def test_split_form_cancellation_is_admitted_and_package_exports_function() -> None:
    from jacobian.math.polynomials.tropical import (
        tropical_polynomial_univariate_split_form,
    )

    assert callable(tropical_polynomial_univariate_split_form)
    denominator = 10**5_000 + 7
    source = _poly(
        "MIN_PLUS",
        "QQ",
        ((0, Fraction(1, denominator)), (1, Fraction(0))),
    )
    result = tropical_polynomial_univariate_split_form(source)
    assert _evaluate(result, Fraction(0)) == _evaluate(source, Fraction(0))
    assert (
        max(
            len(str(abs(term.coefficient.value.num)))  # type: ignore[union-attr]
            for term in result.terms
        )
        <= MAX_TROPICAL_SCALAR_DIGITS
    )


def test_split_form_skips_unused_root_profile_byte_limit() -> None:
    from jacobian.math.polynomials.tropical.operations import (
        tropical_polynomial_univariate_roots,
    )

    coefficient = Fraction(10**4_999 + 123)
    source = _poly(
        "MIN_PLUS",
        "ZZ",
        tuple((exponent, coefficient) for exponent in range(207)),
    )
    with pytest.raises(OperationResourceAdmissionError, match="result-byte"):
        tropical_polynomial_univariate_roots(source)
    result = compute_univariate_split_form(
        UnivariateSplitFormRequest(polynomial=source)
    )
    assert len(result.terms) == 207
    assert _evaluate(result, Fraction(0)) == coefficient


def test_split_form_is_published_as_one_immutable_operation() -> None:
    tools = {tool.operation_id: tool for tool in TOOLS}
    assert (
        tools["tropical.polynomial.univariate_split_form.compute"].result_type
        is TropicalPolynomial
    )
