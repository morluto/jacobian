from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.graded._models import (
    HilbertFunctionResult,
    HilbertSeriesResult,
    HVectorResult,
    StandardMonomialsResult,
)
from jacobian.math.polynomials.graded.operations import (
    h_vector,
    hilbert_dimension,
    hilbert_function,
    hilbert_multiplicity,
    hilbert_polynomial,
    hilbert_series,
    initial_monomial_ideal,
    standard_monomials,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _ideal(*exponents: tuple[int, ...]) -> RationalPolynomialIdeal:
    variables = tuple("xy"[: len(exponents[0])])
    return RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=exponent,
                        ),
                    )
                ),
            )
            for exponent in exponents
        ),
    )


def test_initial_ideal_standard_monomials_and_hilbert_prefix_compose() -> None:
    initial = initial_monomial_ideal(_ideal((2, 0)))
    assert initial.initial_ideal.generators[0].polynomial.terms[0].exponents == (2, 0)
    degree_two = standard_monomials(initial.initial_ideal, 2)
    assert degree_two.monomials == ((1, 1), (0, 2))
    profile = hilbert_function(_ideal((2, 0)), max_degree=4)
    assert profile.values == (1, 2, 2, 2, 2)


def test_standard_monomials_reject_nonhomogeneous_initial_input() -> None:
    variables = ("x", "y")
    nonhomogeneous = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(1, 0),
                        ),
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0, 0),
                        ),
                    )
                ),
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        initial_monomial_ideal(nonhomogeneous)


def test_hilbert_series_polynomial_dimension_multiplicity_and_h_vector() -> None:
    ideal = _ideal((2, 0))
    series = hilbert_series(ideal, prefix_degree=4)
    assert series.prefix == (1, 2, 2, 2, 2)
    assert series.denominator_exponent == 1
    assert series.ambient_denominator_exponent == 2
    assert series.reduced_numerator.polynomial.terms[
        0
    ].coefficient == CanonicalRational(num=-1, den=1)
    assert series.h_numerator.polynomial.terms[0].coefficient == CanonicalRational(
        num=1, den=1
    )
    assert h_vector(ideal).h_vector == (1, 1)
    assert hilbert_dimension(ideal).dimension == 1
    assert hilbert_multiplicity(ideal).multiplicity == 2
    polynomial = hilbert_polynomial(ideal).polynomial
    assert polynomial.variables == ("m",)
    assert polynomial.polynomial.terms[0].coefficient == CanonicalRational(num=2, den=1)
    assert polynomial.polynomial.terms[0].exponents == (0,)
    assert type(series).model_validate_json(series.model_dump_json()) == series


def test_zero_dimensional_macaulay_fixture_has_length_one_series() -> None:
    ideal = _ideal((1, 0), (0, 1))
    series = hilbert_series(ideal, prefix_degree=3)
    assert series.prefix == (1, 0, 0, 0)
    assert series.denominator_exponent == 0
    assert hilbert_dimension(ideal).dimension == 0
    assert hilbert_multiplicity(ideal).multiplicity == 1
    polynomial = hilbert_polynomial(ideal).polynomial
    assert polynomial.variables == ("m",)
    assert polynomial.polynomial.terms == ()


def test_odd_dimension_keeps_canonical_series_sign_separate_from_h_numerator() -> None:
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    assert series.denominator_exponent == 1
    assert series.series.numerator == series.reduced_numerator.polynomial
    assert series.reduced_numerator.polynomial.terms[
        0
    ].coefficient == CanonicalRational(num=-1, den=1)
    assert series.h_numerator.polynomial.terms[0].coefficient == CanonicalRational(
        num=1, den=1
    )
    assert h_vector(_ideal((2, 0))).h_vector == (1, 1)


def test_hilbert_invariants_are_order_invariant_even_when_initial_generators_differ() -> (
    None
):
    variables = ("x", "y")
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(2, 0),
                        ),
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0, 2),
                        ),
                    )
                ),
            ),
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(1, 1),
                        ),
                    )
                ),
            ),
        ),
    )
    assert (
        hilbert_series(ideal, "lex", prefix_degree=5).prefix
        == hilbert_series(ideal, "grevlex", prefix_degree=5).prefix
    )


def test_graded_result_axes_publish_structural_transport_bounds() -> None:
    assert (
        StandardMonomialsResult.model_json_schema()["properties"]["monomials"][
            "maxItems"
        ]
        == 20_000
    )
    assert (
        HilbertFunctionResult.model_json_schema()["properties"]["values"]["maxItems"]
        == 33
    )
    assert (
        HilbertSeriesResult.model_json_schema()["properties"]["prefix"]["maxItems"]
        == 17
    )
    assert HVectorResult.model_json_schema()["properties"]["h_vector"]["maxItems"] == 65


@pytest.mark.parametrize("degree", (-1, 33))
def test_native_standard_monomial_degree_bounds_precede_ideal_inspection(
    monkeypatch: pytest.MonkeyPatch, degree: int
) -> None:
    import jacobian.math.polynomials.graded.operations as graded_operations

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("inadmissible degrees must not inspect the monomial ideal")

    monkeypatch.setattr(graded_operations, "_require_monomial_ideal", fail)
    with pytest.raises(
        OperationResourceAdmissionError, match="degrees from 0 through 32"
    ):
        standard_monomials(_ideal((2, 0)), degree)


def test_hilbert_series_result_binds_ambient_denominator_to_source_ring() -> None:
    result = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    forged = result.model_dump()
    forged["ambient_denominator_exponent"] = 99

    with pytest.raises(ValidationError, match="source-ring dimension"):
        HilbertSeriesResult.model_validate(forged)


@pytest.mark.parametrize(
    ("operation", "keyword", "value"),
    (
        (hilbert_function, "max_degree", -1),
        (hilbert_function, "max_degree", 33),
        (hilbert_series, "prefix_degree", -1),
        (hilbert_series, "prefix_degree", 17),
    ),
)
def test_native_hilbert_prefix_bounds_precede_initial_ideal_expansion(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., object],
    keyword: str,
    value: int,
) -> None:
    import jacobian.math.polynomials.graded.operations as graded_operations

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("inadmissible prefixes must not expand an initial ideal")

    monkeypatch.setattr(graded_operations, "initial_monomial_ideal", fail)
    with pytest.raises(OperationResourceAdmissionError, match="prefixes support"):
        operation(_ideal((2, 0)), **{keyword: value})
