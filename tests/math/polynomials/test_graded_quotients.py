from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.graded import operations as graded_operations
from jacobian.math.polynomials.graded._models import (
    HilbertDimensionResult,
    HilbertFunctionResult,
    HilbertMultiplicityResult,
    HilbertPolynomialRequest,
    HilbertPolynomialResult,
    HilbertSeriesRequest,
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
    variables = tuple("xyzwuvst"[: len(exponents[0])])
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


@pytest.mark.parametrize("degree", (1.5, True))
def test_native_standard_monomial_degree_rejects_non_integers_before_arithmetic(
    monkeypatch: pytest.MonkeyPatch, degree: object
) -> None:
    import jacobian.math.polynomials.graded.operations as graded_operations

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("non-integer degrees must not inspect the monomial ideal")

    monkeypatch.setattr(graded_operations, "_require_monomial_ideal", fail)
    with pytest.raises(OperationDomainValidationError, match="must be an integer"):
        standard_monomials(_ideal((2, 0)), degree)  # type: ignore[arg-type]


def test_hilbert_series_admits_reduced_degree_below_rational_function_envelope() -> (
    None
):
    ideal = _ideal((20, 0, 0, 0), (0, 20, 0, 0), (0, 0, 20, 0), (0, 0, 0, 5))
    series = hilbert_series(ideal)
    reduced_degree = max(
        (term.exponents[0] for term in series.series.numerator.terms),
        default=0,
    )
    assert reduced_degree <= 61
    assert series.denominator_exponent == 0


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


def test_hilbert_series_rejects_nine_minimal_monomials_before_groebner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.graded.operations as graded_operations

    ideal = _ideal(*((8 - index, index) for index in range(9)))

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("series generator overflow must not expand Groebner")

    monkeypatch.setattr(graded_operations, "initial_monomial_ideal", fail)
    with pytest.raises(
        OperationResourceAdmissionError, match="at most 8 minimal generators"
    ):
        hilbert_series(ideal, prefix_degree=1)


def test_hilbert_function_rejects_eight_variable_degree_eleven_before_groebner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.graded.operations as graded_operations

    variables = tuple(f"x{index}" for index in range(8))
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(1,) + (0,) * 7,
                        ),
                    )
                ),
            ),
        ),
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("function slice overflow must not expand Groebner")

    monkeypatch.setattr(graded_operations, "initial_monomial_ideal", fail)
    with pytest.raises(
        OperationResourceAdmissionError, match="standard-monomial domain"
    ):
        hilbert_function(ideal, max_degree=11)


def test_wide_unit_coefficient_remains_admitted_for_graded_series() -> None:
    from jacobian.math.polynomials.ideals._models import MAX_COEFFICIENT_DIGITS

    variables = ("x",)
    wide = 10**MAX_COEFFICIENT_DIGITS
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=wide, den=1),
                            exponents=(0,),
                        ),
                    )
                ),
            ),
        ),
    )
    series = hilbert_series(ideal, prefix_degree=2)
    assert series.prefix == (0, 0, 0)
    assert series.denominator_exponent == 0


def test_projection_helpers_are_not_public_catalog_operations() -> None:
    from jacobian.math.polynomials.graded._tools import TOOLS

    published = {tool.operation_id for tool in TOOLS}
    assert "graded_quotient.dimension.compute" not in published
    assert "graded_quotient.multiplicity.compute" not in published
    assert "graded_quotient.h_vector.compute" not in published
    assert hilbert_dimension(_ideal((2, 0))).dimension == 1
    assert hilbert_multiplicity(_ideal((2, 0))).multiplicity == 2
    assert h_vector(_ideal((2, 0))).h_vector == (1, 1)


def test_unit_ideal_hilbert_function_skips_ambient_slice_cap() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0,) * 8,
                        ),
                    )
                ),
            ),
        ),
    )
    profile = hilbert_function(ideal, max_degree=11)
    assert profile.values == (0,) * 12


def test_duplicate_source_monomials_do_not_inflate_series_cap() -> None:
    ideal = _ideal(*((2, 0) for _ in range(9)))
    series = hilbert_series(ideal, prefix_degree=3)
    assert series.prefix == (1, 2, 2, 2)


def test_hilbert_polynomial_binds_source_ring_and_m_axis() -> None:
    result = hilbert_polynomial(_ideal((2, 0)))
    other_variables = ("u", "v")
    other = hilbert_polynomial(
        RationalPolynomialIdeal(
            variables=other_variables,
            generators=(
                RationalPolynomial(
                    variables=other_variables,
                    polynomial=SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=(2, 0),
                            ),
                        )
                    ),
                ),
            ),
        )
    )
    payload = result.model_dump()
    payload["initial_ideal"] = other.initial_ideal.model_dump()
    with pytest.raises(ValidationError, match="source ring"):
        HilbertPolynomialResult.model_validate(payload)
    forged = result.model_dump()
    forged["polynomial"]["variables"] = ["t"]
    with pytest.raises(ValidationError, match="m axis"):
        HilbertPolynomialResult.model_validate(forged)


def test_hilbert_polynomial_request_omits_series_prefix() -> None:
    schema = HilbertPolynomialRequest.model_json_schema()
    assert "prefix_degree" not in schema["properties"]
    with pytest.raises(ValidationError, match="extra"):
        HilbertPolynomialRequest.model_validate(
            {"ideal": _ideal((2, 0)).model_dump(), "prefix_degree": 17}
        )
    HilbertSeriesRequest.model_validate(
        {"ideal": _ideal((2, 0)).model_dump(), "prefix_degree": 16}
    )


def test_hilbert_function_and_series_prefixes_are_nonempty_and_nonnegative() -> None:
    function = hilbert_function(_ideal((2, 0)), max_degree=2)
    payload = function.model_dump()
    payload["values"] = []
    with pytest.raises(ValidationError):
        HilbertFunctionResult.model_validate(payload)
    payload["values"] = [-1, 2, 2]
    with pytest.raises(ValidationError):
        HilbertFunctionResult.model_validate(payload)
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    series_payload = series.model_dump()
    series_payload["prefix"] = []
    with pytest.raises(ValidationError):
        HilbertSeriesResult.model_validate(series_payload)


def test_hilbert_series_rejects_noncanonical_denominator_shape() -> None:
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    payload = series.model_dump()
    degree = payload["denominator_exponent"]
    payload["series"]["denominator"]["terms"] = [
        {"coefficient": {"num": 1, "den": 1}, "exponents": [degree]}
    ]
    with pytest.raises(ValidationError, match="denominator"):
        HilbertSeriesResult.model_validate(payload)


def test_explicit_unit_generator_short_circuits_before_homogeneity() -> None:
    variables = ("x",)
    unit = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(0,),
                ),
            )
        ),
    )
    redundant = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(2,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(1,),
                ),
            )
        ),
    )
    ideal = RationalPolynomialIdeal(variables=variables, generators=(unit, redundant))
    initial = initial_monomial_ideal(ideal)
    assert initial.initial_ideal.generators[0].polynomial.terms[0].exponents == (0,)
    assert initial.groebner_basis.generators[0].polynomial.terms[0].exponents == (0,)


def test_quadratic_hypersurface_stabilizes_from_degree_zero() -> None:
    variables = ("x", "y", "z")
    hypersurface = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(2, 0, 0),
                        ),
                    )
                ),
            ),
        ),
    )
    polynomial = hilbert_polynomial(hypersurface)
    assert polynomial.dimension == 2
    assert polynomial.polynomial.polynomial.terms == (
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=2, den=1), exponents=(1,)
        ),
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=1, den=1), exponents=(0,)
        ),
    )
    assert polynomial.stabilization_degree == 0
    prefix = hilbert_function(hypersurface, max_degree=3)
    assert prefix.values == (1, 3, 5, 7)


def test_hilbert_series_denominator_exponent_is_bounded_before_expansion() -> None:
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    payload = series.model_dump()
    payload["denominator_exponent"] = 10_000_000
    with pytest.raises(ValidationError):
        HilbertSeriesResult.model_validate(payload)


def test_native_monomial_order_is_validated_before_ideal_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid monomial orders must not inspect the ideal")

    monkeypatch.setattr(graded_operations, "_require_homogeneous", fail)
    with pytest.raises(OperationDomainValidationError, match="lex, grlex, or grevlex"):
        initial_monomial_ideal(_ideal((2, 0)), "degrevlex")  # type: ignore[arg-type]
    unit = _ideal((0,))
    with pytest.raises(OperationDomainValidationError, match="lex, grlex, or grevlex"):
        initial_monomial_ideal(unit, "degrevlex")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "result_type",
    (HilbertDimensionResult, HilbertMultiplicityResult, HVectorResult),
)
def test_hilbert_projection_results_bind_the_source_ring(
    result_type: type[object],
) -> None:
    if result_type is HilbertDimensionResult:
        result = hilbert_dimension(_ideal((2, 0)))
    elif result_type is HilbertMultiplicityResult:
        result = hilbert_multiplicity(_ideal((2, 0)))
    else:
        result = h_vector(_ideal((2, 0)))
    other = initial_monomial_ideal(_ideal((2, 0, 0)))
    payload = result.model_dump()
    payload["initial_ideal"] = other.initial_ideal.model_dump()
    with pytest.raises(ValidationError, match="source ring"):
        result_type.model_validate(payload)
