from __future__ import annotations

from collections.abc import Callable
from time import monotonic

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.graded import operations as graded_operations
from jacobian.math.polynomials.graded._models import (
    MAX_GRADED_DEGREE,
    HilbertDimensionResult,
    HilbertFunctionRequest,
    HilbertFunctionResult,
    HilbertMultiplicityResult,
    HilbertPolynomialRequest,
    HilbertPolynomialResult,
    HilbertSeriesRequest,
    HilbertSeriesResult,
    HVectorResult,
    InitialMonomialIdealRequest,
    InitialMonomialIdealResult,
    StandardMonomialsRequest,
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
    MAX_POLYNOMIAL_EXPONENT,
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


def test_hilbert_function_request_defaults_max_degree_to_zero() -> None:
    request = HilbertFunctionRequest(ideal=_ideal((1, 0)))
    assert request.max_degree == 0
    assert hilbert_function(_ideal((1, 0))).values == (1,)


def test_mixed_unit_ideal_skips_hilbert_function_slice_budget() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    unit = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(0,) * 8,
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
                    exponents=(2,) + (0,) * 7,
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(1,) + (0,) * 7,
                ),
            )
        ),
    )
    result = hilbert_function(
        RationalPolynomialIdeal(variables=variables, generators=(unit, redundant)),
        max_degree=11,
    )
    assert result.values == (0,) * 12


def test_hilbert_series_rejects_reduced_numerator_beyond_exponent_bound() -> None:
    generators = tuple(
        tuple(20 if axis == index else 0 for axis in range(8)) for index in range(8)
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="rational-function envelope"
    ):
        hilbert_series(_ideal(*generators))


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
                            exponents=(2,) + (0,) * 7,
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


def test_hilbert_series_ambient_exponent_is_a_resource_rejection() -> None:
    """An LCM past the shared exponent bound is a typed resource rejection.

    ``(x^32768, y^32768)`` has representable generators, but the
    inclusion-exclusion subset LCM reaches ``t^65536`` past the shared
    polynomial exponent carrier, so the ambient numerator must be refused as a
    resource admission rather than leaking a Pydantic shape failure.
    """

    ideal = _ideal((MAX_POLYNOMIAL_EXPONENT, 0), (0, MAX_POLYNOMIAL_EXPONENT))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        hilbert_series(ideal, prefix_degree=2)
    assert exc_info.value.errors()[0]["type"] == (
        "graded_ideal.series_ambient_exponent_budget"
    )


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


def test_linear_generators_prune_standard_monomial_domain() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    generators = tuple(
        RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=1, den=1),
                        exponents=tuple(1 if axis == index else 0 for axis in range(8)),
                    ),
                )
            ),
        )
        for index in range(7)
    )
    ideal = RationalPolynomialIdeal(variables=variables, generators=generators)
    enumerated = standard_monomials(ideal, 11)
    assert enumerated.count == 1
    assert enumerated.monomials == ((0, 0, 0, 0, 0, 0, 0, 11),)
    profile = hilbert_function(ideal, max_degree=11)
    assert profile.values[-1] == 1


def test_unit_generator_with_oversized_redundant_summand_short_circuits() -> None:
    from jacobian.math.polynomials.ideals._models import MAX_INPUT_EXPONENT

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
    oversized = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(MAX_INPUT_EXPONENT + 1,),
                ),
            )
        ),
    )
    ideal = RationalPolynomialIdeal(variables=variables, generators=(unit, oversized))
    initial = initial_monomial_ideal(ideal)
    assert initial.initial_ideal.generators[0].polynomial.terms[0].exponents == (0,)


def test_initial_ideal_result_rejects_multiterm_generators() -> None:
    result = initial_monomial_ideal(_ideal((2, 0)))
    payload = result.model_dump()
    payload["initial_ideal"]["generators"][0]["polynomial"]["terms"] = [
        {"coefficient": {"num": 1, "den": 1}, "exponents": [1, 0]},
        {"coefficient": {"num": 1, "den": 1}, "exponents": [0, 1]},
    ]
    with pytest.raises(ValidationError, match="unit monomials"):
        InitialMonomialIdealResult.model_validate(payload)


def test_hilbert_results_bound_dimension_by_the_ambient_ring() -> None:
    series = hilbert_series(_ideal((1, 0), (0, 1)), prefix_degree=0)
    payload = series.model_dump()
    payload["denominator_exponent"] = 3
    payload["series"]["denominator"]["terms"] = [
        {"coefficient": {"num": 1, "den": 1}, "exponents": [3]},
        {"coefficient": {"num": -3, "den": 1}, "exponents": [2]},
        {"coefficient": {"num": 3, "den": 1}, "exponents": [1]},
        {"coefficient": {"num": -1, "den": 1}, "exponents": [0]},
    ]
    with pytest.raises(ValidationError, match="source-ring dimension"):
        HilbertSeriesResult.model_validate(payload)
    polynomial = hilbert_polynomial(_ideal((2, 0)))
    poly_payload = polynomial.model_dump()
    poly_payload["dimension"] = 3
    with pytest.raises(ValidationError, match="source-ring dimension"):
        HilbertPolynomialResult.model_validate(poly_payload)


def test_catalog_examples_state_homogeneity_and_monomial_shape() -> None:
    from jacobian.math.polynomials.graded._tools import TOOLS

    descriptions = {tool.operation_id: tool.examples[0].description for tool in TOOLS}
    assert "homogeneous" in descriptions["graded_quotient.hilbert_series.compute"]
    assert "homogeneous" in descriptions["graded_quotient.hilbert_polynomial.compute"]
    assert (
        "homogeneous" in descriptions["polynomial.ideal.initial_monomial_ideal.compute"]
    )
    assert (
        "unit monomial"
        in descriptions["monomial_ideal.standard_monomials.degree.compute"]
    )
    assert "homogeneous" in descriptions["graded_quotient.hilbert_function.compute"]


def test_standard_monomial_result_rejects_divisible_exponents() -> None:
    payload = standard_monomials(_ideal((2, 0)), 2).model_dump()
    payload["monomials"] = [(2, 0)]
    payload["count"] = 1
    with pytest.raises(
        ValidationError, match="divisible by an initial-ideal generator"
    ):
        StandardMonomialsResult.model_validate(payload)


def test_hilbert_series_admits_after_source_leadings_reduce() -> None:
    variables = ("x", "y")
    linear_pair = RationalPolynomialIdeal(
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
                            exponents=(0, 1),
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
                            exponents=(1, 0),
                        ),
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=-1, den=1),
                            exponents=(0, 1),
                        ),
                    )
                ),
            ),
        ),
    )
    result = hilbert_series(linear_pair, "lex", prefix_degree=2)
    assert result.prefix == (1, 0, 0)


def test_nonzero_monomial_coefficients_count_toward_series_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("x", "y")
    generators = tuple(
        RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=2, den=1),
                        exponents=(8 - index, index),
                    ),
                )
            ),
        )
        for index in range(9)
    )
    ideal = RationalPolynomialIdeal(variables=variables, generators=generators)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("scalar monomials must preflight before Groebner")

    monkeypatch.setattr(graded_operations, "initial_monomial_ideal", fail)
    with pytest.raises(
        OperationResourceAdmissionError, match="at most 8 minimal generators"
    ):
        hilbert_series(ideal, prefix_degree=1)


def test_linear_generators_prune_degree_thirty_two_traversal() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    generators = tuple(
        RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=1, den=1),
                        exponents=tuple(1 if axis == index else 0 for axis in range(8)),
                    ),
                )
            ),
        )
        for index in range(7)
    )
    ideal = RationalPolynomialIdeal(variables=variables, generators=generators)
    enumerated = standard_monomials(ideal, 32)
    assert enumerated.count == 1
    assert enumerated.monomials == ((0, 0, 0, 0, 0, 0, 0, 32),)


def test_graded_binds_one_deadline_before_groebner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian._execution import current_request_execution, request_execution
    from jacobian.math.polynomials.ideals._models import IdealComputationBudget

    observed: dict[str, float | None] = {}
    real_groebner = graded_operations.groebner_basis

    def wrapped(*args: object, **kwargs: object) -> object:
        execution = current_request_execution()
        observed["deadline"] = None if execution is None else execution.deadline
        return real_groebner(*args, **kwargs)

    monkeypatch.setattr(graded_operations, "groebner_basis", wrapped)
    started = monotonic()
    with request_execution(started, outer_deadline=started + 30):
        initial_monomial_ideal(
            _nonmonomial_quadratic(),
            resource_budget=IdealComputationBudget(wall_seconds=5),
        )
    assert observed["deadline"] is not None
    assert observed["deadline"] <= started + 5 + 1


def _monomial_ideal(
    variables: tuple[str, ...], exponents: tuple[tuple[int, ...], ...]
) -> RationalPolynomialIdeal:
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


def _nonmonomial_quadratic() -> RationalPolynomialIdeal:
    """Homogeneous x^2 + xy: genuinely non-monomial, so nested calls run."""
    variables = ("x", "y")
    return RationalPolynomialIdeal(
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
                            exponents=(1, 1),
                        ),
                    )
                ),
            ),
        ),
    )


def test_hilbert_function_admits_mixed_generator_pruning() -> None:
    """(x0*x1, ..., x0*x7) at max_degree=11 has 12377 standard monomials."""
    variables = tuple(f"x{index}" for index in range(8))
    generators = tuple(
        tuple(1 if axis in (0, index) else 0 for axis in range(8))
        for index in range(1, 8)
    )
    ideal = _monomial_ideal(variables, generators)
    result = hilbert_function(ideal, max_degree=11)
    assert result.values[-1] == 12377


def test_hilbert_function_reports_nonhomogeneous_before_slice_budget() -> None:
    """x0^2 + x0 is nonhomogeneous and must report the domain error, not a bound."""
    variables = tuple(f"x{index}" for index in range(8))
    generator = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(2,) + (0,) * 7,
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(1,) + (0,) * 7,
                ),
            )
        ),
    )
    ideal = RationalPolynomialIdeal(variables=variables, generators=(generator,))
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        hilbert_function(ideal, max_degree=11)


def test_hilbert_series_ambient_numerator_must_reduce_to_series() -> None:
    """A zeroed ambient numerator cannot coexist with a nonzero series."""
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    payload = series.model_dump()
    payload["ambient_numerator"]["polynomial"]["terms"] = []
    with pytest.raises(ValidationError):
        HilbertSeriesResult.model_validate(payload)


@pytest.mark.scale
def test_hilbert_function_admits_mixed_constraints_above_enumerator_ceiling() -> None:
    """The 28 pairwise products have 8 standard monomials at degree 32.

    This sits at the published degree envelope (``MAX_GRADED_DEGREE``) and
    charges a full Groebner pass, so it is near-envelope evidence rather than
    an ordinary regression.  An explicit budget keeps it off the operation's
    short default when the scheduled scale lane is loaded.
    """
    from jacobian.math.polynomials.ideals._models import IdealComputationBudget

    variables = tuple("xyzwuvst")
    generators = tuple(
        tuple(1 if axis in (left, right) else 0 for axis in range(8))
        for left in range(8)
        for right in range(left + 1, 8)
    )
    ideal = _monomial_ideal(variables, generators)
    result = hilbert_function(
        ideal,
        max_degree=MAX_GRADED_DEGREE,
        resource_budget=IdealComputationBudget(wall_seconds=60),
    )
    assert result.values[-1] == 8


def test_hilbert_function_uses_reduced_initial_ideal_leadings() -> None:
    """(x0+x1, x0-x1) under lex reduces to (x0, x1), not the source leadings."""
    variables = tuple("xyzwuvst")
    e0 = (1, 0, 0, 0, 0, 0, 0, 0)
    e1 = (0, 1, 0, 0, 0, 0, 0, 0)

    def binomial(first: int, second: int) -> RationalPolynomial:
        return RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=first, den=1), exponents=e0
                    ),
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=second, den=1), exponents=e1
                    ),
                )
            ),
        )

    ideal = RationalPolynomialIdeal(
        variables=variables, generators=(binomial(1, 1), binomial(1, -1))
    )
    result = hilbert_function(ideal, "lex", max_degree=15)
    assert result.values[-1] == 15504


def test_hilbert_series_structural_validator_rejects_bad_axis() -> None:
    """A numerator in the wrong ring is rejected without symbolic cancellation."""
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    payload = series.model_dump()
    payload["h_numerator"]["variables"] = ["m"]
    with pytest.raises(ValidationError):
        HilbertSeriesResult.model_validate(payload)


def test_support_bound_accounts_for_exponent_thresholds() -> None:
    """x_i^17 x_j^17 cannot divide a degree-32 monomial, so it must not prune."""
    from jacobian.math.polynomials.graded.operations import _support_relaxation_bound

    generators = tuple(
        tuple(17 if axis in (left, right) else 0 for axis in range(8))
        for left in range(8)
        for right in range(left + 1, 8)
    )
    # Non-squarefree generators have exponent thresholds, so the support
    # relaxation is not a sound upper bound and must decline.
    assert _support_relaxation_bound(generators, 8, 32) is None


def test_series_rejects_common_t_minus_one_factor() -> None:
    """A numerator sharing (t-1) with the denominator lowers the dimension.

    For (x) in QQ[x,y] the reduced form is 1/(1-t) with dimension 1; a forged
    payload claiming dimension 2 with numerator 1-t satisfies every other
    identity but reduces to dimension 1.
    """
    import json

    variables = ("x", "y")
    ideal = _monomial_ideal(variables, ((1, 0),))
    base = json.loads(hilbert_series(ideal, prefix_degree=3).model_dump_json())
    numerator_terms = [
        {"coefficient": {"num": "-1", "den": "1"}, "exponents": [1]},
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
    ]
    base["ambient_numerator"] = {
        "domain": "QQ",
        "variables": ["t"],
        "polynomial": {"terms": numerator_terms},
    }
    base["series"]["numerator"] = {"terms": numerator_terms}
    base["reduced_numerator"] = {
        "domain": "QQ",
        "variables": ["t"],
        "polynomial": {"terms": numerator_terms},
    }
    base["h_numerator"] = {
        "domain": "QQ",
        "variables": ["t"],
        "polynomial": {"terms": numerator_terms},
    }
    base["denominator_exponent"] = 2
    base["series"]["denominator"] = {
        "terms": [
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [2]},
            {"coefficient": {"num": "-2", "den": "1"}, "exponents": [1]},
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
        ]
    }
    with pytest.raises(ValidationError, match=r"t-1"):
        HilbertSeriesResult.model_validate_json(json.dumps(base))


def test_series_prefix_must_match_the_h_numerator() -> None:
    """A forged constant prefix coefficient is rejected."""
    series = hilbert_series(_ideal((2, 0)), prefix_degree=3)
    payload = series.model_dump()
    payload["prefix"] = [value + 1 for value in payload["prefix"]]
    with pytest.raises(ValidationError, match="prefix"):
        HilbertSeriesResult.model_validate(payload)


def test_series_rejects_fractional_h_numerator() -> None:
    """A nonintegral h-numerator coefficient is truncated to an invalid series."""
    import json

    series = hilbert_series(_ideal((2, 0)), prefix_degree=3)
    payload = json.loads(series.model_dump_json())
    payload["h_numerator"]["polynomial"]["terms"] = [
        {"coefficient": {"num": "3", "den": "2"}, "exponents": [0]}
    ]
    with pytest.raises(ValidationError):
        HilbertSeriesResult.model_validate_json(json.dumps(payload))


def test_request_models_document_structural_preconditions() -> None:
    """The graded request schemas describe their structural ideal constraints."""
    for model in (
        InitialMonomialIdealRequest,
        HilbertFunctionRequest,
        HilbertSeriesRequest,
        HilbertPolynomialRequest,
    ):
        assert model.model_fields["ideal"].description
    assert StandardMonomialsRequest.model_fields["initial_ideal"].description


def test_pure_power_ideal_skips_the_exact_slice_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pure-power bound is exact, so no composition scan runs.

    A degree slice above the result bound but below the enumeration ceiling
    would otherwise scan every composition before returning the same rejection.
    """
    from jacobian.math.polynomials.graded import operations as module

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pure-power slice entered the exact scan")

    monkeypatch.setattr(module, "_enumerated_standard_monomial_count", fail)
    variables = tuple(f"x{index}" for index in range(8))
    ideal = _monomial_ideal(
        variables,
        (
            (16, 0, 0, 0, 0, 0, 0, 0),
            (0, 16, 0, 0, 0, 0, 0, 0),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        hilbert_function(ideal, max_degree=30)


def test_hilbert_function_inherits_the_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested initial-ideal computation receives the outer deadline."""
    from jacobian.math.polynomials.graded import operations as module

    captured: list[float | None] = []
    original = module.initial_monomial_ideal

    def tracked(ideal: object, order: str, **kwargs: object) -> object:
        captured.append(kwargs.get("_outer_deadline"))  # type: ignore[arg-type]
        return original(ideal, order, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "initial_monomial_ideal", tracked)
    hilbert_function(_ideal((2, 0)), max_degree=2)
    assert captured and captured[0] is not None


def test_nested_groebner_deadline_never_exceeds_the_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subsecond outer remainder must not grant a fresh full second."""
    from jacobian.math.polynomials.graded import operations as module
    from jacobian.math.polynomials.ideals import operations as ideal_module

    seen: list[float | None] = []
    original = ideal_module.groebner_basis

    def tracked(ideal: object, order: str = "grevlex", **kwargs: object) -> object:
        seen.append(kwargs.get("_outer_deadline"))  # type: ignore[arg-type]
        return original(ideal, order, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "groebner_basis", tracked)
    hilbert_function(_nonmonomial_quadratic(), max_degree=2)
    assert seen and seen[0] is not None


def test_nested_budget_keeps_the_request_start_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remaining duration must not be rebased to the request start."""
    import time as time_module

    from jacobian._execution import request_execution
    from jacobian.math.polynomials.graded import operations as module
    from jacobian.math.polynomials.ideals._models import IdealComputationBudget

    now = [1_000.0]
    monkeypatch.setattr(time_module, "monotonic", lambda: now[0])

    class StopError(Exception):
        pass

    captured: dict[str, object] = {}

    def fake_groebner(
        ideal: object, order: str = "grevlex", **kwargs: object
    ) -> object:
        captured["wall_seconds"] = kwargs["resource_budget"].wall_seconds  # type: ignore[union-attr]
        captured["outer_deadline"] = kwargs["_outer_deadline"]
        raise StopError()

    monkeypatch.setattr(module, "groebner_basis", fake_groebner)
    # Six seconds into a ten-second request, four seconds remain before the
    # outer deadline. Rebasing that remainder to the request start would bind
    # the nested worker at started_at + 4 and time out immediately. The source
    # is genuinely non-monomial so the nested Groebner call is reached.
    with (
        request_execution(now[0] - 6, outer_deadline=now[0] + 4),
        pytest.raises(StopError),
    ):
        initial_monomial_ideal(
            _nonmonomial_quadratic(),
            resource_budget=IdealComputationBudget(wall_seconds=10),
        )
    assert captured["wall_seconds"] == 10
    assert captured["outer_deadline"] == now[0] + 4


def test_monomial_ideal_skips_backend_source_admission() -> None:
    """A representable monomial exponent past 20 needs no backend work."""
    result = initial_monomial_ideal(_ideal((21,)))
    assert [
        generator.polynomial.terms[0].exponents
        for generator in result.initial_ideal.generators
    ] == [(21,)]
    assert result.groebner_basis.generators == result.initial_ideal.generators
    assert hilbert_function(_ideal((21,)), max_degree=4).values == (1, 1, 1, 1, 1)


def test_monomial_initial_ideal_minimizes_divisible_generators() -> None:
    """Divisible monomials drop out of the exact initial ideal."""
    result = initial_monomial_ideal(_monomial_ideal(("x",), ((2,), (3,))))
    assert [
        generator.polynomial.terms[0].exponents
        for generator in result.initial_ideal.generators
    ] == [(2,)]
    result = initial_monomial_ideal(
        _monomial_ideal(("x", "y"), ((2, 0), (1, 1), (0, 2)))
    )
    assert sorted(
        generator.polynomial.terms[0].exponents
        for generator in result.initial_ideal.generators
    ) == [(0, 2), (1, 1), (2, 0)]


def test_embedded_initial_ideals_must_be_monomial() -> None:
    """A non-monomial value cannot hide under an initial-ideal context."""
    from jacobian.math.polynomials.values import (
        SparseRationalPolynomial,
    )

    variables = ("x", "y")

    def _poly(terms: tuple[tuple[tuple[int, ...], int], ...]) -> RationalPolynomial:
        from jacobian.math.polynomials.values import RationalPolynomialTerm

        return RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=num, den=1),
                        exponents=exponents,
                    )
                    for exponents, num in terms
                ),
            ),
        )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(_poly((((1, 0), 1),)),),
    )
    non_monomial = RationalPolynomialIdeal(
        variables=variables,
        generators=(_poly((((1, 0), 1), ((0, 1), 1))),),
    )
    with pytest.raises(ValidationError):
        HilbertFunctionResult(
            ideal=ideal,
            initial_ideal=non_monomial,
            monomial_order="grevlex",
            values=(1,),
        )
    with pytest.raises(ValidationError):
        HilbertPolynomialResult(
            ideal=ideal,
            initial_ideal=non_monomial,
            monomial_order="grevlex",
            dimension=1,
            polynomial=RationalPolynomial(
                variables=("m",),
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(1,),
                        ),
                    )
                ),
            ),
            stabilization_degree=1,
        )
    with pytest.raises(ValidationError):
        HilbertDimensionResult(
            ideal=ideal,
            initial_ideal=non_monomial,
            monomial_order="grevlex",
            dimension=1,
        )
    with pytest.raises(ValidationError):
        HilbertMultiplicityResult(
            ideal=ideal,
            initial_ideal=non_monomial,
            monomial_order="grevlex",
            dimension=1,
            multiplicity=1,
        )
    with pytest.raises(ValidationError):
        HVectorResult(
            ideal=ideal,
            initial_ideal=non_monomial,
            monomial_order="grevlex",
            dimension=1,
            h_vector=(1,),
        )
    series = hilbert_series(_ideal((2, 0)), prefix_degree=2)
    payload = series.model_dump(mode="json")
    payload["initial_ideal"] = non_monomial.model_dump(mode="json")
    with pytest.raises(ValidationError):
        HilbertSeriesResult.model_validate(payload)
