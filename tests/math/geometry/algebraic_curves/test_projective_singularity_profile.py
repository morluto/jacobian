"""Exact global projective plane-curve singular-locus evidence."""

from __future__ import annotations

import shutil
import threading
from fractions import Fraction
from itertools import product
from typing import cast

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.algebraic_curves._singularity import (
    _admit_singularity,
    _ideal_projection_limit_failure,
    _normalized_source,
    _profile,
    _to_public_polynomial,
)
from jacobian.math.geometry.algebraic_curves._singularity_models import (
    MAX_PROJECTIVE_SINGULAR_COMPONENTS,
    PositiveDimensionalProjectivePlaneCurveSingularLocus,
    ProjectivePlaneCurveSingularityProfile,
    ProjectivePlaneCurveSingularityRequest,
)
from jacobian.math.geometry.algebraic_curves._singularity_point_worker import (
    _shape_data,
)
from jacobian.math.geometry.algebraic_curves._tools import (
    TOOLS,
    compute_projective_plane_curve_singularity_profile,
)
from jacobian.math.geometry.algebraic_curves.operations import singularity_profile
from jacobian.math.number_theory.number_fields.values import (
    ComplexNumberFieldEmbedding,
    RealNumberFieldEmbedding,
)
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import bounded_process_cancellation

pytestmark = pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="the exact projective singular-locus backend is unavailable",
)

_AXIS = ("X", "Y", "Z")


def _rational(value: int | Fraction) -> CanonicalRational:
    fraction = Fraction(value)
    return CanonicalRational.from_fraction(fraction)


def _polynomial(
    *terms: tuple[int | Fraction, tuple[int, int, int]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=_AXIS,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _chart_polynomial(
    *terms: tuple[int, tuple[int, int]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("u", "v"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _compute(source: RationalPolynomial) -> ProjectivePlaneCurveSingularityProfile:
    return compute_projective_plane_curve_singularity_profile(
        ProjectivePlaneCurveSingularityRequest(polynomial=source)
    )


def _assert_singular_point_defining_identities(
    result: ProjectivePlaneCurveSingularityProfile,
) -> None:
    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    parameter = sympy.Symbol("alpha")
    for record in result.outcome.points:
        presentation = record.point.embedding.presentation
        modulus = sympy.Poly.from_list(
            [int(coefficient) for coefficient in presentation.coefficients_descending],
            gens=parameter,
            domain=sympy.QQ,
        )
        coordinate_expressions = tuple(
            sum(
                sympy.Rational(*coefficient.as_integer_ratio()) * parameter**power
                for power, coefficient in enumerate(coordinate.coefficients_ascending)
            )
            for coordinate in record.point.coordinates
        )
        for polynomial in (
            result.source_polynomial,
            *result.partial_derivatives,
        ):
            backend = rational_polynomial_to_sympy(polynomial)
            substitutions = dict(zip(backend.gens, coordinate_expressions, strict=True))
            value = sympy.Poly(
                sympy.expand(backend.as_expr().subs(substitutions)),
                parameter,
                domain=sympy.QQ,
            ).rem(modulus)
            assert value.is_zero


def _ideal_with_shape(
    *,
    coefficient: CanonicalRational,
    generator_count: int,
    terms_per_generator: int,
    axis: tuple[str, str, str] = _AXIS,
) -> RationalPolynomialIdeal:
    exponents = tuple(
        sorted(
            (
                exponent
                for exponent in product(range(5), repeat=3)
                if sum(exponent) <= 4
            ),
            reverse=True,
        )[:terms_per_generator]
    )
    generator = RationalPolynomial(
        variables=axis,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=coefficient,
                    exponents=exponent,
                )
                for exponent in exponents
            )
        ),
    )
    return RationalPolynomialIdeal(
        variables=axis,
        generators=(generator,) * generator_count,
    )


def test_conjugate_nonrational_singularities_keep_distinct_embedding_identity() -> None:
    result = _compute(
        _polynomial(
            (1, (2, 0, 1)),
            (1, (0, 2, 1)),
            (1, (0, 0, 3)),
        )
    )

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    assert len(result.outcome.points) == 2
    roots: list[int] = []
    for record in result.outcome.points:
        point = record.point
        assert isinstance(point.embedding, ComplexNumberFieldEmbedding)
        assert point.embedding.presentation.coefficients_descending == ("1", "0", "1")
        roots.append(point.embedding.root.root_index)
        assert tuple(
            tuple(
                coefficient.as_fraction()
                for coefficient in coordinate.coefficients_ascending
            )
            for coordinate in point.coordinates
        ) == (
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
            (Fraction(0), Fraction(0)),
        )
        assert point.chart_index == 0
    assert roots == [0, 1]
    _assert_singular_point_defining_identities(result)

    payload = result.model_dump(mode="json")
    assert ProjectivePlaneCurveSingularityProfile.model_validate(payload) == result


def test_geometrically_split_cubic_keeps_its_complete_galois_orbit() -> None:
    # This is Norm_{QQ(cuberoot(2))/QQ}(X + alpha*Y + alpha^2*Z).
    # Its three conjugate lines meet in one degree-three closed-point orbit.
    result = _compute(
        _polynomial(
            (1, (3, 0, 0)),
            (-6, (1, 1, 1)),
            (2, (0, 3, 0)),
            (4, (0, 0, 3)),
        )
    )

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    assert len(result.outcome.points) == 3
    presentations = {
        record.point.embedding.presentation for record in result.outcome.points
    }
    assert len(presentations) == 1
    (presentation,) = presentations
    assert presentation.degree == 3
    assert presentation.coefficients_descending[0] != "1"

    real_embeddings = [
        record.point.embedding
        for record in result.outcome.points
        if isinstance(record.point.embedding, RealNumberFieldEmbedding)
    ]
    complex_embeddings = [
        record.point.embedding
        for record in result.outcome.points
        if isinstance(record.point.embedding, ComplexNumberFieldEmbedding)
    ]
    assert len(real_embeddings) == 1
    assert len(complex_embeddings) == 2
    assert real_embeddings[0].root.real_root_index == 0
    assert sorted(embedding.root.root_index for embedding in complex_embeddings) == [
        1,
        2,
    ]
    _assert_singular_point_defining_identities(result)


@pytest.mark.parametrize(
    "source",
    [
        _polynomial((1, (2, 0, 0)), (1, (0, 2, 0)), (1, (0, 0, 2))),
        _polynomial((1, (3, 0, 0)), (1, (0, 3, 0)), (1, (0, 0, 3))),
    ],
)
def test_smooth_curve_retains_the_saturated_unit_ideal(
    source: RationalPolynomial,
) -> None:
    result = _compute(source)

    assert result.outcome.status == "SMOOTH_OVER_ALGEBRAIC_CLOSURE"
    saturation = result.outcome.saturated_jacobian_ideal
    assert len(saturation.generators) == 1
    assert rational_polynomial_to_sympy(saturation.generators[0]).as_expr() == 1


@pytest.mark.parametrize(
    ("source", "chart_index", "coordinates"),
    [
        (
            _polynomial((1, (3, 0, 0)), (-1, (0, 2, 1))),
            2,
            (Fraction(0), Fraction(0), Fraction(1)),
        ),
        (
            _polynomial((1, (2, 1, 0)), (-1, (0, 0, 3))),
            1,
            (Fraction(0), Fraction(1), Fraction(0)),
        ),
    ],
)
def test_disjoint_chart_cover_finds_singularities_at_infinity(
    source: RationalPolynomial,
    chart_index: int,
    coordinates: tuple[Fraction, Fraction, Fraction],
) -> None:
    result = _compute(source)

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    assert len(result.outcome.points) == 1
    point = result.outcome.points[0].point
    assert point.chart_index == chart_index
    assert (
        tuple(
            coordinate.coefficients_ascending[0].as_fraction()
            for coordinate in point.coordinates
        )
        == coordinates
    )


def test_nodal_cubic_returns_an_exact_zero_first_jet() -> None:
    result = _compute(
        _polynomial(
            (-1, (3, 0, 0)),
            (-1, (2, 0, 1)),
            (1, (0, 2, 1)),
        )
    )

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    (record,) = result.outcome.points
    assert all(
        coefficient.as_fraction() == 0
        for value in (record.first_jet.value, *record.first_jet.gradient)
        for coefficient in value.coefficients_ascending
    )
    _assert_singular_point_defining_identities(result)


def test_three_coordinate_lines_return_one_point_in_each_disjoint_chart() -> None:
    result = _compute(_polynomial((1, (1, 1, 1))))

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    assert len(result.outcome.points) == 3
    assert {record.point.chart_index for record in result.outcome.points} == {0, 1, 2}


def test_concurrent_lines_reduce_a_nonreduced_local_scheme_to_one_point() -> None:
    result = _compute(
        _polynomial(
            (1, (2, 1, 0)),
            (-1, (1, 2, 0)),
        )
    )

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    assert len(result.outcome.points) == 1
    assert result.outcome.points[0].point.chart_index == 2


def test_second_chart_keeps_a_complete_quadratic_galois_orbit() -> None:
    result = _compute(
        _polynomial(
            (1, (1, 2, 0)),
            (1, (1, 0, 2)),
        )
    )

    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    second_chart = [
        record for record in result.outcome.points if record.point.chart_index == 1
    ]
    assert len(second_chart) == 2
    assert [
        cast(ComplexNumberFieldEmbedding, record.point.embedding).root.root_index
        for record in second_chart
    ] == [0, 1]


def test_repeated_line_is_positive_dimensional_without_squarefree_replacement() -> None:
    source = _polynomial((1, (2, 0, 1)))
    result = _compute(source)

    assert result.outcome.status == "SINGULAR_POSITIVE_DIMENSIONAL"
    assert result.outcome.projective_dimension == 1
    assert result.outcome.affine_cone_dimension == 2
    assert result.source_polynomial == source
    assert result.outcome.rational_minimal_components
    assert any(
        rational_polynomial_to_sympy(generator).as_expr() == sympy.Symbol("X")
        for component in result.outcome.rational_minimal_components
        for generator in component.generators
    )


def test_nonzero_rational_scalar_associates_have_the_same_profile() -> None:
    primitive = _polynomial(
        (1, (3, 0, 0)),
        (-1, (0, 2, 1)),
    )
    scaled = _polynomial(
        (Fraction(-2, 3), (3, 0, 0)),
        (Fraction(2, 3), (0, 2, 1)),
    )

    assert _compute(primitive) == _compute(scaled)


def test_shape_search_passes_two_colliding_integer_forms_before_separating() -> None:
    four_points = RationalPolynomialIdeal(
        variables=("u", "v"),
        generators=(
            _chart_polynomial((1, (2, 0)), (-1, (1, 0))),
            _chart_polynomial((1, (0, 2)), (-1, (0, 1))),
        ),
    )

    parameter, eliminant, coordinates = _shape_data(four_points)

    assert eliminant.degree() == 4
    assert (
        sympy.expand(coordinates[0].as_expr() + 2 * coordinates[1].as_expr())
        == parameter
    )


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            RationalPolynomial(
                variables=_AXIS,
                polynomial=SparseRationalPolynomial(terms=()),
            ),
            "projective_plane_curve.zero_source",
        ),
        (
            _polynomial((1, (2, 0, 0)), (1, (0, 1, 0))),
            "projective_plane_curve.homogeneity",
        ),
        (
            _polynomial((1, (4, 0, 0)), (1, (0, 4, 0))),
            "projective_plane_curve.source_bound",
        ),
    ],
)
def test_request_rejects_sources_outside_the_exact_projective_domain(
    source: RationalPolynomial,
    code: str,
) -> None:
    with pytest.raises(ValidationError) as caught:
        ProjectivePlaneCurveSingularityRequest(polynomial=source)
    assert caught.value.errors()[0]["type"] == code


def test_normalized_height_is_rejected_before_singular_launch() -> None:
    accepted = _polynomial(
        (99_999_999, (3, 0, 0)),
        (1, (0, 2, 1)),
    )
    assert _compute(accepted).outcome.status == "SINGULAR_ZERO_DIMENSIONAL"

    source = _polynomial(
        (100_000_000, (3, 0, 0)),
        (1, (0, 2, 1)),
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        singularity_profile(source)
    assert caught.value.errors()[0]["type"] == (
        "projective_plane_curve.normalized_height_bound"
    )


def test_decoded_ideal_cannot_exceed_the_admitted_coefficient_bound() -> None:
    source = _normalized_source(
        _polynomial(
            (99_999_999, (3, 0, 0)),
            (1, (2, 1, 0)),
        )
    )
    admission = _admit_singularity(source)
    over_bound = CanonicalRational(
        num="1" + "0" * admission.macaulay_minor_component_digits,
        den="1",
    )
    ideal = _ideal_with_shape(
        coefficient=over_bound,
        generator_count=1,
        terms_per_generator=1,
    )

    failure = _ideal_projection_limit_failure(
        "SATURATION",
        (ideal,),
        admission,
        maximum_ideals=1,
    )

    assert failure is not None
    assert failure.status == "LIMIT_EXCEEDED"
    assert failure.stage == "SATURATION"


def test_derived_result_bound_covers_the_maximal_admitted_ideal_shape() -> None:
    maximal_axis = ("X" * 32, "Y" * 32, "Z" * 32)
    cubic_exponents = tuple(
        sorted(
            exponent for exponent in product(range(4), repeat=3) if sum(exponent) == 3
        )
    )
    source_backend = _normalized_source(
        RationalPolynomial(
            variables=maximal_axis,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=_rational(99_999_999 if index == 1 else index),
                        exponents=exponents,
                    )
                    for index, exponents in enumerate(
                        reversed(cubic_exponents), start=1
                    )
                )
            ),
        )
    )
    admission = _admit_singularity(source_backend)
    assert admission.macaulay_minor_component_digits == 139
    numerator = 10 ** (admission.macaulay_minor_component_digits - 1)
    denominator_base = 10**admission.macaulay_minor_component_digits

    saturation = _ideal_with_shape(
        coefficient=CanonicalRational(
            num=str(numerator),
            den=str(denominator_base - 1),
        ),
        generator_count=64,
        terms_per_generator=16,
        axis=maximal_axis,
    )
    components = tuple(
        _ideal_with_shape(
            coefficient=CanonicalRational(
                num=str(numerator),
                den=str(denominator_base - (10 * index + 1)),
            ),
            generator_count=4,
            terms_per_generator=16,
            axis=maximal_axis,
        )
        for index in range(MAX_PROJECTIVE_SINGULAR_COMPONENTS)
    )
    components = tuple(sorted(components, key=lambda ideal: ideal.model_dump_json()))

    assert (
        _ideal_projection_limit_failure(
            "SATURATION",
            (saturation,),
            admission,
            maximum_ideals=1,
        )
        is None
    )
    assert (
        _ideal_projection_limit_failure(
            "PROJECTIVE_COMPONENTS",
            components,
            admission,
            maximum_ideals=MAX_PROJECTIVE_SINGULAR_COMPONENTS,
        )
        is None
    )

    axis = cast(
        tuple[str, str, str], tuple(str(symbol) for symbol in source_backend.gens)
    )
    source = _to_public_polynomial(source_backend, axis)
    partials = cast(
        tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial],
        tuple(
            _to_public_polynomial(source_backend.diff(symbol), axis)
            for symbol in source_backend.gens
        ),
    )
    profile = _profile(
        source=source,
        partials=partials,
        outcome=PositiveDimensionalProjectivePlaneCurveSingularLocus._from_kernel(
            ideal=saturation,
            components=components,
        ),
    )

    encoded_size = len(encode_strict_json(profile.model_dump(mode="json")))
    assert encoded_size <= admission.predicted_result_bytes
    assert admission.predicted_result_bytes < 10 * 1_024 * 1_024


def test_every_outcome_discriminator_is_required_by_schema_and_runtime() -> None:
    schema = ProjectivePlaneCurveSingularityProfile.model_json_schema()
    for definition in (
        "SmoothProjectivePlaneCurve",
        "ZeroDimensionalProjectivePlaneCurveSingularLocus",
        "PositiveDimensionalProjectivePlaneCurveSingularLocus",
        "IncompleteProjectivePlaneCurveSingularityComputation",
    ):
        assert "status" in schema["$defs"][definition]["required"]


def test_catalog_contract_and_example_round_trip() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "algebraic_geometry.projective_plane_curve.singularity_profile.compute"
    )

    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.outcome.status == "SINGULAR_ZERO_DIMENSIONAL"
    assert tool.result_type.model_validate(result.model_dump(mode="json")) == result


def test_cancelled_saturation_is_not_a_mathematical_outcome() -> None:
    cancellation = threading.Event()
    cancellation.set()
    source = _polynomial((1, (2, 0, 0)), (1, (0, 2, 0)), (1, (0, 0, 2)))

    with bounded_process_cancellation(cancellation):
        result = _compute(source)

    assert result.outcome.status == "CANCELLED"
    assert result.outcome.stage == "SATURATION"


def test_native_operation_accepts_the_canonical_polynomial_value() -> None:
    source = _polynomial((1, (2, 0, 0)), (1, (0, 2, 0)), (1, (0, 0, 2)))

    assert singularity_profile(source).outcome.status == (
        "SMOOTH_OVER_ALGEBRAIC_CLOSURE"
    )
