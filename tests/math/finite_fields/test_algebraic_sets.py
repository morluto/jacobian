"""Algebraic-set slice (#3726): affine/projective zeros, counts, base change."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields import (
    Axis,
    FieldEmbedding,
    FiniteFieldElement,
    FiniteFieldPresentation,
    affine_zero_count,
    affine_zero_set,
    base_change_system,
    element,
    finite_field,
    projective_zero_count,
    projective_zero_set,
)
from jacobian.math.finite_fields._algebraic_set_models import (
    AffineZeroCountRequest,
    AffineZeroSetRequest,
    BaseChangeRequest,
    ProjectiveZeroCountRequest,
    ProjectiveZeroSetRequest,
    ProjectiveZeroSetResult,
)
from jacobian.math.finite_fields._algebraic_sets import (
    AffinePoint,
    AlgebraicMonomial,
    AlgebraicPolynomial,
    PolynomialSystem,
    verify_affine_zero_set,
)
from jacobian.math.finite_fields._tools import (
    _affine_zero_count,
    _affine_zero_set,
    _base_change,
    _projective_zero_count,
    _projective_zero_set,
)
from jacobian.math.finite_fields.values import ProjectivePoint


def _f2() -> FiniteFieldPresentation:
    return finite_field(2, (0, 1))


def _one(presentation: FiniteFieldPresentation) -> FiniteFieldElement:
    return element(
        presentation,
        (1,) * presentation.degree
        if presentation.degree == 1
        else (1,) + (0,) * (presentation.degree - 1),
    )


def _split_system() -> PolynomialSystem:
    presentation = _f2()
    axis = Axis(name="vars", labels=("x",))
    one = element(presentation, (1,))
    poly = AlgebraicPolynomial._from_kernel(
        presentation=presentation,
        variable_axis=axis,
        terms=(
            AlgebraicMonomial._from_kernel(coefficient=one, exponents=(2,)),
            AlgebraicMonomial._from_kernel(coefficient=one, exponents=(1,)),
        ),
    )
    return PolynomialSystem._from_kernel(
        presentation=presentation, variable_axis=axis, equations=(poly,)
    )


def _projective_system() -> PolynomialSystem:
    presentation = _f2()
    axis = Axis(name="vars", labels=("x", "y"))
    one = element(presentation, (1,))
    poly = AlgebraicPolynomial._from_kernel(
        presentation=presentation,
        variable_axis=axis,
        terms=(AlgebraicMonomial._from_kernel(coefficient=one, exponents=(1, 0)),),
    )
    return PolynomialSystem._from_kernel(
        presentation=presentation, variable_axis=axis, equations=(poly,)
    )


def test_affine_split_zeros_known_answer() -> None:
    system = _split_system()
    points = affine_zero_set(system)
    assert tuple(
        tuple(coordinate.coordinates for coordinate in point.coordinates)
        for point in points
    ) == (((0,),), ((1,),))
    assert affine_zero_count(system) == 2
    # Count agrees exactly with enumeration.
    counted = _affine_zero_count(AffineZeroCountRequest(system=system))
    enumerated = _affine_zero_set(AffineZeroSetRequest(system=system))
    assert counted.point_count == enumerated.point_count == 2


def test_affine_count_does_not_construct_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _split_system()

    def unexpected_point(*args: object, **kwargs: object) -> object:
        raise AssertionError("count-only enumeration constructed a point")

    monkeypatch.setattr(AffinePoint, "_from_kernel", unexpected_point)
    assert affine_zero_count(system) == 2


def test_affine_count_retains_near_envelope_case() -> None:
    presentation = _gf4()
    axis = Axis(name="vars", labels=tuple("xyzwuvpq"))
    zero = element(presentation, (0, 0))
    polynomial = AlgebraicPolynomial._from_kernel(
        presentation=presentation,
        variable_axis=axis,
        terms=(
            AlgebraicMonomial._from_kernel(
                coefficient=zero, exponents=(0,) * len(axis.labels)
            ),
        ),
    )
    system = PolynomialSystem._from_kernel(
        presentation=presentation, variable_axis=axis, equations=(polynomial,)
    )
    assert affine_zero_count(system) == 65_536


def test_projective_count_does_not_construct_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.finite_fields import _algebraic_sets as algebraic_sets

    system = _projective_system()

    def unexpected_point(*args: object, **kwargs: object) -> object:
        raise AssertionError("count-only enumeration constructed a point")

    monkeypatch.setattr(algebraic_sets, "ProjectivePoint", unexpected_point)
    assert projective_zero_count(system) == 1


def test_projective_single_class_known_answer() -> None:
    system = _projective_system()
    points = projective_zero_set(system)
    assert len(points) == 1
    assert points[0].coordinates[0].is_zero
    assert points[0].coordinates[1].is_one
    assert projective_zero_count(system) == 1
    counted = _projective_zero_count(ProjectiveZeroCountRequest(system=system))
    enumerated = _projective_zero_set(ProjectiveZeroSetRequest(system=system))
    assert counted.point_count == enumerated.point_count == 1


def test_projective_rejects_inhomogeneous() -> None:
    system = _split_system()
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        projective_zero_set(system)


def test_equation_reordering_is_a_metamorphism() -> None:
    system = _split_system()
    reordered = PolynomialSystem._from_kernel(
        presentation=system.presentation,
        variable_axis=system.variable_axis,
        equations=tuple(reversed(system.equations)),
    )
    assert affine_zero_count(reordered) == affine_zero_count(system)
    # Term reordering within one equation preserves the locus.
    equation = system.equations[0]
    swapped = AlgebraicPolynomial._from_kernel(
        presentation=equation.presentation,
        variable_axis=equation.variable_axis,
        terms=tuple(reversed(equation.terms)),
    )
    swapped_system = PolynomialSystem._from_kernel(
        presentation=system.presentation,
        variable_axis=system.variable_axis,
        equations=(swapped,),
    )
    assert affine_zero_count(swapped_system) == 2


def test_base_change_f2_to_gf4_preserves_split_count() -> None:
    system = _split_system()
    target = finite_field(2, (1, 1, 1))
    embedding = FieldEmbedding._from_kernel(
        source=system.presentation,
        target=target,
        generator_image=element(target, (0, 0)),
    )
    transported = base_change_system(system, embedding)
    assert transported.presentation == target
    assert transported.variable_axis == system.variable_axis
    assert affine_zero_count(transported) == 2
    result = _base_change(BaseChangeRequest(system=system, embedding=embedding))
    assert result.transported == transported


def test_base_change_rejects_bad_generator_image() -> None:
    system = _split_system()
    target = finite_field(2, (1, 1, 1))
    bad = FieldEmbedding._from_kernel(
        source=system.presentation,
        target=target,
        generator_image=element(target, (1, 0)),
    )
    with pytest.raises(OperationDomainValidationError, match="source modulus"):
        base_change_system(system, bad)


def test_base_change_rejects_characteristic_mismatch() -> None:
    system = _split_system()
    target = finite_field(3, (0, 1))
    embedding = FieldEmbedding._from_kernel(
        source=system.presentation,
        target=target,
        generator_image=element(target, (0,)),
    )
    with pytest.raises(OperationDomainValidationError, match="characteristics"):
        base_change_system(system, embedding)


def test_ambient_budget_is_resource_refusal() -> None:
    # GF(2^8) has order 256; three variables give an ambient 16M > 65536.
    presentation = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    axis = Axis(name="vars", labels=("x", "y", "z"))
    one = element(presentation, (1,) + (0,) * 7)
    poly = AlgebraicPolynomial._from_kernel(
        presentation=presentation,
        variable_axis=axis,
        terms=(AlgebraicMonomial._from_kernel(coefficient=one, exponents=(1, 0, 0)),),
    )
    system = PolynomialSystem._from_kernel(
        presentation=presentation, variable_axis=axis, equations=(poly,)
    )
    with pytest.raises(OperationResourceAdmissionError, match="ambient"):
        affine_zero_set(system)


def test_serialized_affine_result_round_trip() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.finite_fields._algebraic_set_models import AffineZeroSetResult

    system = _split_system()
    result = _affine_zero_set(AffineZeroSetRequest(system=system))
    restored = AffineZeroSetResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def _gf4() -> FiniteFieldPresentation:
    return finite_field(2, (1, 1, 1))


def _vanishing_x_system(
    presentation: FiniteFieldPresentation, labels: tuple[str, ...] = ("x", "y")
) -> PolynomialSystem:
    axis = Axis(name="vars", labels=labels)
    one = _one(presentation)
    poly = AlgebraicPolynomial._from_kernel(
        presentation=presentation,
        variable_axis=axis,
        terms=(
            AlgebraicMonomial._from_kernel(
                coefficient=one, exponents=(1,) + (0,) * (len(labels) - 1)
            ),
        ),
    )
    return PolynomialSystem._from_kernel(
        presentation=presentation, variable_axis=axis, equations=(poly,)
    )


def test_projective_classes_are_normalized_and_unique_over_gf4() -> None:
    presentation = _gf4()
    system = _vanishing_x_system(presentation)
    points = projective_zero_set(system)
    assert projective_zero_count(system) == 1
    assert len(points) == 1
    assert len({point.digest for point in points}) == 1
    coordinates = points[0].coordinates
    assert coordinates[0].is_zero
    assert coordinates[1].is_one
    assert points[0].axis == system.variable_axis


def test_affine_cone_count_recovers_projective_count() -> None:
    for presentation in (_f2(), _gf4()):
        system = _vanishing_x_system(presentation)
        order = presentation.order
        affine = affine_zero_count(system)
        assert affine > 1
        assert projective_zero_count(system) == (affine - 1) // (order - 1)


def test_projective_result_rejects_duplicate_scalar_classes() -> None:
    system = _projective_system()
    points = projective_zero_set(system)
    assert len(points) == 1
    with pytest.raises(ValidationError, match="distinct scalar classes"):
        ProjectiveZeroSetResult(
            system=system,
            points=(*points, points[0]),
            point_count=2,
        )


def test_projective_point_rejects_nonnormalized_coordinates() -> None:
    presentation = _gf4()
    axis = Axis(name="vars", labels=("x", "y"))
    with pytest.raises(ValidationError, match="normalized"):
        ProjectivePoint(
            presentation=presentation,
            axis=axis,
            coordinates=(
                element(presentation, (1, 1)),
                element(presentation, (1, 0)),
            ),
        )


def test_verify_affine_zero_set_accepts_exact_family_and_rejects_forgery() -> None:
    system = _split_system()
    points = affine_zero_set(system)
    assert verify_affine_zero_set(system, points) is True
    forged = (
        AffinePoint(
            presentation=points[0].presentation,
            variable_axis=points[0].variable_axis,
            coordinates=points[0].coordinates,
        ),
    )
    assert verify_affine_zero_set(system, forged) is False


def test_homogeneous_projective_count_survives_base_change() -> None:
    system = _vanishing_x_system(_f2())
    embedding = FieldEmbedding._from_kernel(
        source=system.presentation,
        target=_gf4(),
        generator_image=element(_gf4(), (0, 0)),
    )
    transported = base_change_system(system, embedding)
    assert projective_zero_count(transported) == projective_zero_count(system) == 1
