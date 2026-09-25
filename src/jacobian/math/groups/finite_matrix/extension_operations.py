"""Exact bounded GL/SL constructions over presented extension fields."""

from __future__ import annotations

from itertools import product
from math import ceil
from typing import Any, NoReturn

import rfc8785
from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields.values import (
    Axis,
    AxisBoundMatrix,
    FiniteFieldElement,
    FiniteFieldPresentation,
    ProjectivePoint,
)
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.groups.finite_matrix._models import (
    MAX_EXTENSION_FIELD_LINEAR_GROUP_GENERATORS,
    MAX_LINEAR_GROUP_DIMENSION,
    MAX_LINEAR_GROUP_GENERATOR_CELLS,
    MAX_LINEAR_GROUP_ORDER_DIGITS,
    MAX_PROJECTIVE_POINT_ACTION_SIZE,
    ExtensionFieldGeneralLinearGroup,
    ExtensionFieldGeneralLinearProjectiveAction,
    ExtensionFieldSpecialLinearGroup,
    ExtensionFieldSpecialLinearProjectiveAction,
)

_MAX_PRIMITIVE_SEARCH_WORK = 10_000_000
_MAX_GROUP_OUTPUT_BYTES = CanonicalLimits().max_output_bytes


def _domain_error(
    reason: str, message: str, location: tuple[str | int, ...] = ()
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location, code=f"finite_matrix_group.{reason}", message=message
    )


def _resource_error(
    reason: str, message: str, location: tuple[str | int, ...] = ()
) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=location, code=f"finite_matrix_group.{reason}", message=message
    )


def _field_context(presentation: FiniteFieldPresentation) -> Any:
    from flint import fmpz_mod_poly_ctx, fq_default_ctx

    modulus = fmpz_mod_poly_ctx(presentation.characteristic)(
        list(presentation.modulus_coefficients)
    )
    return fq_default_ctx(modulus=modulus)


def _field_element(
    presentation: FiniteFieldPresentation, value: Any
) -> FiniteFieldElement:
    degree = presentation.degree
    coordinates = tuple(int(coefficient) for coefficient in value.to_list())
    coordinates = (coordinates + (0,) * degree)[:degree]
    return FiniteFieldElement(presentation=presentation, coordinates=coordinates)


def _encoded_element(
    context: Any, presentation: FiniteFieldPresentation, value: int
) -> Any:
    prime = presentation.characteristic
    coordinates = tuple(
        (value // prime**power) % prime for power in range(presentation.degree)
    )
    return context(list(coordinates))


def _factor_distinct(value: int) -> tuple[int, ...]:
    factors: list[int] = []
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1 if divisor == 2 else 2
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def _primitive_element(context: Any, presentation: FiniteFieldPresentation) -> Any:
    order = presentation.order
    multiplicative_order = order - 1
    factors = _factor_distinct(multiplicative_order)
    work_bound = multiplicative_order * max(1, len(factors)) * ceil(order.bit_length())
    if work_bound > _MAX_PRIMITIVE_SEARCH_WORK:
        _resource_error(
            "primitive_element_search_bound",
            "the exact primitive-element search exceeds its work envelope",
        )
    for encoded in range(1, order):
        request_checkpoint("finite-field primitive-element search")
        candidate = _encoded_element(context, presentation, encoded)
        if all(
            candidate ** (multiplicative_order // factor) != context.one()
            for factor in factors
        ):
            return candidate
    raise RuntimeError("finite field has no primitive element")


def _matrix(
    presentation: FiniteFieldPresentation,
    axis: Axis,
    entries: tuple[tuple[Any, ...], ...],
) -> AxisBoundMatrix:
    return AxisBoundMatrix(
        presentation=presentation,
        row_axis=axis,
        column_axis=axis,
        entries=tuple(
            tuple(_field_element(presentation, entry) for entry in row)
            for row in entries
        ),
    )


def _quoted_text_bytes_upper_bound(value: str) -> int:
    return len(rfc8785.dumps(value))


def _presentation_bytes_upper_bound(presentation: FiniteFieldPresentation) -> int:
    return (
        256
        + _quoted_text_bytes_upper_bound(presentation.generator)
        + 16 * len(presentation.modulus_coefficients)
    )


def _axis_bytes_upper_bound(axis: Axis) -> int:
    return (
        256
        + _quoted_text_bytes_upper_bound(axis.name)
        + sum(_quoted_text_bytes_upper_bound(label) for label in axis.labels)
        + 2 * len(axis.labels)
    )


def _check_group_output_size(
    presentation: FiniteFieldPresentation,
    axis: Axis,
    generator_count: int,
    order_digit_bound: int,
) -> None:
    dimension = len(axis.labels)
    presentation_bytes = _presentation_bytes_upper_bound(presentation)
    axis_bytes = _axis_bytes_upper_bound(axis)
    # Each field element serializes its complete presentation, and each matrix
    # repeats both labelled axes. This deliberately bounds the actual carrier
    # JSON shape rather than only counting matrix cells.
    element_bytes = presentation_bytes + 16 * presentation.degree + 96
    matrix_bytes = (
        presentation_bytes
        + 2 * axis_bytes
        + dimension * dimension * element_bytes
        + 512
    )
    total = (
        presentation_bytes
        + axis_bytes
        + order_digit_bound
        + generator_count * matrix_bytes
        + 1024
        # Reserve for SL's ambient order and determinant-index fields. GL may
        # use less, but both constructors share this admission path.
        + 2 * order_digit_bound
    )
    if total > _MAX_GROUP_OUTPUT_BYTES:
        _resource_error(
            "serialized_group_output_bound",
            "the complete group value exceeds the 10 MiB canonical output envelope",
            ("vector_axis",),
        )


def _check_projective_action_output_size(
    group: ExtensionFieldGeneralLinearGroup | ExtensionFieldSpecialLinearGroup,
    point_count: int,
) -> None:
    presentation = group.presentation
    axis = group.vector_axis
    dimension = len(axis.labels)
    presentation_bytes = _presentation_bytes_upper_bound(presentation)
    axis_bytes = _axis_bytes_upper_bound(axis)
    element_bytes = presentation_bytes + 16 * presentation.degree + 96
    group_bytes = _group_value_bytes_upper_bound(group)
    point_bytes = presentation_bytes + axis_bytes + dimension * element_bytes + 512
    # The point labels encode every coordinate as at most five decimal digits;
    # permutation entries are indices into an axis of at most 50 points.
    label_bytes = 16 + dimension * presentation.degree * 6
    action_bytes = (
        point_count * (point_bytes + label_bytes)
        + len(group.generators) * point_count * 4
        + 1024
    )
    if group_bytes + action_bytes > _MAX_GROUP_OUTPUT_BYTES:
        _resource_error(
            "serialized_projective_action_output_bound",
            "the complete projective action value exceeds the 10 MiB canonical output envelope",
            ("group", "vector_axis"),
        )


def _group_value_bytes_upper_bound(
    group: ExtensionFieldGeneralLinearGroup | ExtensionFieldSpecialLinearGroup,
) -> int:
    dimension = len(group.vector_axis.labels)
    presentation_bytes = _presentation_bytes_upper_bound(group.presentation)
    axis_bytes = _axis_bytes_upper_bound(group.vector_axis)
    element_bytes = presentation_bytes + 16 * group.presentation.degree + 96
    matrix_bytes = (
        presentation_bytes
        + 2 * axis_bytes
        + dimension * dimension * element_bytes
        + 512
    )
    total = (
        presentation_bytes
        + axis_bytes
        + len(str(group.order))
        + len(group.generators) * matrix_bytes
        + 2048
    )
    if isinstance(group, ExtensionFieldSpecialLinearGroup):
        total += len(str(group.ambient_general_linear_order)) + len(
            str(group.determinant_index)
        )
    return total


def _identity(context: Any, dimension: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(
            context.one() if row == column else context.zero()
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _elementary_generators(
    context: Any, presentation: FiniteFieldPresentation, axis: Axis
) -> tuple[AxisBoundMatrix, ...]:
    dimension = len(axis.labels)
    if dimension == 1:
        return (_matrix(presentation, axis, _identity(context, dimension)),)
    basis = tuple(context.gen() ** power for power in range(presentation.degree))
    generators: list[AxisBoundMatrix] = []
    for coefficient in basis:
        for index in range(dimension - 1):
            for source, target in ((index + 1, index), (index, index + 1)):
                entries = [list(row) for row in _identity(context, dimension)]
                entries[target][source] = coefficient
                generators.append(
                    _matrix(
                        presentation,
                        axis,
                        tuple(tuple(row) for row in entries),
                    )
                )
    return tuple(generators)


def _general_generators(
    context: Any, presentation: FiniteFieldPresentation, axis: Axis
) -> tuple[AxisBoundMatrix, ...]:
    dimension = len(axis.labels)
    if dimension == 1:
        identity = [list(row) for row in _identity(context, dimension)]
        identity[0][0] = _primitive_element(context, presentation)
        return (_matrix(presentation, axis, tuple(tuple(row) for row in identity)),)
    result = list(_elementary_generators(context, presentation, axis))
    diagonal = [list(row) for row in _identity(context, dimension)]
    diagonal[0][0] = _primitive_element(context, presentation)
    result.append(_matrix(presentation, axis, tuple(tuple(row) for row in diagonal)))
    return tuple(result)


def _linear_group_order(field_order: int, dimension: int) -> int:
    full_space = field_order**dimension
    result = 1
    for rank in range(dimension):
        result *= full_space - field_order**rank
    return result


def _admit_group_parameters(
    presentation: object, axis: object
) -> tuple[FiniteFieldPresentation, Axis, Any, int, int]:
    if type(presentation) is not FiniteFieldPresentation:
        _domain_error(
            "presentation_type",
            "presentation must be a canonical finite-field presentation",
            ("presentation",),
        )
    if type(axis) is not Axis:
        _domain_error(
            "axis_type",
            "vector_axis must be a canonical finite-field axis",
            ("vector_axis",),
        )
    try:
        presentation = FiniteFieldPresentation.model_validate_json(
            presentation.model_dump_json(warnings=False), strict=True
        )
        axis = Axis.model_validate_json(
            axis.model_dump_json(warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=(),
            code="finite_matrix_group.input_shape",
            message="the finite-field presentation and vector axis must be canonical values",
        ) from exc
    if presentation.degree < 2:
        _domain_error(
            "extension_field_required",
            "extension-field matrix groups require presentation degree at least two",
            ("presentation",),
        )
    if not 1 <= len(axis.labels) <= MAX_LINEAR_GROUP_DIMENSION:
        _resource_error(
            "dimension_bound",
            f"dimension must not exceed {MAX_LINEAR_GROUP_DIMENSION}",
            ("vector_axis",),
        )
    require_field(presentation)
    q = presentation.order
    dimension = len(axis.labels)
    generator_count = (
        1 if dimension == 1 else 2 * (dimension - 1) * presentation.degree + 1
    )
    if generator_count > MAX_EXTENSION_FIELD_LINEAR_GROUP_GENERATORS:
        _resource_error(
            "generator_count_bound",
            "the canonical complete generator family exceeds its count envelope",
            ("presentation", "vector_axis"),
        )
    if generator_count * dimension * dimension > MAX_LINEAR_GROUP_GENERATOR_CELLS:
        _resource_error(
            "generator_output_bound",
            "the canonical generator matrices exceed the aggregate output envelope",
            ("vector_axis",),
        )
    order_digit_bound = dimension * dimension * len(str(q)) + 1
    if order_digit_bound > MAX_LINEAR_GROUP_ORDER_DIGITS:
        _resource_error(
            "order_output_bound",
            "the exact group order exceeds its canonical digit budget",
            ("vector_axis",),
        )
    _check_group_output_size(
        presentation,
        axis,
        generator_count,
        order_digit_bound,
    )
    order = _linear_group_order(q, dimension)
    return presentation, axis, _field_context(presentation), q, order


def construct_extension_general_linear_group(
    presentation: FiniteFieldPresentation, vector_axis: Axis
) -> ExtensionFieldGeneralLinearGroup:
    presentation, vector_axis, context, _q, order = _admit_group_parameters(
        presentation, vector_axis
    )
    generators = _general_generators(context, presentation, vector_axis)
    return ExtensionFieldGeneralLinearGroup(
        presentation=presentation,
        vector_axis=vector_axis,
        order=order,
        generators=generators,
    )


def construct_extension_special_linear_group(
    presentation: FiniteFieldPresentation, vector_axis: Axis
) -> ExtensionFieldSpecialLinearGroup:
    presentation, vector_axis, context, q, general_order = _admit_group_parameters(
        presentation, vector_axis
    )
    generators = _elementary_generators(context, presentation, vector_axis)
    determinant_index = q - 1
    return ExtensionFieldSpecialLinearGroup(
        presentation=presentation,
        vector_axis=vector_axis,
        order=general_order // determinant_index,
        generators=generators,
        ambient_general_linear_order=general_order,
        determinant_index=determinant_index,
    )


def _revalidate_general_linear_group(
    group: ExtensionFieldGeneralLinearGroup,
) -> ExtensionFieldGeneralLinearGroup:
    try:
        admitted = ExtensionFieldGeneralLinearGroup.model_validate_json(
            group.model_dump_json(warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("group",),
            code="finite_matrix_group.group_shape",
            message="group must satisfy its complete canonical carrier contract",
        ) from exc
    expected = construct_extension_general_linear_group(
        admitted.presentation, admitted.vector_axis
    )
    if admitted != expected:
        _domain_error(
            "authored_group_mismatch",
            "authored order and generators must equal the canonical full named group",
            ("group",),
        )
    return admitted


def _revalidate_special_linear_group(
    group: ExtensionFieldSpecialLinearGroup,
) -> ExtensionFieldSpecialLinearGroup:
    try:
        admitted = ExtensionFieldSpecialLinearGroup.model_validate_json(
            group.model_dump_json(warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("group",),
            code="finite_matrix_group.group_shape",
            message="group must satisfy its complete canonical carrier contract",
        ) from exc
    expected = construct_extension_special_linear_group(
        admitted.presentation, admitted.vector_axis
    )
    if admitted != expected:
        _domain_error(
            "authored_group_mismatch",
            "authored order and generators must equal the canonical full named group",
            ("group",),
        )
    return admitted


def _field_elements(
    context: Any, presentation: FiniteFieldPresentation
) -> tuple[Any, ...]:
    return tuple(
        _encoded_element(context, presentation, encoded)
        for encoded in range(presentation.order)
    )


def _projective_points(
    context: Any, presentation: FiniteFieldPresentation, axis: Axis
) -> tuple[ProjectivePoint, ...]:
    dimension = len(axis.labels)
    points: list[ProjectivePoint] = []
    zero = context.zero()
    one = context.one()
    elements = _field_elements(context, presentation) if dimension > 1 else ()
    for pivot in range(dimension):
        for suffix in product(elements, repeat=dimension - pivot - 1):
            values = (*((zero,) * pivot), one, *suffix)
            points.append(
                ProjectivePoint(
                    presentation=presentation,
                    axis=axis,
                    coordinates=tuple(
                        _field_element(presentation, value) for value in values
                    ),
                )
            )
    return tuple(points)


def _point_label(point: ProjectivePoint) -> str:
    coordinates = ";".join(
        ",".join(map(str, coordinate.coordinates)) for coordinate in point.coordinates
    )
    return f"[{coordinates}]"


def _projective_action(
    group: ExtensionFieldGeneralLinearGroup | ExtensionFieldSpecialLinearGroup,
) -> tuple[tuple[ProjectivePoint, ...], FinitePermutationAction]:
    points_bound = sum(
        group.presentation.order**power
        for power in range(len(group.vector_axis.labels))
    )
    if points_bound > MAX_PROJECTIVE_POINT_ACTION_SIZE:
        _resource_error(
            "projective_action_size_bound",
            "the complete projective-point axis exceeds the 50-point output envelope",
            ("group",),
        )
    _check_projective_action_output_size(group, points_bound)
    context = _field_context(group.presentation)
    points = _projective_points(context, group.presentation, group.vector_axis)
    point_keys = tuple(
        tuple(coordinate.coordinates for coordinate in point.coordinates)
        for point in points
    )
    positions = {key: index for index, key in enumerate(point_keys)}
    permutations: list[tuple[int, ...]] = []
    for generator in group.generators:
        image_positions: list[int] = []
        rows = tuple(
            tuple(context(list(entry.coordinates)) for entry in row)
            for row in generator.entries
        )
        for point in points:
            request_checkpoint("finite-field projective matrix action")
            vector = tuple(
                context(list(entry.coordinates)) for entry in point.coordinates
            )
            image = tuple(
                sum(
                    (
                        rows[row][column] * vector[column]
                        for column in range(len(vector))
                    ),
                    context.zero(),
                )
                for row in range(len(vector))
            )
            pivot = next((value for value in image if not value.is_zero()), None)
            if pivot is None:
                raise RuntimeError(
                    "an invertible matrix sent a projective point to zero"
                )
            inverse = pivot.inverse()
            key = tuple(
                tuple((value * inverse).to_list()) + (0,) * group.presentation.degree
                for value in image
            )
            key = tuple(value[: group.presentation.degree] for value in key)
            image_positions.append(positions[key])
        permutations.append(tuple(image_positions))
    action = FinitePermutationAction(
        domain=tuple(_point_label(point) for point in points),
        generators=tuple(permutations),
    )
    return points, action


def extension_general_linear_projective_action(
    group: ExtensionFieldGeneralLinearGroup,
) -> ExtensionFieldGeneralLinearProjectiveAction:
    admitted = _revalidate_general_linear_group(group)
    points, action = _projective_action(admitted)
    return ExtensionFieldGeneralLinearProjectiveAction(
        group=admitted, points=points, action=action
    )


def extension_special_linear_projective_action(
    group: ExtensionFieldSpecialLinearGroup,
) -> ExtensionFieldSpecialLinearProjectiveAction:
    admitted = _revalidate_special_linear_group(group)
    points, action = _projective_action(admitted)
    return ExtensionFieldSpecialLinearProjectiveAction(
        group=admitted, points=points, action=action
    )


__all__ = [
    "construct_extension_general_linear_group",
    "construct_extension_special_linear_group",
    "extension_general_linear_projective_action",
    "extension_special_linear_projective_action",
]
