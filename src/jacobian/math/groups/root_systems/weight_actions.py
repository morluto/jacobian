"""Exact Weyl actions on parent-bound weight-lattice values."""

from __future__ import annotations

from fractions import Fraction
from math import factorial

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    MAX_ROOT_COORDINATE,
    CartanMatrix,
    FiniteCartanDatum,
    WeightLatticeVector,
    WeylElement,
    WeylElementWeightActionRequest,
)
from jacobian.math.groups.root_systems.operations import (
    _admit_lattice_coordinates,
    _admit_weyl_element_value,
    _as_cartan,
    _canonical_lattice_vector,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_WEYL_WEIGHT_ACTION_WORK = 150_000
MAX_WEYL_WEIGHT_ACTION_ALLOCATION_CELLS = 12 * MAX_RANK**2 + 4 * MAX_RANK
MAX_WEYL_WEIGHT_ACTION_RATIONAL_BITS = 128


def _integer_matrix_children_are_bounded(value: IntegerMatrix) -> bool:
    try:
        domain = value.domain
        rows = value.entries
        row_count, column_count = value.row_count, value.column_count
    except AttributeError:
        return False
    if (
        domain != "ZZ"
        or type(row_count) is not int
        or type(column_count) is not int
        or not isinstance(rows, tuple)
        or not 1 <= len(rows) <= MAX_RANK
        or row_count != len(rows)
        or column_count != len(rows)
    ):
        return False
    return all(
        isinstance(row, tuple)
        and len(row) == len(rows)
        and all(type(value) is int for value in row)
        for row in rows
    )


def _cartan_children_are_bounded(value: CartanMatrix) -> bool:
    try:
        matrix, axis = value.matrix, value.simple_root_axis
    except AttributeError:
        return False
    return (
        isinstance(matrix, IntegerMatrix)
        and _integer_matrix_children_are_bounded(matrix)
        and isinstance(axis, tuple)
        and axis == tuple(range(matrix.row_count))
    )


def _matrix_inverse(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    rank = len(matrix)
    augmented = [
        [Fraction(value) for value in row]
        + [Fraction(int(row_index == column)) for column in range(rank)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(rank):
        pivot = next(
            (row for row in range(column, rank) if augmented[row][column]), None
        )
        if pivot is None:
            raise OperationDomainValidationError(
                location=("element", "matrix"),
                code="root_system.weight_action_singular_cartan",
                message="the finite Cartan matrix must be invertible",
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(rank):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [
                    left - scale * right
                    for left, right in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return tuple(tuple(row[rank:]) for row in augmented)


def _multiply(
    left: tuple[tuple[int, ...] | tuple[Fraction, ...], ...],
    right: tuple[tuple[int, ...] | tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    rank = len(left)
    return tuple(
        tuple(
            sum(
                (
                    Fraction(left[row][inner]) * right[inner][column]
                    for inner in range(rank)
                ),
                Fraction(0),
            )
            for column in range(rank)
        )
        for row in range(rank)
    )


def _weight_action_matrix(
    cartan: tuple[tuple[int, ...], ...],
    root_action: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Conjugate root action through Q -> P, whose matrix is the Cartan matrix."""
    inverse = _matrix_inverse(cartan)
    cartan_root_action = _multiply(cartan, root_action)
    rational = _multiply(cartan_root_action, inverse)
    if any(value.denominator != 1 for row in rational for value in row):
        raise OperationDomainValidationError(
            location=("element", "root_action"),
            code="root_system.invalid_weight_action",
            message="the root action must preserve the integral weight lattice",
        )
    return tuple(tuple(int(value) for value in row) for row in rational)


def _weight_action_preflight(rank: int) -> tuple[int, int]:
    """Bound matrix cells and exact rational digits before matrix expansion.

    Finite Cartan entries have absolute value at most 3 and Weyl root-action
    entries at most 6. Reduced Gauss-Jordan entries are ratios of minors, so
    Cramer's determinant bound controls each stored Fraction. A pivot-row scale
    and row update can form raw cross-products of those fractions before
    reduction; four determinant bounds plus rank slack cover those temporary
    numerators and denominators as well as the final matrix products.
    """
    allocation_cells = 12 * rank**2 + 4 * rank
    determinant_bound = factorial(rank) * 3**rank
    cofactor_bound = factorial(rank - 1) * 3 ** (rank - 1)
    numerator_bound = 3 * MAX_ROOT_COORDINATE * rank**2 * cofactor_bound
    rational_bits = max(
        4 * determinant_bound.bit_length() + 2 * rank.bit_length() + 4,
        numerator_bound.bit_length(),
    )
    return allocation_cells, rational_bits


def _request_inputs(
    request: WeylElementWeightActionRequest,
) -> tuple[WeylElement, WeightLatticeVector]:
    if not isinstance(request, WeylElementWeightActionRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="root_system.weyl_weight_action_request_type",
            message="request must bind a Weyl element and weight-lattice vector",
        )
    try:
        element, weight = request.element, request.weight
    except AttributeError as error:
        raise OperationDomainValidationError(
            location=("request",),
            code="root_system.weyl_weight_action_request_shape",
            message="request must contain both the Weyl element and weight value",
        ) from error
    return element, weight


def _weyl_parent(element: WeylElement) -> CartanMatrix:
    if not isinstance(element, WeylElement):
        raise OperationDomainValidationError(
            location=("element",),
            code="root_system.invalid_weyl_element",
            message="element must be a typed finite Weyl-group value",
        )
    try:
        element_matrix, element_action = element.matrix, element.root_action
    except AttributeError as error:
        raise OperationDomainValidationError(
            location=("element",),
            code="root_system.invalid_weyl_element",
            message="element must contain a typed Cartan parent and integer root action",
        ) from error
    if (
        not isinstance(element_matrix, CartanMatrix)
        or not _cartan_children_are_bounded(element_matrix)
        or not isinstance(element_action, IntegerMatrix)
        or not _integer_matrix_children_are_bounded(element_action)
    ):
        raise OperationDomainValidationError(
            location=("element",),
            code="root_system.invalid_weyl_element",
            message="element must contain a typed Cartan parent and integer root action",
        )
    try:
        cartan = _as_cartan(element_matrix)
    except OperationDomainValidationError:
        raise
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise OperationDomainValidationError(
            location=("element", "matrix"),
            code="root_system.invalid_weyl_element_parent",
            message="the Weyl element must retain a canonical finite Cartan matrix",
        ) from error
    return cartan


def _admitted_root_action(element: WeylElement) -> tuple[tuple[int, ...], ...]:
    try:
        return _admit_weyl_element_value(element)
    except OperationDomainValidationError:
        raise
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise OperationDomainValidationError(
            location=("element", "root_action"),
            code="root_system.invalid_weyl_action_shape",
            message="the supplied Weyl element contains malformed action data",
        ) from error


def _canonical_weight(
    weight: WeightLatticeVector,
) -> tuple[FiniteCartanDatum, tuple[int, ...]]:
    if not isinstance(weight, WeightLatticeVector):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.invalid_weight_lattice_value",
            message="weight must contain a typed finite Cartan datum",
        )
    try:
        datum_value = weight.datum
        weight_cartan = datum_value.cartan_matrix
        root_to_weight = datum_value.root_to_weight
        coroot_to_coweight = datum_value.coroot_to_coweight
        symmetrizer = datum_value.symmetrizer
        coordinates_value = weight.coordinates
    except AttributeError as error:
        raise OperationDomainValidationError(
            location=("weight", "datum"),
            code="root_system.invalid_weight_lattice_shape",
            message="weight datum and coordinates must retain their typed axes",
        ) from error
    if (
        not isinstance(datum_value, FiniteCartanDatum)
        or not isinstance(weight_cartan, CartanMatrix)
        or not _cartan_children_are_bounded(weight_cartan)
        or not isinstance(root_to_weight, IntegerMatrix)
        or not _integer_matrix_children_are_bounded(root_to_weight)
        or not isinstance(coroot_to_coweight, IntegerMatrix)
        or not _integer_matrix_children_are_bounded(coroot_to_coweight)
        or not isinstance(symmetrizer, tuple)
        or len(symmetrizer) != len(weight_cartan)
        or any(not isinstance(value, CanonicalRational) for value in symmetrizer)
        or not isinstance(coordinates_value, tuple)
        or len(coordinates_value) != len(weight_cartan)
        or any(type(value) is not int for value in coordinates_value)
    ):
        raise OperationDomainValidationError(
            location=("weight", "datum"),
            code="root_system.invalid_weight_lattice_shape",
            message="weight datum matrices, symmetrizer, and coordinates must retain their typed axes",
        )
    try:
        return _canonical_lattice_vector(weight, WeightLatticeVector, output_bound=True)
    except OperationDomainValidationError:
        raise
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.invalid_weight_lattice_value",
            message="weight must retain a canonical finite Cartan datum and bounded coordinates",
        ) from error


def weyl_element_act_on_weight(
    request: WeylElementWeightActionRequest,
) -> WeightLatticeVector:
    """Return the exact image in the same ordered fundamental-weight lattice."""
    element, weight = _request_inputs(request)
    cartan = _weyl_parent(element)
    rows = cartan.entries
    # Re-admission closes at most 120 positive roots, checks each image, then
    # performs at most one inverse and matrix product per length descent. Count
    # root closure, root-image checks, every descent, datum canonicalization,
    # and the induced weight-matrix work at the maximum supported rank.
    work = (
        2 * MAX_POSITIVE_ROOTS * MAX_RANK**2
        + 2 * (MAX_POSITIVE_ROOTS + 1) * MAX_RANK**3
        + 6 * MAX_RANK**3
        + 20 * MAX_RANK**2
    )
    if work > MAX_WEYL_WEIGHT_ACTION_WORK:
        raise OperationResourceAdmissionError(
            location=("element",),
            code="root_system.weyl_weight_action_work_bound",
            message="exact weight-action validation and matrix work exceed the admitted bound",
        )
    rank = len(cartan)
    allocation_cells, rational_bits = _weight_action_preflight(rank)
    if allocation_cells > MAX_WEYL_WEIGHT_ACTION_ALLOCATION_CELLS:
        raise OperationResourceAdmissionError(
            location=("weight", "datum"),
            code="root_system.weyl_weight_action_allocation_bound",
            message="the exact weight-action matrices exceed the admitted cell bound",
        )
    if rational_bits > MAX_WEYL_WEIGHT_ACTION_RATIONAL_BITS:
        raise OperationResourceAdmissionError(
            location=("element", "matrix"),
            code="root_system.weyl_weight_action_digit_bound",
            message="exact Cartan inverse intermediates exceed the admitted integer-digit bound",
        )
    # Rebuild the complete nested value through its validated model before the
    # admission helper dereferences fields (model_construct is caller-accessible).
    try:
        element = WeylElement.model_validate(
            {"matrix": element.matrix, "root_action": element.root_action}
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise OperationDomainValidationError(
            location=("element",),
            code="root_system.invalid_weyl_element",
            message="the supplied Weyl element must have canonical nested values",
        ) from error
    root_action = _admitted_root_action(element)
    datum, coordinates = _canonical_weight(weight)
    if datum.cartan_matrix != cartan:
        raise OperationDomainValidationError(
            location=("weight", "datum"),
            code="root_system.weyl_weight_parent_mismatch",
            message="the Weyl element and weight must use the same ordered Cartan datum",
        )
    action = _weight_action_matrix(rows, root_action)
    # Rank is at most eight and both operands have admitted bounds; calculate
    # the exact bounded image so cancellations are not mistaken for growth.
    image = tuple(
        sum(action[row][column] * coordinates[column] for column in range(rank))
        for row in range(rank)
    )
    _admit_lattice_coordinates(image, rank, output=True)
    return WeightLatticeVector.model_construct(datum=datum, coordinates=image)
