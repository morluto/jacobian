"""Exact Weyl actions on parent-bound root-lattice values."""

from __future__ import annotations

from math import factorial

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    MAX_LATTICE_OUTPUT_COORDINATE_BITS,
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    MAX_ROOT_COORDINATE,
    CartanMatrix,
    FiniteCartanDatum,
    RootLatticeVector,
    WeylElement,
)
from jacobian.math.groups.root_systems.operations import (
    _admit_weyl_element_value,
    _cartan_datum_from_admitted,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_WEYL_ROOT_ACTION_WORK = 150_000
MAX_WEYL_ROOT_ACTION_RATIONAL_BITS = 160
MAX_WEYL_ROOT_ACTION_ALLOCATION_CELLS = (
    8 * MAX_POSITIVE_ROOTS * MAX_RANK + 32 * MAX_RANK**2 + 32 * MAX_RANK
)
MAX_WEYL_ROOT_ACTION_RESULT_CELLS = 6 * MAX_RANK**2 + MAX_RANK
MAX_WEYL_ROOT_ACTION_INTERMEDIATE_BITS = (
    MAX_LATTICE_OUTPUT_COORDINATE_BITS
    + (MAX_RANK * MAX_ROOT_COORDINATE - 1).bit_length()
)


def _canonical_element(element: WeylElement) -> WeylElement:
    if type(element) is not WeylElement:
        raise OperationDomainValidationError(
            location=("element",),
            code="root_system.invalid_weyl_element",
            message="element must be a typed finite Weyl-group value",
        )
    try:
        matrix = element.matrix
        root_action = element.root_action
        if type(matrix) is not CartanMatrix or type(root_action) is not IntegerMatrix:
            raise ValueError("invalid Weyl matrix value types")
        cartan_matrix = matrix.matrix
        if type(cartan_matrix) is not IntegerMatrix:
            raise ValueError("invalid Cartan matrix value type")
        cartan_entries = cartan_matrix.entries
        action_entries = root_action.entries
        rank = cartan_matrix.row_count
        if (
            cartan_matrix.domain != "ZZ"
            or type(rank) is not int
            or not 1 <= rank <= MAX_RANK
            or type(cartan_matrix.column_count) is not int
            or cartan_matrix.column_count != rank
            or matrix.simple_root_axis != tuple(range(rank))
            or type(cartan_entries) is not tuple
            or len(cartan_entries) != rank
            or any(
                type(row) is not tuple
                or len(row) != rank
                or any(type(value) is not int or not -3 <= value <= 2 for value in row)
                for row in cartan_entries
            )
            or root_action.domain != "ZZ"
            or type(root_action.row_count) is not int
            or type(root_action.column_count) is not int
            or root_action.row_count != rank
            or root_action.column_count != rank
            or type(action_entries) is not tuple
            or len(action_entries) != rank
            or any(
                type(row) is not tuple
                or len(row) != rank
                or any(
                    type(value) is not int or abs(value) > MAX_ROOT_COORDINATE
                    for value in row
                )
                for row in action_entries
            )
        ):
            raise ValueError("invalid bounded Weyl action structure")
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("element",),
            code="root_system.invalid_weyl_element",
            message="element must retain bounded canonical Cartan and integer action matrices",
        ) from error
    return element


def _canonical_vector(
    vector: RootLatticeVector,
    element: WeylElement,
    canonical_datum: FiniteCartanDatum,
) -> tuple[int, ...]:
    if type(vector) is not RootLatticeVector:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.lattice_vector_type",
            message="the action requires a root-lattice vector",
        )
    try:
        datum = vector.datum
        raw_coordinates = vector.coordinates
    except AttributeError as error:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.invalid_lattice_vector",
            message="vector must contain root-lattice coordinates",
        ) from error
    if (
        not isinstance(raw_coordinates, tuple)
        or len(raw_coordinates) != len(element.matrix)
        or any(type(value) is not int for value in raw_coordinates)
    ):
        raise OperationDomainValidationError(
            location=("vector", "coordinates"),
            code="root_system.invalid_lattice_vector_coordinates",
            message="root-lattice coordinates must be a bounded integer tuple",
        )
    if any(
        abs(value).bit_length() > MAX_LATTICE_OUTPUT_COORDINATE_BITS
        for value in raw_coordinates
    ):
        raise OperationResourceAdmissionError(
            location=("vector", "coordinates"),
            code="root_system.lattice_coordinates_over_envelope",
            message=(
                "root-lattice coordinates must fit the canonical vector carrier "
                f"bound of {MAX_LATTICE_OUTPUT_COORDINATE_BITS} bits"
            ),
        )
    if type(datum) is not FiniteCartanDatum:
        raise OperationDomainValidationError(
            location=("vector", "datum"),
            code="root_system.invalid_lattice_vector_datum",
            message="vector must retain a typed finite Cartan datum",
        )
    try:
        datum_cartan = datum.cartan_matrix
    except AttributeError as error:
        raise OperationDomainValidationError(
            location=("vector", "datum"),
            code="root_system.invalid_lattice_vector_datum",
            message="vector datum must retain its Cartan matrix",
        ) from error
    # CartanMatrix equality deliberately ignores its axis annotation. Validate
    # the complete nested parent before comparing it with the action's parent.
    try:
        datum_rank = len(datum_cartan)
        symmetrizer = datum.symmetrizer
        if (
            not isinstance(symmetrizer, tuple)
            or any(
                type(value.num) is not int or type(value.den) is not int
                for value in symmetrizer
                if type(value) is CanonicalRational
            )
            or any(type(value) is not CanonicalRational for value in symmetrizer)
            or type(datum_cartan) is not CartanMatrix
            or datum_cartan.simple_root_axis != tuple(range(datum_rank))
            or type(datum_cartan.matrix) is not IntegerMatrix
            or datum_cartan.matrix.domain != "ZZ"
            or datum_cartan.matrix.row_count != datum_rank
            or datum_cartan.matrix.column_count != datum_rank
        ):
            raise ValueError("noncanonical Cartan matrix axis or shape")
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("vector", "datum", "cartan_matrix"),
            code="root_system.invalid_lattice_vector_datum",
            message="vector datum must retain a canonical ordered Cartan axis",
        ) from error
    if datum_cartan != element.matrix:
        raise OperationDomainValidationError(
            location=("vector", "datum"),
            code="root_system.weyl_root_parent_mismatch",
            message="the Weyl element and root vector must use the same ordered Cartan datum",
        )
    try:
        canonical = datum == canonical_datum
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("vector", "datum"),
            code="root_system.invalid_lattice_vector_datum",
            message="vector datum must contain canonical lattice maps and symmetrizer",
        ) from error
    if not canonical:
        raise OperationDomainValidationError(
            location=("vector", "datum"),
            code="root_system.invalid_lattice_vector_datum",
            message="the root vector's lattice maps and symmetrizer must be canonical",
        )

    return tuple(vector.coordinates)


def weyl_element_act_on_root(
    element_value: WeylElement,
    vector_value: RootLatticeVector,
) -> RootLatticeVector:
    """Apply one admitted Weyl action in the same ordered root lattice."""
    element = _canonical_element(element_value)

    # Membership checking re-closes at most 120 positive roots and performs
    # at most one inverse/reflection update per positive-root descent. The
    # root action and this vector product are rank-squared exact arithmetic.
    rank = len(element.matrix)
    work_bound = (
        2 * MAX_POSITIVE_ROOTS * MAX_RANK**2
        + 2 * (MAX_POSITIVE_ROOTS + 1) * MAX_RANK**3
        + 7 * MAX_RANK**3
        + 24 * MAX_RANK**2
    )
    allocation_cells = 8 * MAX_POSITIVE_ROOTS * rank + 32 * rank**2 + 32 * rank
    result_cells = 6 * rank**2 + rank
    if work_bound > MAX_WEYL_ROOT_ACTION_WORK:
        raise OperationResourceAdmissionError(
            location=("element",),
            code="root_system.weyl_root_action_work_bound",
            message="Weyl-element membership and root action exceed the admitted work bound",
        )
    if allocation_cells > MAX_WEYL_ROOT_ACTION_ALLOCATION_CELLS:
        raise OperationResourceAdmissionError(
            location=("element",),
            code="root_system.weyl_root_action_allocation_bound",
            message="Weyl-element validation exceeds the admitted exact-cell bound",
        )
    if result_cells > MAX_WEYL_ROOT_ACTION_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("vector",),
            code="root_system.weyl_root_action_output_bound",
            message="the root-lattice result exceeds its admitted exact-cell bound",
        )
    determinant_bound = factorial(rank) * MAX_ROOT_COORDINATE**rank
    rational_bits = 4 * determinant_bound.bit_length() + 2 * rank.bit_length() + 4
    if rational_bits > MAX_WEYL_ROOT_ACTION_RATIONAL_BITS:
        raise OperationResourceAdmissionError(
            location=("element", "root_action"),
            code="root_system.weyl_root_action_rational_bound",
            message="exact Weyl-membership inverse intermediates exceed their rational-digit bound",
        )
    action = _admit_weyl_element_value(element)
    canonical_datum = _cartan_datum_from_admitted(element.matrix)
    coordinates = _canonical_vector(vector_value, element, canonical_datum)
    coordinate_bits = max((abs(value).bit_length() for value in coordinates), default=0)
    maximum_row_l1 = max(
        (sum(abs(value) for value in row) for row in action), default=0
    )
    intermediate_bits = (
        coordinate_bits + (maximum_row_l1 - 1).bit_length()
        if coordinate_bits and maximum_row_l1
        else 0
    )
    if intermediate_bits > MAX_WEYL_ROOT_ACTION_INTERMEDIATE_BITS:
        raise OperationResourceAdmissionError(
            location=("vector", "coordinates"),
            code="root_system.weyl_root_action_digit_bound",
            message="the exact root-action intermediates exceed their admitted digit bound",
        )
    image = tuple(
        sum(action[row][column] * coordinates[column] for column in range(rank))
        for row in range(rank)
    )
    if any(
        abs(value).bit_length() > MAX_LATTICE_OUTPUT_COORDINATE_BITS for value in image
    ):
        raise OperationResourceAdmissionError(
            location=("vector", "coordinates"),
            code="root_system.weyl_root_action_image_bound",
            message="the exact root-lattice image exceeds the canonical coordinate bound",
        )
    return RootLatticeVector.model_construct(
        datum=canonical_datum,
        coordinates=image,
    )
