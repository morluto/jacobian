"""Exact root system operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._cartan import (
    connected_components,
)
from jacobian.math.groups.root_systems._cartan import (
    positive_roots as enumerate_positive_roots,
)
from jacobian.math.groups.root_systems._cartan import (
    simple_reflection as _simple_reflection_kernel,
)
from jacobian.math.groups.root_systems._models import (
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    MAX_REFLECTION_COORDINATE,
    PositiveRootsResult,
    RootComponentData,
    RootSystemDataResult,
    SimpleReflectionResult,
    WeylGroupOrderResult,
)

MAX_SIGNED_ROOT_ACTION_DEGREE = 2 * MAX_POSITIVE_ROOTS


def _admit_cartan_finite_type(matrix: tuple[tuple[int, ...], ...]) -> None:
    """Admit the finite-type Cartan domain before invoking a root kernel."""
    from jacobian.math.groups.root_systems._cartan import require_finite_type

    rank = len(matrix)
    if not 1 <= rank <= MAX_RANK:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.rank_out_of_range",
            message=f"rank must be between 1 and {MAX_RANK}",
        )
    if any(len(row) != rank for row in matrix):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.not_square",
            message="Cartan matrix must be square",
        )
    if any(matrix[index][index] != 2 for index in range(rank)):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.diagonal_entry",
            message="diagonal entries must be 2",
        )
    for row in range(rank):
        for column in range(rank):
            if row == column:
                continue
            entry = matrix[row][column]
            transpose_entry = matrix[column][row]
            if entry > 0:
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.positive_off_diagonal",
                    message="off-diagonal entries must be non-positive",
                )
            if entry * transpose_entry not in (0, 1, 2, 3):
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.off_diagonal_product",
                    message="off-diagonal product must be 0, 1, 2, or 3",
                )
            if (entry == 0) != (transpose_entry == 0):
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.zero_pattern",
                    message="generalized Cartan matrix requires a_ij == 0 iff a_ji == 0",
                )

    try:
        require_finite_type(matrix)
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.finite_type",
            message=str(error),
        ) from error


def root_system_data(matrix: tuple[tuple[int, ...], ...]) -> RootSystemDataResult:
    """Compute complete root-system data from a canonical Cartan matrix."""
    _admit_cartan_finite_type(matrix)
    n = len(matrix)
    simple_roots = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    roots = enumerate_positive_roots(matrix)
    components: list[RootComponentData] = []
    for indices in connected_components(matrix):
        component_roots = tuple(
            root
            for root in roots
            if any(root[index] for index in indices)
            and all(root[index] == 0 for index in range(n) if index not in indices)
        )
        highest = max(component_roots, key=lambda root: sum(root))
        marks = tuple(highest[index] for index in indices)
        components.append(
            RootComponentData(
                simple_root_indices=indices,
                positive_roots=component_roots,
                highest_root=highest,
                marks=marks,
                coxeter_number=sum(marks) + 1,
            )
        )

    return RootSystemDataResult._from_kernel(
        matrix,
        positive_roots=roots,
        negative_roots=tuple(tuple(-value for value in root) for root in roots),
        simple_roots=simple_roots,
        components=tuple(components),
    )


def positive_roots(matrix: tuple[tuple[int, ...], ...]) -> PositiveRootsResult:
    """Compute all positive roots of a root system from its Cartan matrix."""
    _admit_cartan_finite_type(matrix)
    all_positive = enumerate_positive_roots(matrix)
    return PositiveRootsResult._from_kernel(matrix, all_positive)


def _apply_reflection(
    cartan: list[list[int]], vector: list[int], simple_idx: int
) -> list[int]:
    """Apply simple reflection s_i to a root lattice vector.

    For a vector v = sum v_j alpha_j, s_i(v) = v - (sum_j v_j A[i][j]) alpha_i.
    """
    n = len(cartan)
    inner = sum(vector[j] * cartan[simple_idx][j] for j in range(n))
    result = list(vector)
    result[simple_idx] -= inner
    return result


def _signed_roots(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    positive = enumerate_positive_roots(matrix)
    roots = tuple(
        sorted((*positive, *(tuple(-value for value in root) for root in positive)))
    )
    if len(roots) > MAX_SIGNED_ROOT_ACTION_DEGREE:
        raise ValueError("signed root action exceeds the bounded degree")
    return roots


def _weyl_group_order(matrix: tuple[tuple[int, ...], ...]) -> int:
    """Return |W| through its faithful action on the complete signed root set."""
    from sympy.combinatorics import Permutation, PermutationGroup

    roots = _signed_roots(matrix)
    root_index = {root: index for index, root in enumerate(roots)}
    generators = []
    for simple_index in range(len(matrix)):
        images = tuple(
            root_index[_simple_reflection_kernel(root, simple_index, matrix)]
            for root in roots
        )
        generators.append(Permutation(images))
    return int(PermutationGroup(*generators).order())


def simple_reflection(
    matrix: tuple[tuple[int, ...], ...],
    vector: tuple[int, ...],
    simple_index: int,
) -> SimpleReflectionResult:
    """Apply a simple reflection to a root lattice vector."""
    _admit_cartan_finite_type(matrix)
    rank = len(matrix)
    if type(simple_index) is not int or simple_index < 0 or simple_index >= rank:
        raise OperationDomainValidationError(
            location=("simple_index",),
            code="root_system.simple_index_out_of_range",
            message="simple_index out of range",
        )
    if len(vector) != rank:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.vector_length_mismatch",
            message="vector length must match rank",
        )
    if any(
        type(coordinate) is not int or abs(coordinate) > MAX_REFLECTION_COORDINATE
        for coordinate in vector
    ):
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.vector_coordinate_out_of_range",
            message="vector coordinates exceed the bounded root-lattice axis",
        )
    reflected = tuple(
        _apply_reflection(
            [list(row) for row in matrix],
            list(vector),
            simple_index,
        )
    )
    return SimpleReflectionResult._from_kernel(matrix, vector, simple_index, reflected)


def weyl_group_order(matrix: tuple[tuple[int, ...], ...]) -> WeylGroupOrderResult:
    """Compute the exact order of a finite Weyl group without enumeration."""
    _admit_cartan_finite_type(matrix)
    return WeylGroupOrderResult._from_kernel(matrix, _weyl_group_order(matrix))


__all__ = [
    "positive_roots",
    "root_system_data",
    "simple_reflection",
    "weyl_group_order",
]
