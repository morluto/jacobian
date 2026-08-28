"""Exact root system operations."""

from __future__ import annotations

from jacobian.math.groups.root_systems._cartan import (
    connected_components,
    simple_reflection,
)
from jacobian.math.groups.root_systems._cartan import (
    positive_roots as enumerate_positive_roots,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrixRequest,
    PositiveRootsResult,
    RootComponentData,
    RootSystemDataResult,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    WeylGroupOrderResult,
)

MAX_SIGNED_ROOT_ACTION_DEGREE = 240


def compute_positive_roots(request: CartanMatrixRequest) -> PositiveRootsResult:
    """Compute all positive roots of a root system from its Cartan matrix."""
    all_positive = enumerate_positive_roots(request.matrix)
    return PositiveRootsResult._from_kernel(request, all_positive)


def compute_root_system_data(request: CartanMatrixRequest) -> RootSystemDataResult:
    """Compute complete root system data from a Cartan matrix."""
    n = len(request.matrix)
    simple_roots = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    roots = enumerate_positive_roots(request.matrix)
    components: list[RootComponentData] = []
    for indices in connected_components(request.matrix):
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
        request,
        positive_roots=roots,
        negative_roots=tuple(tuple(-value for value in root) for root in roots),
        simple_roots=simple_roots,
        components=tuple(components),
    )


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
            root_index[simple_reflection(root, simple_index, matrix)] for root in roots
        )
        generators.append(Permutation(images))
    return int(PermutationGroup(*generators).order())


def compute_simple_reflection(
    request: SimpleReflectionRequest,
) -> SimpleReflectionResult:
    """Apply a simple reflection to a root lattice vector."""
    reflected = tuple(
        _apply_reflection(
            [list(row) for row in request.matrix],
            list(request.vector),
            request.simple_index,
        )
    )
    return SimpleReflectionResult._from_kernel(request, reflected)


def compute_weyl_group_order(request: CartanMatrixRequest) -> WeylGroupOrderResult:
    """Compute the exact order of a finite Weyl group without enumeration."""
    return WeylGroupOrderResult._from_kernel(request, _weyl_group_order(request.matrix))


__all__ = [
    "compute_positive_roots",
    "compute_root_system_data",
    "compute_simple_reflection",
    "compute_weyl_group_order",
]
