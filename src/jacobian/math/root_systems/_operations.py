"""Exact root system operations."""

from __future__ import annotations

from jacobian.math.root_systems._cartan import (
    connected_components,
    simple_reflection,
)
from jacobian.math.root_systems._cartan import (
    positive_roots as enumerate_positive_roots,
)
from jacobian.math.root_systems._models import (
    CartanMatrixRequest,
    PositiveRootsResult,
    RootComponentData,
    RootSystemDataResult,
)


def compute_positive_roots(request: CartanMatrixRequest) -> PositiveRootsResult:
    """Compute all positive roots of a root system from its Cartan matrix."""
    n = len(request.matrix)
    all_positive = enumerate_positive_roots(request.matrix)

    return PositiveRootsResult(
        matrix=request.matrix,
        rank=n,
        positive_roots=all_positive,
        num_positive_roots=len(all_positive),
    )


def compute_simple_reflection(
    vector: list[int],
    simple_index: int,
    cartan: list[list[int]],
) -> list[int]:
    """Apply a simple reflection s_i to a root lattice vector."""
    return list(
        simple_reflection(tuple(vector), simple_index, tuple(map(tuple, cartan)))
    )


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

    return RootSystemDataResult(
        rank=n,
        cartan_matrix=request.matrix,
        positive_roots=roots,
        negative_roots=tuple(tuple(-value for value in root) for root in roots),
        simple_roots=simple_roots,
        num_positive_roots=len(roots),
        components=tuple(components),
    )
