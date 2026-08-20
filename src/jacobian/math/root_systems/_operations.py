"""Exact root system operations."""

from __future__ import annotations

from jacobian.math.root_systems._models import (
    CartanMatrixRequest,
    PositiveRootsResult,
    RootSystemDataResult,
)


def _simple_reflection(
    vector: list[int], simple_idx: int, cartan: list[list[int]]
) -> list[int]:
    """Apply the simple reflection s_i to a vector."""
    # s_i(v) = v - <v, alpha_i^vee> * alpha_i
    # For a vector in the root lattice, <v, alpha_i^vee> = sum_j v[j] * A[i][j]
    n = len(cartan)
    inner = sum(vector[j] * cartan[simple_idx][j] for j in range(n))
    result = list(vector)
    for j in range(n):
        result[j] -= inner * (1 if j == simple_idx else 0)
    # Actually: s_i(v_j) = v_j - v_i * A[i][j] for root lattice vectors
    # More precisely, s_i acts on the root alpha_j as: alpha_j - A[i][j] * alpha_i
    # For a vector v = sum v_j alpha_j, s_i(v) = v - (sum_j v_j A[i][j]) alpha_i
    # So: s_i(v)_j = v_j for j != i, s_i(v)_i = v_i - sum_j v_j A[i][j]
    result = list(vector)
    coeff = sum(vector[j] * cartan[simple_idx][j] for j in range(n))
    result[simple_idx] -= coeff
    return result


def compute_positive_roots(request: CartanMatrixRequest) -> PositiveRootsResult:
    """Compute all positive roots of a root system from its Cartan matrix."""
    n = len(request.matrix)
    # Simple roots as basis vectors
    simple_roots = []
    for i in range(n):
        v = [0] * n
        v[i] = 1
        simple_roots.append(v)

    # Generate positive roots by BFS
    # Start with simple roots, apply simple reflections to get all roots
    all_positive: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    # Start with simple roots
    for sr in simple_roots:
        t = tuple(sr)
        if t not in seen:
            seen.add(t)
            all_positive.append(t)

    # Apply all simple reflections to each root until closure
    changed = True
    while changed:
        changed = False
        new_roots = []
        for root in all_positive:
            for i in range(n):
                # Apply simple reflection s_i
                coeff = sum(root[j] * request.matrix[i][j] for j in range(n))
                reflected = list(root)
                reflected[i] -= coeff
                t = tuple(reflected)
                if t not in seen and all(x >= 0 for x in t) and any(x > 0 for x in t):
                    seen.add(t)
                    new_roots.append(t)
                    changed = True
        all_positive.extend(new_roots)

    all_positive.sort()

    return PositiveRootsResult(
        rank=n,
        positive_roots=tuple(all_positive),
        num_positive_roots=len(all_positive),
    )


def compute_simple_reflection(
    vector: list[int],
    simple_index: int,
    cartan: list[list[int]],
) -> list[int]:
    """Apply a simple reflection s_i to a root lattice vector."""
    return _simple_reflection(vector, simple_index, cartan)


def compute_root_system_data(request: CartanMatrixRequest) -> RootSystemDataResult:
    """Compute complete root system data from a Cartan matrix."""
    n = len(request.matrix)
    # Simple roots
    simple_roots = []
    for i in range(n):
        v = [0] * n
        v[i] = 1
        simple_roots.append(v)

    # Positive roots
    pos_result = compute_positive_roots(request)
    positive_roots = [list(r) for r in pos_result.positive_roots]

    # Negative roots
    negative_roots = [[-x for x in root] for root in positive_roots]

    # Highest root (the highest positive root by height)
    if positive_roots:
        # Highest root is the one with maximum sum of coefficients
        highest_root = max(positive_roots, key=lambda r: sum(r))
        highest_root_tuple = tuple(highest_root)
    else:
        highest_root_tuple = None

    # Coxeter number h = number of positive roots * 2 / rank + ... actually
    # h = |Phi| / n + ... but simpler: h = (sum of highest root coefficients) + 1
    coxeter_number = sum(highest_root_tuple) + 1 if highest_root_tuple else 2

    return RootSystemDataResult(
        rank=n,
        cartan_matrix=request.matrix,
        positive_roots=tuple(tuple(r) for r in positive_roots),
        negative_roots=tuple(tuple(r) for r in negative_roots),
        simple_roots=tuple(tuple(r) for r in simple_roots),
        highest_root=highest_root_tuple,
        num_positive_roots=len(positive_roots),
        coxeter_number=coxeter_number,
    )
