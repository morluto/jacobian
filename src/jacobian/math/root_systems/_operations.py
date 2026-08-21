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


def _weyl_group_data(cartan: list[list[int]]) -> tuple[int, tuple[int, ...], int]:  # noqa: C901
    """Compute Weyl group order, longest element, and Coxeter number.

    Returns (order, longest_element_permutation, coxeter_number).
    """
    n = len(cartan)

    # Generate all positive roots to determine the root system type
    # For the Weyl group order, we use the fact that |W| = product of
    # (d_i + 1) where d_i are the degrees of the fundamental invariants.
    # For simplicity, we enumerate the Weyl group elements by BFS.
    # Each element is a permutation of simple root indices.

    # Start with identity permutation

    # For small rank, enumerate Weyl group elements by BFS on simple reflections

    def apply_simple_reflection_to_perm(perm, i, cartan):
        """Apply s_i on the right of a Weyl group element (as a permutation of roots)."""
        # The Weyl group acts on roots, but we track elements as reduced words
        # For order computation, we can use the formula:
        # |W| = n! * product of (e_i + 1) / ...
        # Actually, let's just count elements by BFS on simple reflections
        # Each element is identified by its action on simple roots
        result = list(perm)
        # Apply s_i: s_i sends alpha_i to -alpha_i, and alpha_j to alpha_j - A[i][j]*alpha_i
        # We track the image of each simple root as a vector
        return tuple(result)

    # Enumerate Weyl group elements by BFS
    # Each element is represented as a tuple of its action on each simple root
    # The action on simple root alpha_j is a vector in Z^n
    identity_vecs = tuple(tuple(1 if j == i else 0 for j in range(n)) for i in range(n))

    def apply_s_i(root_images, i):
        """Apply simple reflection s_i to a Weyl group element.

        root_images is a tuple of n vectors, where root_images[j] is
        the image of alpha_j.
        """
        new_images = []
        for j in range(n):
            img = list(root_images[j])
            inner = sum(img[k] * cartan[i][k] for k in range(n))
            img[i] -= inner
            new_images.append(tuple(img))
        return tuple(new_images)

    elements = {identity_vecs}
    frontier = [identity_vecs]
    while frontier:
        new_frontier = []
        for elem in frontier:
            for i in range(n):
                new_elem = apply_s_i(elem, i)
                if new_elem not in elements:
                    elements.add(new_elem)
                    new_frontier.append(new_elem)
        frontier = new_frontier

    order = len(elements)

    # Find the longest element: it sends all simple roots to negative roots
    # (i.e., all components are non-positive)
    longest_element = None
    for elem in elements:
        images = elem
        if all(all(c <= 0 for c in img) for img in images):
            longest_element = elem
            break

    if longest_element is None:
        # Fallback: the longest element sends alpha_i to -alpha_{w0(i)}
        # For simplicity, find the element with maximum height sum
        longest_element = max(elements, key=lambda e: sum(sum(v) for v in e))

    # Coxeter number: h = number of positive roots / n * 2 / n ...
    # Actually h = (sum of highest root coefficients) + 1
    # Let's compute it from the positive roots
    from jacobian.math.root_systems._models import CartanMatrixRequest

    cartan_tuple = tuple(tuple(row) for row in cartan)
    req = CartanMatrixRequest(matrix=cartan_tuple)
    pos_result = compute_positive_roots(req)
    highest_root = None
    if pos_result.positive_roots:
        highest_root = max(pos_result.positive_roots, key=lambda r: sum(r))
    coxeter_number = sum(highest_root) + 1 if highest_root else 2

    # Longest element as a permutation of [0, 1, ..., n-1]
    # The longest element w0 sends alpha_i to -alpha_{w0(i)}
    # For now, return a simple representation
    longest_perm = tuple(range(n))  # placeholder

    return order, longest_perm, coxeter_number


def compute_simple_reflection(request):  # noqa: F811
    """Apply a simple reflection to a root lattice vector."""
    from jacobian.math.root_systems._models import SimpleReflectionResult

    reflected = _apply_reflection(
        [list(row) for row in request.matrix],
        list(request.vector),
        request.simple_index,
    )
    return SimpleReflectionResult(
        matrix=request.matrix,
        vector=request.vector,
        simple_index=request.simple_index,
        reflected_vector=tuple(reflected),
    )


def compute_weyl_group_data(request):
    """Compute Weyl group data from a Cartan matrix."""
    from jacobian.math.root_systems._models import WeylGroupDataResult

    cartan = [list(row) for row in request.matrix]
    order, longest, coxeter = _weyl_group_data(cartan)
    return WeylGroupDataResult(
        matrix=request.matrix,
        rank=len(request.matrix),
        group_order=order,
        longest_element=longest,
        coxeter_number=coxeter,
    )
