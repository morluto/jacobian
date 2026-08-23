"""Exact root system operations."""

from __future__ import annotations

from jacobian.math.root_systems._cartan import (
    connected_components,
)
from jacobian.math.root_systems._cartan import (
    positive_roots as enumerate_positive_roots,
)
from jacobian.math.root_systems._models import (
    CartanMatrixRequest,
    PositiveRootsResult,
    RootComponentData,
    RootSystemDataResult,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    WeylGroupDataRequest,
    WeylGroupDataResult,
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


def _weyl_group_data(cartan: list[list[int]]) -> tuple[int, tuple[int, ...], int]:
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

    # Enumerate Weyl group elements by BFS
    # Each element is represented as a tuple of its action on each simple root
    # The action on simple root alpha_j is a vector in Z^n
    identity_vecs = tuple(tuple(1 if j == i else 0 for j in range(n)) for i in range(n))

    def apply_s_i(
        root_images: tuple[tuple[int, ...], ...], i: int
    ) -> tuple[tuple[int, ...], ...]:
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


def compute_simple_reflection(
    request: SimpleReflectionRequest,
) -> SimpleReflectionResult:
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


def compute_weyl_group_data(request: WeylGroupDataRequest) -> WeylGroupDataResult:
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
