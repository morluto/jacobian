"""Exact structural algorithms for bounded finite Cartan data."""

from __future__ import annotations

from collections import deque
from fractions import Fraction

from jacobian.math._exact_linear_algebra import symmetric_inertia

MAX_POSITIVE_ROOTS = 120  # E8 is maximal among crystallographic rank <= 8.
MAX_ROOT_COORDINATE = 6  # Maximal coefficient of an E8 positive root.


def connected_components(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(matrix)))
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            vertex = stack.pop()
            if vertex in component:
                continue
            component.add(vertex)
            remaining.discard(vertex)
            stack.extend(
                neighbor for neighbor in remaining if matrix[vertex][neighbor] != 0
            )
        components.append(tuple(sorted(component)))
    return tuple(components)


def positive_symmetrizer(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[Fraction, ...]:
    values: dict[int, Fraction] = {}
    for component in connected_components(matrix):
        values[component[0]] = Fraction(1)
        queue = deque([component[0]])
        while queue:
            left = queue.popleft()
            left_value = values[left]
            for right in component:
                if matrix[left][right] == 0:
                    continue
                candidate = left_value * Fraction(
                    matrix[left][right], matrix[right][left]
                )
                if right not in values:
                    values[right] = candidate
                    queue.append(right)
                elif values[right] != candidate:
                    raise ValueError("Cartan matrix is not symmetrizable")
    if len(values) != len(matrix):
        raise RuntimeError("Cartan symmetrizer did not cover every matrix index")
    return tuple(values[index] for index in range(len(matrix)))


def require_finite_type(matrix: tuple[tuple[int, ...], ...]) -> None:
    symmetrizer = positive_symmetrizer(matrix)
    symmetric = tuple(
        tuple(symmetrizer[row] * matrix[row][column] for column in range(len(matrix)))
        for row in range(len(matrix))
    )
    denominators = [entry.denominator for row in symmetric for entry in row]
    scale = 1
    from math import lcm

    for denominator in denominators:
        scale = lcm(scale, denominator)
    integral = tuple(tuple(int(entry * scale) for entry in row) for row in symmetric)
    positive, negative, zero = symmetric_inertia(integral)
    if (positive, negative, zero) != (len(matrix), 0, 0):
        raise ValueError("Cartan matrix must be of finite type")


def simple_reflection(
    root: tuple[int, ...], index: int, matrix: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    coefficient = sum(root[j] * matrix[index][j] for j in range(len(matrix)))
    reflected = list(root)
    reflected[index] -= coefficient
    return tuple(reflected)


def positive_roots(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    rank = len(matrix)
    initial = [tuple(int(i == j) for j in range(rank)) for i in range(rank)]
    seen = set(initial)
    queue = deque(initial)
    reflection_applications = 0
    while queue:
        root = queue.popleft()
        for index in range(rank):
            reflection_applications += 1
            if reflection_applications > MAX_POSITIVE_ROOTS * rank:
                raise RuntimeError("finite root closure exceeded its reflection bound")
            reflected = simple_reflection(root, index, matrix)
            if not all(value >= 0 for value in reflected) or not any(reflected):
                continue
            if max(reflected) > MAX_ROOT_COORDINATE:
                raise RuntimeError("finite root closure exceeded its coordinate bound")
            if reflected not in seen:
                seen.add(reflected)
                if len(seen) > MAX_POSITIVE_ROOTS:
                    raise RuntimeError(
                        "finite root closure exceeded its root-count bound"
                    )
                queue.append(reflected)
    return tuple(sorted(seen))


VALID_CARTAN_TYPE_RANKS: dict[str, tuple[int, ...]] = {
    "A": (1, 2, 3, 4, 5, 6, 7, 8),
    "B": (2, 3, 4, 5, 6, 7, 8),
    "C": (2, 3, 4, 5, 6, 7, 8),
    "D": (4, 5, 6, 7, 8),
    "E": (6, 7, 8),
    "F": (4,),
    "G": (2,),
}


def cartan_type_matrix(cartan_type: str, rank: int) -> tuple[tuple[int, ...], ...]:
    """Return the Cartan matrix for one finite Dynkin type and rank.

    The caller establishes that ``(cartan_type, rank)`` is a valid
    finite-type pair. Node labeling is the linear chain
    ``0 - 1 - ...`` except for the branch node documented below:

    - ``A_n``: the linear chain on ``n`` nodes.
    - ``B_n``: the chain with the last node short, so
      ``a[n-2][n-1] = -1`` and ``a[n-1][n-2] = -2``.
    - ``C_n``: the chain with the last node long, so
      ``a[n-2][n-1] = -2`` and ``a[n-1][n-2] = -1``.
    - ``D_n``: the chain ``0 - ... - (n-3)`` with two leaves
      ``n-2`` and ``n-1`` both attached to node ``n-3``.
    - ``E_6``, ``E_7``, ``E_8``: the chain ``0 - ...`` on the first
      ``rank - 1`` nodes with the last node attached to node 2.
    - ``F_4``: the chain with a double bond between nodes 1 and 2, so
      ``a[1][2] = -2`` and ``a[2][1] = -1``.
    - ``G_2``: ``((2, -3), (-1, 2))``.
    """
    if cartan_type == "A":
        return tuple(
            tuple(
                2 if column == row else (-1 if abs(column - row) == 1 else 0)
                for column in range(rank)
            )
            for row in range(rank)
        )
    if cartan_type in ("B", "C"):
        rows = [
            [
                2 if column == row else (-1 if abs(column - row) == 1 else 0)
                for column in range(rank)
            ]
            for row in range(rank)
        ]
        if cartan_type == "B":
            rows[rank - 2][rank - 1] = -1
            rows[rank - 1][rank - 2] = -2
        else:
            rows[rank - 2][rank - 1] = -2
            rows[rank - 1][rank - 2] = -1
        return tuple(tuple(row) for row in rows)
    if cartan_type == "D":
        rows = [[0] * rank for _ in range(rank)]
        for index in range(rank):
            rows[index][index] = 2
        for index in range(rank - 3):
            rows[index][index + 1] = -1
            rows[index + 1][index] = -1
        rows[rank - 3][rank - 2] = -1
        rows[rank - 2][rank - 3] = -1
        rows[rank - 3][rank - 1] = -1
        rows[rank - 1][rank - 3] = -1
        return tuple(tuple(row) for row in rows)
    if cartan_type == "E":
        rows = [[0] * rank for _ in range(rank)]
        for index in range(rank):
            rows[index][index] = 2
        for index in range(rank - 2):
            rows[index][index + 1] = -1
            rows[index + 1][index] = -1
        rows[2][rank - 1] = -1
        rows[rank - 1][2] = -1
        return tuple(tuple(row) for row in rows)
    if cartan_type == "F":
        return (
            (2, -1, 0, 0),
            (-1, 2, -2, 0),
            (0, -1, 2, -1),
            (0, 0, -1, 2),
        )
    return ((2, -3), (-1, 2))


def weyl_word_inversions(
    matrix: tuple[tuple[int, ...], ...], word: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Return the inversion set of a Weyl-group word in sorted order.

    The word ``(i_1, ..., i_k)`` acts on a root by applying the simple
    reflections left to right; its length is the number of positive
    roots sent to a negative root. The caller admits the finite-type
    matrix and the word bounds.
    """

    inversions: list[tuple[int, ...]] = []
    for root in positive_roots(matrix):
        image = root
        for index in word:
            image = simple_reflection(image, index, matrix)
        if all(coordinate <= 0 for coordinate in image):
            inversions.append(root)
    return tuple(sorted(inversions))


def weyl_longest_word(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Return a reduced word for the longest Weyl-group element.

    Greedy weak-order ascent: starting from the identity, prepend a
    simple reflection that sends a simple root to a positive root.
    A word lists its factors in application order, so prepending is
    right multiplication, and a right ascent raises the length by
    exactly one: the word is reduced by construction. The ascent stops
    only when every simple-root image is negative, which characterizes
    the unique longest element. Simple-root images are recomputed from
    the whole word at every step, so each ascent decision sees the
    current element exactly. The caller admits the finite-type matrix;
    termination follows because the length is bounded by the
    positive-root count.
    """

    rank = len(matrix)
    bound = len(positive_roots(matrix))
    word: list[int] = []
    for _ in range(bound + 1):
        ascent = None
        for index in range(rank):
            image = tuple(int(index == j) for j in range(rank))
            for factor in word:
                image = simple_reflection(image, factor, matrix)
            if any(image) and all(value >= 0 for value in image):
                ascent = index
                break
        if ascent is None:
            return tuple(word)
        word.insert(0, ascent)
    raise RuntimeError("weak-order ascent exceeded its length bound")


def weyl_word_descents(
    matrix: tuple[tuple[int, ...], ...], word: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the left and right descent sets of a Weyl-group word.

    A right descent sends its simple root negative under the word;
    a left descent does so under the reversed word, which acts as the
    inverse since every simple reflection is an involution. The caller
    admits the finite-type matrix and the word bounds.
    """

    rank = len(matrix)
    inverse = word[::-1]

    def is_negative(image: tuple[int, ...]) -> bool:
        return any(image) and all(coordinate <= 0 for coordinate in image)

    right: list[int] = []
    left: list[int] = []
    for index in range(rank):
        image = tuple(int(index == j) for j in range(rank))
        for factor in word:
            image = simple_reflection(image, factor, matrix)
        if is_negative(image):
            right.append(index)
        image = tuple(int(index == j) for j in range(rank))
        for factor in inverse:
            image = simple_reflection(image, factor, matrix)
        if is_negative(image):
            left.append(index)
    return tuple(left), tuple(right)


__all__ = [
    "VALID_CARTAN_TYPE_RANKS",
    "cartan_type_matrix",
    "connected_components",
    "positive_roots",
    "positive_symmetrizer",
    "require_finite_type",
    "simple_reflection",
    "weyl_longest_word",
    "weyl_word_descents",
    "weyl_word_inversions",
]
