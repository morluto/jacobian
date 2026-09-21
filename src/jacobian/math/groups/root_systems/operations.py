"""Exact root system operations."""

from __future__ import annotations

from typing import cast

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._cartan import (
    VALID_CARTAN_TYPE_RANKS,
    cartan_type_matrix,
    connected_components,
)
from jacobian.math.groups.root_systems._cartan import (
    positive_roots as enumerate_positive_roots,
)
from jacobian.math.groups.root_systems._cartan import (
    simple_reflection as _simple_reflection_kernel,
)
from jacobian.math.groups.root_systems._cartan import (
    weyl_longest_word as _weyl_longest_word_kernel,
)
from jacobian.math.groups.root_systems._cartan import (
    weyl_word_descents as _weyl_word_descents_kernel,
)
from jacobian.math.groups.root_systems._cartan import (
    weyl_word_inversions as _weyl_word_inversions_kernel,
)
from jacobian.math.groups.root_systems._models import (
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    MAX_REFLECTION_REPRESENTABLE,
    MAX_WEYL_WORD_LENGTH,
    CartanMatrix,
    CartanType,
    CartanTypeResult,
    PositiveRootsResult,
    RootComponentData,
    RootSystemDataResult,
    SimpleReflectionResult,
    SimpleReflectionsResult,
    WeylDescentsResult,
    WeylElementLengthResult,
    WeylGroupOrderResult,
    WeylLongestElementResult,
)

MAX_SIGNED_ROOT_ACTION_DEGREE = 2 * MAX_POSITIVE_ROOTS


def _as_cartan(matrix: CartanMatrix | tuple[tuple[int, ...], ...]) -> CartanMatrix:
    return (
        matrix
        if isinstance(matrix, CartanMatrix)
        else CartanMatrix.model_validate(matrix)
    )


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


def cartan_matrix_from_type(
    cartan_type: str,
    rank: int,
) -> CartanTypeResult:
    """Build the Cartan matrix of a finite Dynkin type and rank.

    The admitted pairs are ``A_n`` (``n >= 1``), ``B_n``/``C_n``
    (``n >= 2``), ``D_n`` (``n >= 4``), ``E_6``/``E_7``/``E_8``,
    ``F_4``, and ``G_2`` with rank at most 8. The kernel asserts its
    own output is finite-type before trusted result construction.
    """
    if type(cartan_type) is not str or cartan_type not in VALID_CARTAN_TYPE_RANKS:
        raise OperationDomainValidationError(
            location=("cartan_type",),
            code="root_system.unknown_cartan_type",
            message="cartan_type must be one of A, B, C, D, E, F, G",
        )
    if type(rank) is not int or rank not in VALID_CARTAN_TYPE_RANKS[cartan_type]:
        raise OperationDomainValidationError(
            location=("rank",),
            code="root_system.invalid_cartan_type_rank",
            message=(
                f"rank {rank!r} is not a finite type for "
                f"{cartan_type}; admitted ranks are "
                f"{list(VALID_CARTAN_TYPE_RANKS[cartan_type])}"
            ),
        )
    rows = cartan_type_matrix(cartan_type, rank)
    _admit_cartan_finite_type(rows)
    cartan = CartanMatrix.model_validate(rows)
    return CartanTypeResult._from_kernel(cast("CartanType", cartan_type), rank, cartan)


def root_system_data(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> RootSystemDataResult:
    """Compute complete root-system data from a canonical Cartan matrix."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    n = len(rows)
    simple_roots = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    roots = enumerate_positive_roots(rows)
    components: list[RootComponentData] = []
    for indices in connected_components(rows):
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
        cartan,
        positive_roots=roots,
        negative_roots=tuple(tuple(-value for value in root) for root in roots),
        simple_roots=simple_roots,
        components=tuple(components),
    )


def positive_roots(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> PositiveRootsResult:
    """Compute all positive roots of a root system from its Cartan matrix."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    all_positive = enumerate_positive_roots(rows)
    return PositiveRootsResult._from_kernel(cartan, all_positive)


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
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    vector: tuple[int, ...],
    simple_index: int,
) -> SimpleReflectionResult:
    """Apply a simple reflection to a root lattice vector."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
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
        type(coordinate) is not int or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
        for coordinate in vector
    ):
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.vector_coordinate_out_of_range",
            message="vector coordinates exceed the bounded root-lattice axis",
        )
    reflected = tuple(
        _apply_reflection(
            [list(row) for row in rows],
            list(vector),
            simple_index,
        )
    )
    if any(abs(coordinate) > MAX_REFLECTION_REPRESENTABLE for coordinate in reflected):
        raise OperationDomainValidationError(
            location=("reflected_vector",),
            code="root_system.reflected_vector_out_of_range",
            message="reflection image exceeds the interoperable root-lattice axis",
        )
    return SimpleReflectionResult._from_kernel(cartan, vector, simple_index, reflected)


def _admit_weyl_word(word: tuple[int, ...] | list[int], rank: int) -> tuple[int, ...]:
    """Admit a bounded Weyl word against a Cartan rank."""
    if isinstance(word, list):
        word = tuple(word)
    if (
        not isinstance(word, tuple)
        or len(word) > MAX_WEYL_WORD_LENGTH
        or any(type(index) is not int or index < 0 or index >= rank for index in word)
    ):
        raise OperationDomainValidationError(
            location=("word",),
            code="root_system.invalid_weyl_word",
            message=(
                "word must hold at most "
                f"{MAX_WEYL_WORD_LENGTH} integer simple-reflection indices "
                f"below the Cartan rank {rank}"
            ),
        )
    return word


def weyl_element_length(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
) -> WeylElementLengthResult:
    """Compute the length, reducedness, and inversion set of a Weyl word.

    The word lists simple-reflection indices applied left to right. The
    length is the number of positive roots the word sends negative; the
    word is reduced exactly when that count equals its factor count.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    word = _admit_weyl_word(word, len(rows))
    inversions = _weyl_word_inversions_kernel(rows, word)
    return WeylElementLengthResult._from_kernel(
        cartan, word, len(inversions), len(inversions) == len(word), inversions
    )


def weyl_element_descents(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
) -> WeylDescentsResult:
    """Compute the left and right descent sets of a Weyl-group word.

    A word lists its factors in application order, so prepending is
    right multiplication and appending is left multiplication: a right
    descent lowers the length when prepended, a left descent when
    appended. Both sets are carried as sorted simple-root index sets.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    admitted = _admit_weyl_word(word, len(rows))
    left_descents, right_descents = _weyl_word_descents_kernel(rows, admitted)
    return WeylDescentsResult._from_kernel(
        cartan, admitted, left_descents, right_descents
    )


def weyl_longest_element(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> WeylLongestElementResult:
    """Compute a reduced word for the longest Weyl-group element.

    Greedy weak-order ascent appends one length-raising simple
    reflection at a time, so the word is reduced by construction. The
    kernel asserts maximality before trusted construction: the word's
    inversion count must equal the positive-root count.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    word = _weyl_longest_word_kernel(rows)
    num_positive_roots = len(enumerate_positive_roots(rows))
    inversions = _weyl_word_inversions_kernel(rows, word)
    if len(word) != len(inversions) or len(word) != num_positive_roots:
        raise RuntimeError(
            "Weyl longest-word kernel returned an incomplete reduced word"
        )
    return WeylLongestElementResult._from_kernel(
        cartan, word, len(word), num_positive_roots
    )


def weyl_group_order(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> WeylGroupOrderResult:
    """Compute the exact order of a finite Weyl group without enumeration."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    return WeylGroupOrderResult._from_kernel(cartan, _weyl_group_order(rows))


def _reflection_matrix(
    rows: tuple[tuple[int, ...], ...], index: int, *, transpose: bool
) -> tuple[tuple[int, ...], ...]:
    """Return the exact matrix of ``s_index`` on a root-like lattice.

    On the root lattice ``s_i(v) = v - <v, alpha_i^vee> alpha_i`` with
    ``<alpha_j, alpha_i^vee> = A[i][j]``; the coroot lattice uses the
    transposed pairing ``A[j][i]``.
    """
    rank = len(rows)
    matrix: list[list[int]] = [[int(k == j) for j in range(rank)] for k in range(rank)]
    for target in range(rank):
        for source in range(rank):
            pairing = rows[index][source] if not transpose else rows[source][index]
            matrix[target][source] -= int(target == index) * pairing
    return tuple(tuple(row) for row in matrix)


def _weight_reflection_matrix(
    rows: tuple[tuple[int, ...], ...], index: int, *, transpose: bool
) -> tuple[tuple[int, ...], ...]:
    """Return the exact matrix of ``s_index`` on a weight-like lattice.

    For weights ``s_i(lambda) = lambda - lambda_i alpha_i`` with
    ``alpha_i = sum_k A[k][i] omega_k``; coweights use the transposed
    coefficients ``A[i][k]``.
    """
    rank = len(rows)
    matrix: list[list[int]] = [[int(k == j) for j in range(rank)] for k in range(rank)]
    for target in range(rank):
        coefficient = rows[target][index] if not transpose else rows[index][target]
        matrix[target][index] -= coefficient
    return tuple(tuple(row) for row in matrix)


def _square_is_identity(matrix: tuple[tuple[int, ...], ...]) -> bool:
    rank = len(matrix)
    return all(
        sum(matrix[row][k] * matrix[k][column] for k in range(rank))
        == int(row == column)
        for row in range(rank)
        for column in range(rank)
    )


def simple_reflections(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> SimpleReflectionsResult:
    """Compute every simple-reflection matrix on the four root-datum lattices."""
    from jacobian.math.matrices.values import IntegerMatrix

    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)

    def _as_integer_matrix(
        entries: tuple[tuple[int, ...], ...],
    ) -> IntegerMatrix:
        return IntegerMatrix(
            row_count=rank,
            column_count=rank,
            entries=entries,
        )

    root = tuple(
        _as_integer_matrix(_reflection_matrix(rows, i, transpose=False))
        for i in range(rank)
    )
    coroot = tuple(
        _as_integer_matrix(_reflection_matrix(rows, i, transpose=True))
        for i in range(rank)
    )
    weight = tuple(
        _as_integer_matrix(_weight_reflection_matrix(rows, i, transpose=False))
        for i in range(rank)
    )
    coweight = tuple(
        _as_integer_matrix(_weight_reflection_matrix(rows, i, transpose=True))
        for i in range(rank)
    )
    for family in (root, coroot, weight, coweight):
        for reflection in family:
            if not _square_is_identity(reflection.entries):
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.reflection_not_involution",
                    message="simple reflections must square to the identity",
                )
    return SimpleReflectionsResult._from_kernel(
        cartan,
        root_matrices=root,
        coroot_matrices=coroot,
        weight_matrices=weight,
        coweight_matrices=coweight,
    )


__all__ = [
    "cartan_matrix_from_type",
    "positive_roots",
    "root_system_data",
    "simple_reflection",
    "simple_reflections",
    "weyl_element_descents",
    "weyl_element_length",
    "weyl_group_order",
    "weyl_longest_element",
]
