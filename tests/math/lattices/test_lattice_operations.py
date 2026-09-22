"""Tests for the issue-#1739 integer-lattice structural operations."""

from __future__ import annotations

import random
from collections.abc import Callable
from fractions import Fraction
from functools import reduce
from itertools import combinations, pairwise, permutations
from math import gcd

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices._hnf import compute_hermite_normal_form
from jacobian.math.lattices._lattice import reduce_lattice_basis
from jacobian.math.lattices._lattice_ops import smith_invariant_factors
from jacobian.math.lattices._models import (
    HermiteNormalFormRequest,
    IntegerLattice,
    LatticeReductionRequest,
    SublatticeIndexRequest,
)
from jacobian.math.lattices.operations import (
    compute_canonical_basis,
    compute_direct_sum,
    compute_discriminant_group,
    compute_dual,
    compute_orthogonal_complement,
    compute_orthogonal_sum,
    compute_rank_gram,
    compute_saturation,
    compute_sublattice_index,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    IntegerMatrix,
    RationalMatrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lattice(ambient: int, basis: list[list[int]]) -> IntegerLattice:
    return IntegerLattice.model_validate(
        {
            "ambient_dimension": ambient,
            "basis": {"entries": [list(row) for row in basis]},
        }
    )


def _identity_entries(order: int) -> list[list[int]]:
    return [[int(row == column) for column in range(order)] for row in range(order)]


# ---------------------------------------------------------------------------
# IntegerLattice value model
# ---------------------------------------------------------------------------


def test_integer_lattice_rejects_column_mismatch() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _lattice(2, [[1, 0, 0], [0, 1, 0]])
    assert exc_info.value.errors()[0]["type"] == "lattice.basis_columns_mismatch"


def test_integer_lattice_rejects_too_many_rows() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _lattice(1, [[1], [2]])
    assert exc_info.value.errors()[0]["type"] == "lattice.rank_exceeds_ambient"


def test_integer_lattice_rejects_empty_basis() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IntegerLattice.model_validate(
            {"ambient_dimension": 2, "basis": {"entries": []}}
        )
    assert exc_info.value.errors()[0]["type"] == "lattice.basis_columns_mismatch"


# ---------------------------------------------------------------------------
# lattice.rank_gram.compute
# ---------------------------------------------------------------------------


def test_rank_gram_of_identity_is_identity() -> None:
    result = compute_rank_gram(_lattice(2, [[1, 0], [0, 1]]))
    assert result.rank == 2
    assert result.ambient_dimension == 2
    assert result.squared_covolume == 1
    assert result.covolume_rational is True
    gram = result.gram_matrix.entries
    assert gram == ((1, 0), (0, 1))


def test_rank_gram_of_scaled_lattice() -> None:
    """Gram of diag(2,3) is diag(4,9) with squared covolume 36."""
    result = compute_rank_gram(_lattice(2, [[2, 0], [0, 3]]))
    assert result.rank == 2
    gram = result.gram_matrix.entries
    assert gram == ((4, 0), (0, 9))
    assert result.squared_covolume == 36


def test_rank_gram_rejects_dependent_basis_rows() -> None:
    with pytest.raises(ValueError, match="full row rank"):
        compute_rank_gram(_lattice(2, [[1, 0], [2, 0]]))


# ---------------------------------------------------------------------------
# lattice.canonical_basis.compute
# ---------------------------------------------------------------------------


def test_canonical_basis_of_identity() -> None:
    result = compute_canonical_basis(_lattice(2, [[1, 0], [0, 1]]))
    assert result.rank == 2
    assert result.canonical_basis.entries == ((1, 0), (0, 1))


def test_canonical_basis_is_hnf() -> None:
    """A non-HNF basis maps to its HNF canonical form."""
    result = compute_canonical_basis(_lattice(2, [[2, 1], [1, 1]]))
    # HNF is upper-triangular with positive pivots.
    hnf = result.canonical_basis.entries
    assert int(hnf[1][0]) == 0
    assert int(hnf[0][0]) > 0
    assert int(hnf[1][1]) > 0


def test_canonical_basis_transformation_binds() -> None:
    """T @ basis == canonical_basis (the transformation relation)."""
    result = compute_canonical_basis(_lattice(2, [[3, 1], [1, 2]]))
    hnf = result.canonical_basis.entries
    transform = result.transformation.entries
    basis = [[3, 1], [1, 2]]
    for i in range(2):
        for j in range(2):
            value = sum(int(transform[i][k]) * basis[k][j] for k in range(2))
            assert int(hnf[i][j]) == value


# ---------------------------------------------------------------------------
# lattice.dual.compute
# ---------------------------------------------------------------------------


def test_dual_of_unimodular_is_itself() -> None:
    result = compute_dual(_lattice(2, [[1, 0], [0, 1]]))
    dual = result.dual_basis.entries
    assert dual[0][0].num == 1 and dual[0][0].den == 1
    assert dual[0][1].num == 0 and dual[0][1].den == 1
    assert dual[1][0].num == 0 and dual[1][0].den == 1
    assert dual[1][1].num == 1 and dual[1][1].den == 1


def test_dual_pairing_is_integer() -> None:
    """The dual basis times the basis transpose should be the identity."""
    result = compute_dual(_lattice(2, [[2, 0], [0, 3]]))
    # B* = (B B^T)^{-1} B; B* B^T = I
    # B B^T = diag(4, 9), so B* = diag(1/2, 1/3)
    dual = result.dual_basis.entries
    assert dual[0][0].num == 1 and dual[0][0].den == 2
    assert dual[1][1].num == 1 and dual[1][1].den == 3


# ---------------------------------------------------------------------------
# lattice.saturation.compute
# ---------------------------------------------------------------------------


def test_saturation_of_primitive_lattice_is_identity() -> None:
    result = compute_saturation(_lattice(2, [[1, 0], [0, 1]]))
    assert result.saturated_basis.entries == ((1, 0), (0, 1))
    assert result.saturation_index == 1


def test_saturation_of_2z_squared() -> None:
    """sat(2 ZZ^2) = ZZ^2 with index 4."""
    result = compute_saturation(_lattice(2, [[2, 0], [0, 2]]))
    assert result.saturated_basis.entries == ((1, 0), (0, 1))
    assert result.saturation_index == 4


def test_saturation_index_2() -> None:
    result = compute_saturation(_lattice(2, [[2, 0], [0, 1]]))
    assert result.saturated_basis.entries == ((1, 0), (0, 1))
    assert result.saturation_index == 2


def test_saturation_rank_deficient() -> None:
    """The saturation remains in the rational span of the source lattice."""
    result = compute_saturation(_lattice(2, [[2, 0]]))
    assert result.saturated_basis.entries == ((1, 0),)
    assert result.inclusion_transform.entries == ((2,),)
    assert result.saturation_index == 2


def test_saturation_of_diagonal_line_is_its_primitive_line() -> None:
    result = compute_saturation(_lattice(2, [[2, 2]]))

    assert result.saturated_basis.entries == ((1, 1),)
    assert result.inclusion_transform.entries == ((2,),)
    assert result.saturation_index == 2


def test_saturation_of_coordinate_plane_does_not_gain_ambient_generator() -> None:
    result = compute_saturation(_lattice(3, [[2, 0, 0], [0, 2, 0]]))

    assert result.saturated_basis.entries == ((1, 0, 0), (0, 1, 0))
    assert result.inclusion_transform.entries == ((2, 0), (0, 2))
    assert result.saturation_index == 4


# ---------------------------------------------------------------------------
# lattice.sublattice_index.compute
# ---------------------------------------------------------------------------


def test_sublattice_index_double() -> None:
    """Index of 2ZZ inside ZZ is 2."""
    result = compute_sublattice_index(
        _lattice(1, [[2]]),
        _lattice(1, [[1]]),
        IntegerMatrix.model_validate({"entries": [[2]]}),
    )
    assert result.index == 2
    assert result.invariant_factors == (2,)
    assert result.free_rank == 0


def test_sublattice_index_quadratic() -> None:
    """Index of <(2,0),(0,2)> inside <(1,0),(0,1)> is 4."""
    result = compute_sublattice_index(
        _lattice(2, [[2, 0], [0, 2]]),
        _lattice(2, [[1, 0], [0, 1]]),
        IntegerMatrix.model_validate({"entries": [[2, 0], [0, 2]]}),
    )
    assert result.index == 4
    assert result.invariant_factors == (2, 2)
    assert result.free_rank == 0


def test_sublattice_index_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SublatticeIndexRequest.model_validate(
            {
                "sublattice": _lattice(1, [[1]]),
                "parent": _lattice(2, [[1, 0], [0, 1]]),
                "embedding": {"entries": [[1, 0]]},
            }
        )
    assert exc_info.value.errors()[0]["type"] == "lattice.ambient_dimensions_mismatch"


# ---------------------------------------------------------------------------
# Integer Smith invariant factors
# ---------------------------------------------------------------------------


def _minor_determinant(
    matrix: list[list[int]], rows: tuple[int, ...], columns: tuple[int, ...]
) -> int:
    total = 0
    for ordering in permutations(range(len(rows))):
        inversions = sum(
            ordering[left] > ordering[right]
            for left in range(len(ordering))
            for right in range(left + 1, len(ordering))
        )
        total += (-1) ** inversions * reduce(
            lambda product, index: (
                product * matrix[rows[index]][columns[ordering[index]]]
            ),
            range(len(rows)),
            1,
        )
    return total


def _independent_invariant_factors(matrix: list[list[int]]) -> list[int]:
    row_count = len(matrix)
    column_count = len(matrix[0]) if row_count else 0
    previous = 1
    factors: list[int] = []
    for order in range(1, min(row_count, column_count) + 1):
        divisor = 0
        for rows in combinations(range(row_count), order):
            for columns in combinations(range(column_count), order):
                divisor = gcd(divisor, abs(_minor_determinant(matrix, rows, columns)))
        if divisor == 0:
            break
        factors.append(divisor // previous)
        previous = divisor
    return factors


def test_smith_invariants_match_independent_determinantal_divisors() -> None:
    rng = random.Random(1739)
    for _ in range(24):
        rows = rng.randrange(1, 5)
        columns = rng.randrange(1, 5)
        matrix = [[rng.randrange(-4, 5) for _ in range(columns)] for _ in range(rows)]
        expected = _independent_invariant_factors(matrix)
        actual = smith_invariant_factors(matrix)
        assert actual == expected
        assert all(right % left == 0 for left, right in pairwise(actual))
        rank = len(actual)
        if rank:
            product = 1
            for factor in actual:
                product *= factor
            assert product == gcd(
                *(
                    abs(_minor_determinant(matrix, r, c))
                    for r in combinations(range(rows), rank)
                    for c in combinations(range(columns), rank)
                )
            )


# ---------------------------------------------------------------------------
# lattice.discriminant_group.compute
# ---------------------------------------------------------------------------


def test_discriminant_group_of_unimodular_is_trivial() -> None:
    result = compute_discriminant_group(_lattice(2, [[1, 0], [0, 1]]))
    assert result.discriminant_order == 1
    assert result.invariant_factors == ()


def test_discriminant_group_of_2z_squared() -> None:
    """disc group of 2 ZZ^2 has order |det diag(4,4)| = 16."""
    result = compute_discriminant_group(_lattice(2, [[2, 0], [0, 2]]))
    assert result.discriminant_order == 16
    assert result.invariant_factors == (4, 4)


def test_discriminant_group_order_matches_gram_det() -> None:
    """discriminant_order equals |det(B B^T)|."""
    rg = compute_rank_gram(_lattice(2, [[3, 1], [1, 2]]))
    dg = compute_discriminant_group(_lattice(2, [[3, 1], [1, 2]]))
    assert int(rg.squared_covolume) == dg.discriminant_order


# ---------------------------------------------------------------------------
# lattice.orthogonal_complement.compute
# ---------------------------------------------------------------------------


def test_orthogonal_complement_of_line() -> None:
    """Complement of <(1,0)> in QQ^2 is <(0,1)>."""
    result = compute_orthogonal_complement(_lattice(2, [[1, 0]]))
    assert result.complement_rank == 1


def test_orthogonal_complement_vectors_have_expected_dimension_and_pairings() -> None:
    basis = [[1, 2, 3, 4], [2, -1, 0, 1]]
    result = compute_orthogonal_complement(_lattice(4, basis))
    complement = [
        [coordinate.as_fraction() for coordinate in vector]
        for vector in result.complement_basis.vectors
    ]

    assert result.complement_rank == 2
    assert all(
        sum(left[axis] * right[axis] for axis in range(4)) == 0
        for left in basis
        for right in complement
    )
    # Independent rank-two witness: some 2x2 minor must be nonzero.
    assert any(
        complement[0][first] * complement[1][second]
        != complement[0][second] * complement[1][first]
        for first in range(4)
        for second in range(first + 1, 4)
    )


def test_orthogonal_complement_of_full_rank_is_zero() -> None:
    result = compute_orthogonal_complement(_lattice(2, [[1, 0], [0, 1]]))
    assert result.complement_rank == 0
    assert result.complement_basis.ambient_dimension == 2
    assert result.complement_basis.vectors == ()


def test_orthogonal_complement_of_plane_in_3d() -> None:
    """Complement of <(1,0,0),(0,1,0)> in QQ^3 is <(0,0,1)>."""
    result = compute_orthogonal_complement(_lattice(3, [[1, 0, 0], [0, 1, 0]]))
    assert result.complement_rank == 1
    assert result.complement_basis.ambient_dimension == 3
    assert len(result.complement_basis.vectors[0]) == 3


# ---------------------------------------------------------------------------
# lattice.direct_sum.compute and lattice.orthogonal_sum.compute
# ---------------------------------------------------------------------------


def test_direct_sum_of_two_identity() -> None:
    result = compute_direct_sum(
        _lattice(2, [[1, 0], [0, 1]]),
        _lattice(2, [[1, 0], [0, 1]]),
    )
    assert result.ambient_dimension == 4
    assert result.direct_sum_basis.entries == (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )


def test_direct_sum_block_diagonal() -> None:
    result = compute_direct_sum(
        _lattice(1, [[2]]),
        _lattice(1, [[3]]),
    )
    assert result.ambient_dimension == 2
    assert result.direct_sum_basis.entries == ((2, 0), (0, 3))


def test_orthogonal_sum_of_two_identity() -> None:
    result = compute_orthogonal_sum(
        _lattice(2, [[1, 0], [0, 1]]),
        _lattice(1, [[3]]),
    )
    assert result.ambient_dimension == 3
    assert result.orthogonal_sum_basis.entries == (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 3),
    )


@pytest.mark.parametrize("operation", [compute_direct_sum, compute_orthogonal_sum])
def test_lattice_sum_rejects_combined_dimension_before_backend(
    operation: Callable[[IntegerLattice, IntegerLattice], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _lattice(17, [[int(i == j) for j in range(17)] for i in range(17)])
    second = _lattice(17, [[int(i == j) for j in range(17)] for i in range(17)])

    import jacobian.math.lattices.operations as lattice_operations

    monkeypatch.setattr(
        lattice_operations,
        "_direct_sum" if operation is compute_direct_sum else "_orthogonal_sum",
        lambda *_args: pytest.fail("sum backend must not run after admission failure"),
    )
    with pytest.raises(OperationDomainValidationError, match="output envelope"):
        operation(first, second)


# ---------------------------------------------------------------------------
# lattice.basis.reduce and lattice.hermite_normal_form.compute
# ---------------------------------------------------------------------------


def test_integer_matrix_still_parses_order_33_identity() -> None:
    """Shared IntegerMatrix is the exact-linear envelope, not the LLL bound."""

    matrix = IntegerMatrix.model_validate(
        {"entries": _identity_entries(MAX_MATRIX_DIMENSION + 1)}
    )
    assert len(matrix.entries) == MAX_MATRIX_DIMENSION + 1
    assert len(matrix.entries[0]) == MAX_MATRIX_DIMENSION + 1


def test_lattice_reduction_request_rejects_order_33_identity() -> None:
    with pytest.raises(ValidationError) as exc_info:
        LatticeReductionRequest.model_validate(
            {"basis": {"entries": _identity_entries(MAX_MATRIX_DIMENSION + 1)}}
        )
    assert exc_info.value.errors()[0]["type"] == "lattice.budget_exceeded"


def test_hermite_accepts_order_33_identity() -> None:
    request = HermiteNormalFormRequest.model_validate(
        {"matrix": {"entries": _identity_entries(MAX_MATRIX_DIMENSION + 1)}}
    )
    result = compute_hermite_normal_form(request)
    assert result.normal_form == request.matrix
    assert result.transformation == request.matrix


def test_lattice_reduction_accepts_order_32_identity() -> None:
    request = LatticeReductionRequest.model_validate(
        {"basis": {"entries": _identity_entries(MAX_MATRIX_DIMENSION)}}
    )
    result = reduce_lattice_basis(request)
    assert result.rank == MAX_MATRIX_DIMENSION
    assert result.reduced_basis.entries == tuple(
        tuple(row) for row in _identity_entries(MAX_MATRIX_DIMENSION)
    )


def test_hermite_accepts_order_32_identity() -> None:
    request = HermiteNormalFormRequest.model_validate(
        {"matrix": {"entries": _identity_entries(MAX_MATRIX_DIMENSION)}}
    )
    result = compute_hermite_normal_form(request)
    assert result.normal_form.entries == tuple(
        tuple(row) for row in _identity_entries(MAX_MATRIX_DIMENSION)
    )


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def test_lattice_reduction_rejects_order_above_32_before_backend() -> None:
    order = MAX_MATRIX_DIMENSION + 1
    entries = tuple(
        tuple(1 if row == column else 0 for column in range(order))
        for row in range(order)
    )
    matrix = IntegerMatrix(entries=entries)

    with pytest.raises(ValidationError):
        LatticeReductionRequest(basis=matrix)
    with pytest.raises(OperationDomainValidationError, match="32"):
        reduce_lattice_basis(LatticeReductionRequest.model_construct(basis=matrix))


def test_all_new_operations_registered_in_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    expected = {
        "lattice.rank_gram.compute",
        "lattice.canonical_basis.compute",
        "lattice.dual.compute",
        "lattice.saturation.compute",
        "lattice.sublattice_index.compute",
        "lattice.discriminant_group.compute",
        "lattice.orthogonal_complement.compute",
        "lattice.direct_sum.compute",
        "lattice.orthogonal_sum.compute",
    }
    assert expected <= ids


def test_lattice_wire_parsing_does_not_prove_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.lattices.operations as lattice_operations

    lattice = _lattice(2, [[1, 0], [2, 0]])
    monkeypatch.setattr(
        lattice_operations,
        "integer_rank",
        lambda *_: pytest.fail("parsing must not compute rank"),
    )
    assert IntegerLattice.model_validate_json(lattice.model_dump_json()) == lattice


def test_native_lll_uses_the_same_envelope_as_the_published_operation() -> None:
    from jacobian.math.lattices import reduce_basis

    with pytest.raises(OperationDomainValidationError, match="32"):
        reduce_basis(_identity_entries(33))
    with pytest.raises(OperationDomainValidationError, match="256"):
        reduce_basis([[10**256]])


@pytest.mark.parametrize("hermite", [False, True])
def test_lattice_transform_results_retain_their_source_through_serialization(
    hermite: bool,
) -> None:
    matrix = IntegerMatrix(entries=((2, 3, 1), (5, 7, 2)))
    if hermite:
        hnf_result = compute_hermite_normal_form(
            HermiteNormalFormRequest(matrix=matrix)
        )
        restored_hnf = type(hnf_result).model_validate_json(
            hnf_result.model_dump_json()
        )
        assert restored_hnf.matrix == matrix
        left = restored_hnf.normal_form.entries
        transformation = restored_hnf.transformation
    else:
        lll_result = reduce_lattice_basis(LatticeReductionRequest(basis=matrix))
        restored_lll = type(lll_result).model_validate_json(
            lll_result.model_dump_json()
        )
        assert restored_lll.basis == matrix
        left = restored_lll.reduced_basis.entries
        transformation = restored_lll.transformation
    transformed = tuple(
        tuple(
            sum(
                transformation.entries[i][k] * matrix.entries[k][j]
                for k in range(matrix.row_count)
            )
            for j in range(matrix.column_count)
        )
        for i in range(matrix.row_count)
    )
    assert left == transformed


def _rational_entries(matrix: RationalMatrix) -> list[list[Fraction]]:
    return [[entry.as_fraction() for entry in row] for row in matrix.entries]


def test_dual_pairing_identity_on_a_skewed_lattice() -> None:
    basis = [[3, 1], [1, 2]]

    result = compute_dual(_lattice(2, basis))

    dual = _rational_entries(result.dual_basis)
    for left in range(2):
        for right in range(2):
            assert sum(dual[left][k] * basis[right][k] for k in range(2)) == Fraction(
                left == right
            )
    gram = [
        [sum(basis[i][k] * basis[j][k] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    assert _rational_entries(result.dual_gram) == [
        [Fraction(gram[1][1], determinant), Fraction(-gram[0][1], determinant)],
        [Fraction(-gram[1][0], determinant), Fraction(gram[0][0], determinant)],
    ]


def test_dual_pairing_identity_on_a_rectangular_rank_three_lattice() -> None:
    basis = [[2, 1, 0, 3], [1, -1, 2, 0], [0, 2, 1, 1]]

    dual = _rational_entries(compute_dual(_lattice(4, basis)).dual_basis)

    assert [
        [sum(dual[i][k] * basis[j][k] for k in range(4)) for j in range(3)]
        for i in range(3)
    ] == [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]


def test_unimodular_dual_is_integral_and_involutive() -> None:
    basis = [[2, 1], [1, 1]]

    result = compute_dual(_lattice(2, basis))

    dual = _rational_entries(result.dual_basis)
    assert all(entry.denominator == 1 for row in dual for entry in row)
    assert compute_discriminant_group(_lattice(2, basis)).discriminant_order == 1
    involution = compute_dual(
        _lattice(2, [[int(entry) for entry in row] for row in dual])
    )
    assert _rational_entries(involution.dual_basis) == [
        [Fraction(value) for value in row] for row in basis
    ]


def test_canonical_transformation_is_unimodular_on_a_skewed_lattice() -> None:
    result = compute_canonical_basis(_lattice(2, [[3, 1], [1, 2]]))

    transform = result.transformation.entries
    assert (
        abs(
            int(transform[0][0]) * int(transform[1][1])
            - int(transform[0][1]) * int(transform[1][0])
        )
        == 1
    )


def test_discriminant_reconstructs_from_gram_on_a_skewed_lattice() -> None:
    lattice = _lattice(2, [[3, 1], [1, 2]])

    gram = compute_rank_gram(lattice)
    discriminant = compute_discriminant_group(lattice)

    entries = gram.gram_matrix.entries
    determinant = int(entries[0][0]) * int(entries[1][1]) - int(entries[0][1]) * int(
        entries[1][0]
    )
    assert int(gram.squared_covolume) == determinant
    assert determinant == discriminant.discriminant_order == 25
    assert discriminant.invariant_factors == (5, 5)
    product = 1
    for factor in discriminant.invariant_factors:
        product *= int(factor)
    assert product == discriminant.discriminant_order


def test_dual_and_discriminant_reject_a_rank_deficient_basis() -> None:
    deficient = _lattice(2, [[1, 0], [2, 0]])

    with pytest.raises(OperationDomainValidationError, match="full row rank"):
        compute_dual(deficient)
    with pytest.raises(OperationDomainValidationError, match="full row rank"):
        compute_discriminant_group(deficient)
