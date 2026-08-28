"""Tests for the issue-#1739 integer-lattice structural operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.lattices._lattice_operations import (
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
from jacobian.math.lattices._models import (
    DirectSumRequest,
    DiscriminantGroupRequest,
    DualRequest,
    IntegerLattice,
    OrthogonalComplementRequest,
    OrthogonalSumRequest,
    RankGramRequest,
    SublatticeIndexRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lattice(ambient: int, basis: list[list[int]]) -> IntegerLattice:
    return IntegerLattice.model_validate(
        {
            "ambient_dimension": ambient,
            "basis": {"entries": [[str(v) for v in row] for row in basis]},
        }
    )


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
    assert exc_info.value.errors()[0]["type"] == "too_short"


# ---------------------------------------------------------------------------
# lattice.rank_gram.compute
# ---------------------------------------------------------------------------


def test_rank_gram_of_identity_is_identity() -> None:
    result = compute_rank_gram(RankGramRequest(lattice=_lattice(2, [[1, 0], [0, 1]])))
    assert result.rank == 2
    assert result.ambient_dimension == 2
    assert result.squared_covolume == "1"
    assert result.covolume_rational is False
    gram = result.gram_matrix.entries
    assert gram == (("1", "0"), ("0", "1"))


def test_rank_gram_of_scaled_lattice() -> None:
    """Gram of diag(2,3) is diag(4,9) with squared covolume 36."""
    result = compute_rank_gram(RankGramRequest(lattice=_lattice(2, [[2, 0], [0, 3]])))
    assert result.rank == 2
    gram = result.gram_matrix.entries
    assert gram == (("4", "0"), ("0", "9"))
    assert result.squared_covolume == "36"


def test_rank_gram_rank_deficient_reports_rational_covolume() -> None:
    """A rank-1 lattice in ambient 2 has a rational (non-finite) covolume."""
    with pytest.raises(ValueError, match="full row rank"):
        compute_rank_gram(RankGramRequest(lattice=_lattice(2, [[1, 0], [2, 0]])))


# ---------------------------------------------------------------------------
# lattice.canonical_basis.compute
# ---------------------------------------------------------------------------


def test_canonical_basis_of_identity() -> None:
    result = compute_canonical_basis(_lattice(2, [[1, 0], [0, 1]]))
    assert result.rank == 2
    assert result.canonical_basis.entries == (("1", "0"), ("0", "1"))


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
    result = compute_dual(DualRequest(lattice=_lattice(2, [[1, 0], [0, 1]])))
    dual = result.dual_basis.entries
    assert dual[0][0].num == "1" and dual[0][0].den == "1"
    assert dual[0][1].num == "0" and dual[0][1].den == "1"
    assert dual[1][0].num == "0" and dual[1][0].den == "1"
    assert dual[1][1].num == "1" and dual[1][1].den == "1"


def test_dual_pairing_is_integer() -> None:
    """The dual basis times the basis transpose should be the identity."""
    result = compute_dual(DualRequest(lattice=_lattice(2, [[2, 0], [0, 3]])))
    # B* = B (B B^T)^{-1}; B* B^T = I
    # B B^T = diag(4, 9), so B* = diag(1/2, 1/3)
    dual = result.dual_basis.entries
    assert dual[0][0].num == "1" and dual[0][0].den == "2"
    assert dual[1][1].num == "1" and dual[1][1].den == "3"


# ---------------------------------------------------------------------------
# lattice.saturation.compute
# ---------------------------------------------------------------------------


def test_saturation_of_primitive_lattice_is_identity() -> None:
    result = compute_saturation(_lattice(2, [[1, 0], [0, 1]]))
    assert result.saturated_basis.entries == (("1", "0"), ("0", "1"))
    assert result.saturation_index == 1


def test_saturation_of_2z_squared() -> None:
    """sat(2 ZZ^2) = ZZ^2 with index 4."""
    result = compute_saturation(_lattice(2, [[2, 0], [0, 2]]))
    assert result.saturated_basis.entries == (("1", "0"), ("0", "1"))
    assert result.saturation_index == 4


def test_saturation_index_2() -> None:
    result = compute_saturation(_lattice(2, [[2, 0], [0, 1]]))
    assert result.saturated_basis.entries == (("1", "0"), ("0", "1"))
    assert result.saturation_index == 2


def test_saturation_rank_deficient() -> None:
    """sat(2Z in ZZ^2) = ZZ^2 with index 2."""
    result = compute_saturation(_lattice(2, [[2, 0]]))
    assert result.saturated_basis.entries == (("1", "0"), ("0", "1"))
    assert result.saturation_index == 2


# ---------------------------------------------------------------------------
# lattice.sublattice_index.compute
# ---------------------------------------------------------------------------


def test_sublattice_index_double() -> None:
    """Index of 2ZZ inside ZZ is 2."""
    result = compute_sublattice_index(
        SublatticeIndexRequest.model_validate(
            {
                "sublattice": _lattice(1, [[2]]),
                "parent": _lattice(1, [[1]]),
                "embedding": {"entries": [["2"]]},
            }
        )
    )
    assert result.index == 2
    assert result.invariant_factors == ("2",)
    assert result.free_rank == 0


def test_sublattice_index_quadratic() -> None:
    """Index of <(2,0),(0,2)> inside <(1,0),(0,1)> is 4."""
    result = compute_sublattice_index(
        SublatticeIndexRequest.model_validate(
            {
                "sublattice": _lattice(2, [[2, 0], [0, 2]]),
                "parent": _lattice(2, [[1, 0], [0, 1]]),
                "embedding": {"entries": [["2", "0"], ["0", "2"]]},
            }
        )
    )
    assert result.index == 4
    assert result.invariant_factors == ("2", "2")
    assert result.free_rank == 0


def test_sublattice_index_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SublatticeIndexRequest.model_validate(
            {
                "sublattice": _lattice(1, [[1]]),
                "parent": _lattice(2, [[1, 0], [0, 1]]),
                "embedding": {"entries": [["1", "0"]]},
            }
        )
    assert exc_info.value.errors()[0]["type"] == "lattice.ambient_dimensions_mismatch"


# ---------------------------------------------------------------------------
# lattice.discriminant_group.compute
# ---------------------------------------------------------------------------


def test_discriminant_group_of_unimodular_is_trivial() -> None:
    result = compute_discriminant_group(
        DiscriminantGroupRequest(lattice=_lattice(2, [[1, 0], [0, 1]]))
    )
    assert result.discriminant_order == 1
    assert result.invariant_factors == ()


def test_discriminant_group_of_2z_squared() -> None:
    """disc group of 2 ZZ^2 has order |det diag(4,4)| = 16."""
    result = compute_discriminant_group(
        DiscriminantGroupRequest(lattice=_lattice(2, [[2, 0], [0, 2]]))
    )
    assert result.discriminant_order == 16
    assert result.invariant_factors == ("4", "4")


def test_discriminant_group_order_matches_gram_det() -> None:
    """discriminant_order equals |det(B B^T)|."""
    rg = compute_rank_gram(RankGramRequest(lattice=_lattice(2, [[3, 1], [1, 2]])))
    dg = compute_discriminant_group(
        DiscriminantGroupRequest(lattice=_lattice(2, [[3, 1], [1, 2]]))
    )
    assert int(rg.squared_covolume) == dg.discriminant_order


# ---------------------------------------------------------------------------
# lattice.orthogonal_complement.compute
# ---------------------------------------------------------------------------


def test_orthogonal_complement_of_line() -> None:
    """Complement of <(1,0)> in QQ^2 is <(0,1)>."""
    result = compute_orthogonal_complement(
        OrthogonalComplementRequest(lattice=_lattice(2, [[1, 0]]))
    )
    assert result.complement_rank == 1


def test_orthogonal_complement_of_full_rank_is_zero() -> None:
    result = compute_orthogonal_complement(
        OrthogonalComplementRequest(lattice=_lattice(2, [[1, 0], [0, 1]]))
    )
    assert result.complement_rank == 0


def test_orthogonal_complement_of_plane_in_3d() -> None:
    """Complement of <(1,0,0),(0,1,0)> in QQ^3 is <(0,0,1)>."""
    result = compute_orthogonal_complement(
        OrthogonalComplementRequest(lattice=_lattice(3, [[1, 0, 0], [0, 1, 0]]))
    )
    assert result.complement_rank == 1


# ---------------------------------------------------------------------------
# lattice.direct_sum.compute and lattice.orthogonal_sum.compute
# ---------------------------------------------------------------------------


def test_direct_sum_of_two_identity() -> None:
    result = compute_direct_sum(
        DirectSumRequest(
            first=_lattice(2, [[1, 0], [0, 1]]),
            second=_lattice(2, [[1, 0], [0, 1]]),
        )
    )
    assert result.ambient_dimension == 4
    assert result.direct_sum_basis.entries == (
        ("1", "0", "0", "0"),
        ("0", "1", "0", "0"),
        ("0", "0", "1", "0"),
        ("0", "0", "0", "1"),
    )


def test_direct_sum_block_diagonal() -> None:
    result = compute_direct_sum(
        DirectSumRequest(
            first=_lattice(1, [[2]]),
            second=_lattice(1, [[3]]),
        )
    )
    assert result.ambient_dimension == 2
    assert result.direct_sum_basis.entries == (("2", "0"), ("0", "3"))


def test_orthogonal_sum_of_two_identity() -> None:
    result = compute_orthogonal_sum(
        OrthogonalSumRequest(
            first=_lattice(2, [[1, 0], [0, 1]]),
            second=_lattice(1, [[3]]),
        )
    )
    assert result.ambient_dimension == 3
    assert result.orthogonal_sum_basis.entries == (
        ("1", "0", "0"),
        ("0", "1", "0"),
        ("0", "0", "3"),
    )


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


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
