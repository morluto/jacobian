"""Rank-zero lattices retain the ambient integer coordinate space."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices import (
    IntegerLattice,
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
from jacobian.math.lattices._models import SublatticeIndexRequest
from jacobian.math.matrices.values import IntegerMatrix


@pytest.mark.parametrize("ambient", [0, 3])
def test_zero_lattice_structure_retains_ambient_dimension(ambient: int) -> None:
    lattice = IntegerLattice(
        ambient_dimension=ambient,
        basis=IntegerMatrix(entries=(), row_count=0, column_count=ambient),
    )
    lattice = IntegerLattice.model_validate_json(lattice.model_dump_json())
    gram = compute_rank_gram(lattice)
    assert gram.rank == 0
    assert gram.gram_matrix.row_count == gram.gram_matrix.column_count == 0
    assert gram.squared_covolume == 1
    assert gram.covolume_rational
    canonical = compute_canonical_basis(lattice)
    assert canonical.canonical_basis == lattice.basis
    assert canonical.rank == 0
    assert canonical.transformation.column_count == 0
    dual = compute_dual(lattice)
    assert dual.dual_basis.row_count == 0
    assert dual.dual_basis.column_count == ambient
    assert dual.dual_gram.row_count == dual.dual_gram.column_count == 0
    saturation = compute_saturation(lattice)
    assert saturation.saturated_basis == lattice.basis
    assert saturation.saturation_index == 1
    assert saturation.inclusion_transform.column_count == 0
    discriminant = compute_discriminant_group(lattice)
    assert discriminant.discriminant_order == 1
    assert discriminant.invariant_factors == ()
    complement = compute_orthogonal_complement(lattice)
    assert complement.complement_rank == ambient
    assert complement.complement_basis.ambient_dimension == ambient
    assert tuple(
        tuple(value.as_fraction() for value in row)
        for row in complement.complement_basis.vectors
    ) == tuple(tuple(int(i == j) for j in range(ambient)) for i in range(ambient))
    for result in (gram, canonical, dual, saturation, discriminant, complement):
        assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("ambient", [0, 3])
def test_sums_preserve_empty_operand_coordinates(ambient: int) -> None:
    zero = IntegerLattice(
        ambient_dimension=ambient,
        basis=IntegerMatrix(entries=(), row_count=0, column_count=ambient),
    )
    nonzero = IntegerLattice(ambient_dimension=1, basis=IntegerMatrix(entries=((2,),)))
    for first, second, expected in (
        (zero, nonzero, ((0,) * ambient + (2,),)),
        (nonzero, zero, ((2,) + (0,) * ambient,)),
        (zero, zero, ()),
    ):
        direct = compute_direct_sum(first, second)
        orthogonal = compute_orthogonal_sum(first, second)
        assert direct.direct_sum_basis.entries == expected
        assert orthogonal.orthogonal_sum_basis == direct.direct_sum_basis
        assert (
            direct.direct_sum_basis.column_count
            == first.ambient_dimension + second.ambient_dimension
        )
        assert type(direct).model_validate_json(direct.model_dump_json()) == direct
        assert (
            type(orthogonal).model_validate_json(orthogonal.model_dump_json())
            == orthogonal
        )


def test_zero_sublattice_inclusion_retains_parent_rank() -> None:
    zero = IntegerLattice(
        ambient_dimension=2,
        basis=IntegerMatrix(entries=(), row_count=0, column_count=2),
    )
    parent = IntegerLattice(ambient_dimension=2, basis=IntegerMatrix(entries=((2, 0),)))
    embedding = IntegerMatrix(entries=(), row_count=0, column_count=1)
    result = compute_sublattice_index(zero, parent, embedding)
    assert result.index == "INFINITE"
    assert result.free_rank == 1
    assert result.invariant_factors == ()
    assert type(result).model_validate_json(result.model_dump_json()) == result
    bad_embedding = IntegerMatrix(entries=(), row_count=0, column_count=0)
    with pytest.raises(ValueError, match="embedding columns"):
        SublatticeIndexRequest(sublattice=zero, parent=parent, embedding=bad_embedding)
    with pytest.raises(OperationDomainValidationError, match="embedding dimensions"):
        compute_sublattice_index(zero, parent, bad_embedding)
    trivial = compute_sublattice_index(zero, zero, bad_embedding)
    assert trivial.index == 1
    assert trivial.free_rank == 0
