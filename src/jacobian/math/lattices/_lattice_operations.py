"""Wire layer for the issue-#1739 integer-lattice structural operations.

Each public function accepts a typed request model (defined in ``_models``),
delegates to a pure kernel in ``_lattice_ops``, and returns a typed result
model.  All values are exact: integers are transported as canonical strings
and rationals as ``{num, den}`` pairs.
"""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.lattices._lattice_ops import (
    direct_sum as _direct_sum,
)
from jacobian.math.lattices._lattice_ops import (
    discriminant_group as _discriminant_group,
)
from jacobian.math.lattices._lattice_ops import (
    dual_basis as _dual_basis,
)
from jacobian.math.lattices._lattice_ops import (
    gram_matrix as _gram_matrix,
)
from jacobian.math.lattices._lattice_ops import (
    hermite_basis as _hermite_basis,
)
from jacobian.math.lattices._lattice_ops import (
    integer_determinant,
    integer_rank,
)
from jacobian.math.lattices._lattice_ops import (
    orthogonal_complement as _orthogonal_complement,
)
from jacobian.math.lattices._lattice_ops import (
    orthogonal_sum as _orthogonal_sum,
)
from jacobian.math.lattices._lattice_ops import (
    saturate_lattice as _saturate_lattice,
)
from jacobian.math.lattices._lattice_ops import (
    sublattice_index as _sublattice_index,
)
from jacobian.math.lattices._models import (
    CanonicalBasisResult,
    DirectSumRequest,
    DirectSumResult,
    DiscriminantGroupRequest,
    DiscriminantGroupResult,
    DualRequest,
    DualResult,
    IntegerLattice,
    OrthogonalComplementRequest,
    OrthogonalComplementResult,
    OrthogonalSumRequest,
    OrthogonalSumResult,
    RankGramRequest,
    RankGramResult,
    SaturationResult,
    SublatticeIndexRequest,
    SublatticeIndexResult,
)
from jacobian.math.matrices.values import IntegerMatrix, RationalMatrix

__all__ = [
    "compute_canonical_basis",
    "compute_direct_sum",
    "compute_discriminant_group",
    "compute_dual",
    "compute_orthogonal_complement",
    "compute_orthogonal_sum",
    "compute_rank_gram",
    "compute_saturation",
    "compute_sublattice_index",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basis_int_list(lattice: IntegerLattice) -> list[list[int]]:
    return [[parse_canonical_integer(v) for v in row] for row in lattice.basis.entries]


def _require_full_row_rank(lattice: IntegerLattice, *, label: str) -> list[list[int]]:
    entries = _basis_int_list(lattice)
    rows = len(entries)
    if integer_rank(entries) != rows:
        raise ValueError(f"{label} basis must be full row rank over QQ")
    return entries


def _integer_matrix(matrix: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(
        entries=tuple(
            tuple(format_canonical_integer(int(v)) for v in row) for row in matrix
        )
    )


def _rational_matrix(matrix: list[list[Fraction]]) -> RationalMatrix:
    from jacobian._exact import CanonicalRational

    return RationalMatrix(
        entries=tuple(
            tuple(CanonicalRational.from_fraction(frac) for frac in row)
            for row in matrix
        )
    )


def _rational_matrix_from_int(matrix: list[list[int]]) -> RationalMatrix:
    return _rational_matrix([[Fraction(v, 1) for v in row] for row in matrix])


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def compute_rank_gram(request: RankGramRequest) -> RankGramResult:
    basis = _require_full_row_rank(request.lattice, label="rank-gram")
    rank = len(basis)
    gram = _gram_matrix(basis)
    det = integer_determinant(gram)
    covolume_rational = bool(request.lattice.ambient_dimension != rank)
    return RankGramResult(
        rank=rank,
        ambient_dimension=request.lattice.ambient_dimension,
        gram_matrix=_integer_matrix(gram),
        squared_covolume=format_canonical_integer(det),
        covolume_rational=covolume_rational,
    )


def compute_canonical_basis(
    lattice: IntegerLattice,
) -> CanonicalBasisResult:
    basis = _require_full_row_rank(lattice, label="canonical-basis")
    hnf, transform = _hermite_basis(basis)
    rank = integer_rank(hnf)
    return CanonicalBasisResult(
        canonical_basis=_integer_matrix(hnf),
        transformation=_integer_matrix(transform),
        rank=rank,
    )


def compute_dual(request: DualRequest) -> DualResult:
    basis = _require_full_row_rank(request.lattice, label="dual")
    dual = _dual_basis(basis)
    # dual Gram = (B B^T)^{-1}
    from sympy import Matrix

    gram = Matrix(basis) * Matrix(basis).T
    dual_gram = gram.inv()
    dual_gram_fractions: list[list[Fraction]] = []
    for i in range(dual_gram.rows):
        row: list[Fraction] = []
        for j in range(dual_gram.cols):
            entry = dual_gram[i, j]
            if hasattr(entry, "p") and hasattr(entry, "q"):
                row.append(Fraction(int(entry.p), int(entry.q)))
            else:
                row.append(Fraction(int(entry), 1))
        dual_gram_fractions.append(row)
    return DualResult(
        dual_basis=_rational_matrix(dual),
        dual_gram=_rational_matrix(dual_gram_fractions),
    )


def compute_saturation(lattice: IntegerLattice) -> SaturationResult:
    basis = _require_full_row_rank(lattice, label="saturation")
    saturated, inclusion, index = _saturate_lattice(basis)
    return SaturationResult(
        saturated_basis=_integer_matrix(saturated),
        inclusion_transform=_integer_matrix(inclusion),
        saturation_index=index,
    )


def compute_sublattice_index(request: SublatticeIndexRequest) -> SublatticeIndexResult:
    sub_basis = _require_full_row_rank(request.sublattice, label="sublattice")
    parent_basis = _require_full_row_rank(request.parent, label="parent")
    del sub_basis, parent_basis
    embedding = [
        [parse_canonical_integer(v) for v in row] for row in request.embedding.entries
    ]
    parent_rank = len(request.parent.basis.entries)
    index, factors, free_rank = _sublattice_index(
        embedding,
        parent_rank=parent_rank,
    )
    return SublatticeIndexResult(
        index=index,
        invariant_factors=tuple(format_canonical_integer(f) for f in factors),
        free_rank=free_rank,
    )


def compute_discriminant_group(
    request: DiscriminantGroupRequest,
) -> DiscriminantGroupResult:
    basis = _require_full_row_rank(request.lattice, label="discriminant-group")
    order, factors = _discriminant_group(basis)
    return DiscriminantGroupResult(
        discriminant_order=order,
        invariant_factors=tuple(format_canonical_integer(f) for f in factors),
    )


def compute_orthogonal_complement(
    request: OrthogonalComplementRequest,
) -> OrthogonalComplementResult:
    basis = _require_full_row_rank(request.lattice, label="orthogonal-complement")
    complement = _orthogonal_complement(basis)
    if not complement:
        rank = 0
        complement_matrix = [[Fraction(0)]]
    else:
        rank = len(complement)
        complement_matrix = complement
    return OrthogonalComplementResult(
        complement_basis=_rational_matrix(complement_matrix),
        complement_rank=rank,
    )


def compute_direct_sum(request: DirectSumRequest) -> DirectSumResult:
    first = _require_full_row_rank(request.first, label="direct-sum first")
    second = _require_full_row_rank(request.second, label="direct-sum second")
    result = _direct_sum(first, second)
    ambient = request.first.ambient_dimension + request.second.ambient_dimension
    return DirectSumResult(
        direct_sum_basis=_integer_matrix(result),
        ambient_dimension=ambient,
    )


def compute_orthogonal_sum(request: OrthogonalSumRequest) -> OrthogonalSumResult:
    first = _require_full_row_rank(request.first, label="orthogonal-sum first")
    second = _require_full_row_rank(request.second, label="orthogonal-sum second")
    result = _orthogonal_sum(first, second)
    ambient = request.first.ambient_dimension + request.second.ambient_dimension
    return OrthogonalSumResult(
        orthogonal_sum_basis=_integer_matrix(result),
        ambient_dimension=ambient,
    )
