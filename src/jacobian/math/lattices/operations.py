"""Exact lattice operations on canonical integer lattice values."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
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
    _MAX_LATTICE_INPUT_SCALAR_DIGITS,
    CanonicalBasisResult,
    DirectSumResult,
    DiscriminantGroupResult,
    DualResult,
    IntegerLattice,
    OrthogonalComplementResult,
    OrthogonalSumResult,
    RankGramResult,
    SaturationResult,
    SublatticeIndexResult,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    IntegerMatrix,
    rational_matrix_from_fractions,
    rational_vector_space_basis_from_fractions,
    require_matrix_scalar_digits,
)

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
    "hermite_normal_form",
    "reduce_basis",
    "verify_discriminant_group",
    "verify_rank_gram",
    "verify_sublattice_index",
]


def hermite_normal_form(entries: list[list[int]]) -> tuple[Any, Any]:
    """Return the row Hermite normal form and its left transformation."""

    from jacobian.math.lattices._hnf_backend import modular_row_hnf
    from jacobian.math.lattices._hnf_bounds import admit_hermite_normal_form

    admission = admit_hermite_normal_form(entries)
    return modular_row_hnf(entries, admission)


def reduce_basis(
    entries: list[list[int]],
    *,
    delta: float = 0.99,
    eta: float = 0.51,
) -> tuple[Any, Any, int]:
    """Reduce one integer lattice basis with exact-gram LLL.

    Accepts a row-major integer list-of-lists and returns the reduced basis,
    the left transformation, and the rank.  FLINT rejects one-row bases, so
    that mathematically valid boundary case is preserved with the identity
    transformation.
    """

    import flint

    source = flint.fmpz_mat(entries)
    if source.nrows() == 1:
        reduced = source
        transformation = flint.fmpz_mat([[1]])
    else:
        reduced, transformation = source.lll(
            True,
            delta,
            eta,
            "zbasis",
            "exact",
        )
    if transformation * source != reduced:
        raise ValueError("The LLL left transformation does not bind the source basis.")
    return reduced, transformation, int(reduced.rank())


def _basis_int_list(lattice: IntegerLattice) -> list[list[int]]:
    return [[parse_canonical_integer(v) for v in row] for row in lattice.basis.entries]


def _admit_lattice(lattice: IntegerLattice) -> list[list[int]]:
    """Admit the semantic full-row-rank claim once per operation."""

    basis = _basis_int_list(lattice)
    rows = len(basis)
    columns = lattice.ambient_dimension
    if rows * columns > MAX_MATRIX_DIMENSION**2:
        raise OperationDomainValidationError(
            location=("lattice", "basis"),
            code="lattice.rank_work_exceeds_bound",
            message="lattice rank admission exceeds the exact work bound",
        )
    if integer_rank(basis) != rows:
        raise OperationDomainValidationError(
            location=("lattice", "basis"),
            code="lattice.basis_not_full_rank",
            message="lattice basis must have full row rank over QQ",
        )
    return basis


def _admit_lattice_sum(
    first: IntegerLattice, second: IntegerLattice
) -> tuple[list[list[int]], list[list[int]]]:
    """Admit both operands and the combined block-diagonal envelope."""
    first_basis = _admit_lattice(first)
    second_basis = _admit_lattice(second)
    ambient_dimension = first.ambient_dimension + second.ambient_dimension
    rows = len(first_basis) + len(second_basis)
    if ambient_dimension > MAX_MATRIX_DIMENSION or rows * ambient_dimension > (
        MAX_MATRIX_DIMENSION**2
    ):
        raise OperationDomainValidationError(
            location=("lattice",),
            code="lattice.sum_work_exceeds_bound",
            message=(
                "lattice sum exceeds the exact output envelope: combined "
                f"ambient dimension {ambient_dimension} and {rows} basis rows"
            ),
        )
    return first_basis, second_basis


def _integer_matrix(matrix: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(
        entries=tuple(
            tuple(format_canonical_integer(int(v)) for v in row) for row in matrix
        )
    )


def compute_rank_gram(lattice: IntegerLattice) -> RankGramResult:
    """Compute exact rank, Gram matrix, and squared covolume."""

    basis = _admit_lattice(lattice)
    gram = _gram_matrix(basis)
    det = integer_determinant(gram)
    return RankGramResult(
        lattice=lattice,
        gram_matrix=_integer_matrix(gram),
        squared_covolume=det,
        covolume_rational=isqrt(det) ** 2 == det,
    )


def compute_canonical_basis(lattice: IntegerLattice) -> CanonicalBasisResult:
    """Compute the row-Hermite canonical basis of a lattice."""

    hnf, transform = _hermite_basis(_admit_lattice(lattice))
    return CanonicalBasisResult(
        lattice=lattice,
        canonical_basis=_integer_matrix(hnf),
        transformation=_integer_matrix(transform),
        rank=integer_rank(hnf),
    )


def compute_dual(lattice: IntegerLattice) -> DualResult:
    """Compute the exact rational dual basis and dual Gram matrix."""

    basis = _admit_lattice(lattice)
    dual = _dual_basis(basis)
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
        lattice=lattice,
        dual_basis=rational_matrix_from_fractions(dual),
        dual_gram=rational_matrix_from_fractions(dual_gram_fractions),
    )


def compute_saturation(lattice: IntegerLattice) -> SaturationResult:
    """Compute the primitive closure and its exact inclusion index."""

    saturated, inclusion, index = _saturate_lattice(_admit_lattice(lattice))
    return SaturationResult(
        lattice=lattice,
        saturated_basis=_integer_matrix(saturated),
        inclusion_transform=_integer_matrix(inclusion),
        saturation_index=index,
    )


def compute_sublattice_index(
    sublattice: IntegerLattice,
    parent: IntegerLattice,
    embedding: IntegerMatrix,
) -> SublatticeIndexResult:
    """Compute quotient index and Smith invariant factors."""

    require_matrix_scalar_digits(
        embedding.entries,
        maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
        label="sublattice embedding",
    )
    embedding_values = [
        [parse_canonical_integer(v) for v in row] for row in embedding.entries
    ]
    parent_values = _admit_lattice(parent)
    sublattice_values = _admit_lattice(sublattice)
    # The existing lattice axes and scalar bounds also bound this rectangular
    # product. Establish the caller's inclusion before using E's Smith form.
    if (
        len(embedding_values) != len(sublattice_values)
        or any(len(row) != len(parent_values) for row in embedding_values)
        or sublattice.ambient_dimension != parent.ambient_dimension
    ):
        raise OperationDomainValidationError(
            location=("embedding",),
            code="lattice.sublattice_embedding_mismatch",
            message="embedding dimensions must express sublattice = E @ parent",
        )
    if any(
        sum(row[k] * parent_values[k][j] for k in range(len(parent_values)))
        != sublattice_values[i][j]
        for i, row in enumerate(embedding_values)
        for j in range(parent.ambient_dimension)
    ):
        raise OperationDomainValidationError(
            location=("embedding",),
            code="lattice.sublattice_embedding_mismatch",
            message="embedding must satisfy sublattice = E @ parent exactly",
        )
    index, factors, free_rank = _sublattice_index(
        embedding_values,
        parent_rank=len(parent.basis.entries),
    )
    return SublatticeIndexResult(
        sublattice=sublattice,
        parent=parent,
        embedding=embedding,
        index=index,
        invariant_factors=tuple(factors),
        free_rank=free_rank,
    )


def compute_discriminant_group(lattice: IntegerLattice) -> DiscriminantGroupResult:
    """Compute the discriminant order and quotient invariant factors."""

    order, factors = _discriminant_group(_admit_lattice(lattice))
    return DiscriminantGroupResult(
        lattice=lattice,
        discriminant_order=order,
        invariant_factors=tuple(factors),
    )


def compute_orthogonal_complement(
    lattice: IntegerLattice,
) -> OrthogonalComplementResult:
    """Compute a canonical rational basis for the orthogonal complement."""

    complement = _orthogonal_complement(_admit_lattice(lattice))
    return OrthogonalComplementResult(
        lattice=lattice,
        complement_basis=rational_vector_space_basis_from_fractions(
            complement,
            ambient_dimension=lattice.ambient_dimension,
        ),
        complement_rank=len(complement),
    )


def compute_direct_sum(
    first: IntegerLattice, second: IntegerLattice
) -> DirectSumResult:
    """Compute the block-coordinate direct sum of two lattices."""

    first_basis, second_basis = _admit_lattice_sum(first, second)
    result = _direct_sum(first_basis, second_basis)
    return DirectSumResult(
        first=first,
        second=second,
        direct_sum_basis=_integer_matrix(result),
        ambient_dimension=first.ambient_dimension + second.ambient_dimension,
    )


def compute_orthogonal_sum(
    first: IntegerLattice, second: IntegerLattice
) -> OrthogonalSumResult:
    """Compute the block-diagonal orthogonal sum of two lattices."""

    first_basis, second_basis = _admit_lattice_sum(first, second)
    result = _orthogonal_sum(first_basis, second_basis)
    return OrthogonalSumResult(
        first=first,
        second=second,
        orthogonal_sum_basis=_integer_matrix(result),
        ambient_dimension=first.ambient_dimension + second.ambient_dimension,
    )


def verify_rank_gram(claim: RankGramResult) -> bool:
    """Check Gram entries, covolume and square claim after lattice admission."""
    return compute_rank_gram(claim.lattice) == claim


def verify_sublattice_index(claim: SublatticeIndexResult) -> bool:
    """Check the retained inclusion and its bounded Smith quotient invariants."""
    source = claim
    return (
        compute_sublattice_index(source.sublattice, source.parent, source.embedding)
        == claim
    )


def verify_discriminant_group(claim: DiscriminantGroupResult) -> bool:
    """Check discriminant order and factors against the retained lattice."""
    return compute_discriminant_group(claim.lattice) == claim
