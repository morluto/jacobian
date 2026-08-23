"""Typed wire contracts for Hochschild complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.hochschild_complexes._bar import bar_differential_entries
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
)

MAX_ALGEBRA_DIM = 8
MAX_MODULE_DIM = 8
MAX_HOCHSCHILD_DEGREE = 4
MAX_HOCHSCHILD_TENSOR_ELEMENTS = 20_000
# A dense boundary matrix d_k holds n^(k-1) * n^k entries; Gaussian elimination
# copies it and performs O(pivots * entries) field work. The entry budget keeps
# both the matrix and its elimination copy small alongside the tensor budget.
MAX_HOCHSCHILD_MATRIX_ENTRIES = 131_072


class AlgebraStructure(StrictModel):
    """A finite-dimensional associative algebra over GF(p) with an augmentation.

    The multiplication is specified by structure constants:
    e_i * e_j = sum_k c_{ij}^k * e_k. The augmentation epsilon maps each basis
    element to a GF(p) scalar and must be an algebra homomorphism,
    ``epsilon(e_i e_j) == epsilon(e_i) * epsilon(e_j) mod p``; it defines the
    trivial coefficient module K on which A acts through epsilon.
    """

    prime: int = Field(ge=2, le=10_000)
    dimension: int = Field(ge=1, le=MAX_ALGEBRA_DIM)
    structure_constants: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=MAX_ALGEBRA_DIM
    )
    augmentation: tuple[int, ...] = Field(min_length=1, max_length=MAX_ALGEBRA_DIM)

    def _require_canonical_residues(self) -> None:
        """Reject noncanonical constants and augmentation values: each must
        already lie in 0..p-1 to avoid implicit field coercion."""
        prime = self.prime
        dimension = self.dimension
        for i in range(dimension):
            for j in range(dimension):
                if any(not 0 <= c < prime for c in self.structure_constants[i][j]):
                    raise ValueError(
                        "structure constants must be canonical residues in 0..p-1"
                    )
        if any(not 0 <= value < prime for value in self.augmentation):
            raise ValueError("augmentation values must be canonical residues in 0..p-1")

    def _require_multiplicative_augmentation(self) -> None:
        """The trivial module via epsilon is an A-bimodule only when epsilon is
        multiplicative; this is exactly what makes the Hochschild differential
        square to zero."""
        prime = self.prime
        epsilon = self.augmentation
        constants = self.structure_constants
        dimension = self.dimension
        for i in range(dimension):
            for j in range(dimension):
                product_epsilon = (
                    sum(
                        coefficient * epsilon[coefficient_index]
                        for coefficient_index, coefficient in enumerate(constants[i][j])
                    )
                    % prime
                )
                if product_epsilon != (epsilon[i] * epsilon[j]) % prime:
                    raise ValueError(
                        "augmentation must be an algebra homomorphism: "
                        "epsilon(e_i e_j) must equal epsilon(e_i)*epsilon(e_j) mod p"
                    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.structure_constants) != self.dimension:
            raise ValueError("structure_constants must be dimension x dimension")
        for row in self.structure_constants:
            if len(row) != self.dimension:
                raise ValueError("structure_constants must be square")
            for v in row:
                if len(v) != self.dimension:
                    raise ValueError("structure_constants must be 3D")
        if len(self.augmentation) != self.dimension:
            raise ValueError("augmentation must have one entry per basis element")
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        self._require_canonical_residues()
        self._require_associative()
        self._require_multiplicative_augmentation()
        return self

    def _require_associative(self) -> None:
        """Hochschild differentials square to zero only over associative algebras."""
        p = self.prime
        c = self.structure_constants
        n = self.dimension
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for ell in range(n):
                        left = sum(c[i][j][t] * c[t][k][ell] for t in range(n)) % p
                        right = sum(c[j][k][t] * c[i][t][ell] for t in range(n)) % p
                        if left != right:
                            raise ValueError(
                                "multiplication must be associative modulo p"
                            )


class HochschildChainComplexRequest(StrictModel):
    """Compute the Hochschild chain complex with trivial coefficients.

    The coefficient module is ``K = GF(p)`` on which A acts through the
    retained augmentation epsilon. The chain groups are C_n = A^tensor-n and
    the differential is the full Hochschild boundary: interior adjacent
    multiplications plus the two augmentation-dependent endpoint faces
    (epsilon(a_1) a_2 ox ... and (-1)^n epsilon(a_n) a_1 ox ...). It squares
    to zero because epsilon is an algebra homomorphism and the multiplication
    is associative. The operation computes exact HH(A, K) chains; no cyclic
    wraparound beyond the Hochschild endpoint face is applied.
    """

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)

    @model_validator(mode="after")
    def require_within_budget(self) -> Self:
        dimension = self.algebra.dimension
        if dimension ** (self.max_degree + 1) > MAX_HOCHSCHILD_TENSOR_ELEMENTS:
            raise ValueError(
                "requested max_degree exceeds the supported tensor-element budget "
                f"(dimension^{self.max_degree + 1} > {MAX_HOCHSCHILD_TENSOR_ELEMENTS})"
            )
        densest_entries = dimension ** (2 * self.max_degree - 1)
        if densest_entries > MAX_HOCHSCHILD_MATRIX_ENTRIES:
            raise ValueError(
                "requested max_degree exceeds the supported boundary-matrix "
                f"entry budget (dimension^(2*max_degree-1) = {densest_entries} "
                f"> {MAX_HOCHSCHILD_MATRIX_ENTRIES})"
            )
        return self


class HochschildDifferential(StrictModel):
    """One Hochschild differential as the canonical prime-field matrix value.

    ``matrix`` is the domain-owned ``PrimeFieldMatrix`` carrying its source
    prime, entries, and declared column axis, so a serialized boundary feeds
    the GF(p) rank/RREF/nullspace consumers unchanged; ``degree`` stays
    separate chain-complex metadata.
    """

    degree: int = Field(ge=1)
    matrix: PrimeFieldMatrix

    @property
    def source_dim(self) -> int:
        return self.matrix.columns

    @property
    def target_dim(self) -> int:
        return len(self.matrix.entries)


class HochschildChainComplexResult(StrictModel):
    """The Hochschild chain complex prefix, bound to its source algebra."""

    algebra: AlgebraStructure
    algebra_dimension: int = Field(ge=1)
    group_dimensions: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_HOCHSCHILD_DEGREE + 1
    )
    differentials: tuple[HochschildDifferential, ...] = Field(
        min_length=0, max_length=MAX_HOCHSCHILD_DEGREE
    )
    prime: int = Field(ge=2, le=10_000)

    @model_validator(mode="after")
    def require_bound_to_algebra(self) -> Self:
        # Replay the exact derived value from the retained source so an authored
        # payload is never trusted on its shape alone.
        algebra = self.algebra
        if self.algebra_dimension != algebra.dimension or self.prime != algebra.prime:
            raise ValueError(
                "algebra_dimension and prime must match the retained algebra"
            )
        dimension = algebra.dimension
        expected_groups = tuple(
            [1] + [dimension**k for k in range(1, len(self.group_dimensions))]
        )
        if self.group_dimensions != expected_groups:
            raise ValueError(
                "group_dimensions must be the bar complex dimensions "
                "C_k = A^tensor-k of the retained algebra"
            )
        if len(self.differentials) != len(self.group_dimensions) - 1:
            raise ValueError("one differential per positive degree is required")
        for degree, differential in enumerate(self.differentials, start=1):
            if differential.degree != degree:
                raise ValueError("differential degrees must be consecutive from 1")
            if (
                differential.source_dim != dimension** degree
                or differential.target_dim != dimension ** (degree - 1)
            ):
                raise ValueError(
                    "differential dimensions must match the retained algebra"
                )
            if dimension ** (2 * degree - 1) > MAX_HOCHSCHILD_MATRIX_ENTRIES:
                raise ValueError(
                    "differential exceeds the supported boundary-matrix entry budget"
                )
            expected_entries = bar_differential_entries(
                algebra.structure_constants,
                algebra.prime,
                degree,
                algebra.augmentation,
            )
            if differential.matrix.entries != tuple(expected_entries):
                raise ValueError(
                    "differential entries must be the exact bar differential "
                    "of the retained algebra"
                )
        return self


class HochschildHomologyRequest(StrictModel):
    """Compute exact Hochschild homology HH_n(A, K) with trivial coefficients.

    K = GF(p) carries the trivial A-bimodule structure defined by the retained
    augmentation epsilon; the computation replays the full Hochschild boundary
    (interior multiplications plus both augmentation endpoint faces) and
    returns the exact Betti numbers of the retained augmented algebra.
    """

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)

    @model_validator(mode="after")
    def require_within_budget(self) -> Self:
        require_hochschild_budget(self.algebra.dimension, self.max_degree)
        return self


def require_hochschild_budget(dimension: int, max_degree: int) -> None:
    """Reject degrees whose tensor or boundary-matrix budget is exceeded."""

    if dimension ** (max_degree + 1) > MAX_HOCHSCHILD_TENSOR_ELEMENTS:
        raise ValueError(
            "requested max_degree exceeds the supported tensor-element budget "
            f"(dimension^{max_degree + 1} > {MAX_HOCHSCHILD_TENSOR_ELEMENTS})"
        )
    densest_entries = dimension ** (2 * max_degree + 1)
    if densest_entries > MAX_HOCHSCHILD_MATRIX_ENTRIES:
        raise ValueError(
            "requested max_degree exceeds the supported boundary-matrix "
            f"entry budget (dimension^(2*max_degree+1) = {densest_entries} "
            f"> {MAX_HOCHSCHILD_MATRIX_ENTRIES})"
        )


class HochschildHomologyGroup(StrictModel):
    """One Hochschild homology group."""

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)


class HochschildHomologyResult(StrictModel):
    """Hochschild homology groups with trivial coefficients."""

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)
    groups: tuple[HochschildHomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=10_000)

    @model_validator(mode="after")
    def bind_to_source_algebra(self) -> Self:
        from jacobian.math.hochschild_complexes._operations import (
            hochschild_homology_groups,
        )

        if self.prime != self.algebra.prime:
            raise ValueError("prime must match the retained algebra")
        # Reapply the enumeration admission bound before replaying so a
        # directly supplied payload cannot bypass the request budget.
        require_hochschild_budget(self.algebra.dimension, self.max_degree)
        expected_groups = hochschild_homology_groups(self.algebra, self.max_degree)
        if self.groups != expected_groups:
            raise ValueError(
                "groups must equal the exact bar-homology replay of the "
                "retained algebra"
            )
        return self


__all__ = [
    "AlgebraStructure",
    "HochschildChainComplexRequest",
    "HochschildChainComplexResult",
    "HochschildDifferential",
    "HochschildHomologyGroup",
    "HochschildHomologyRequest",
    "HochschildHomologyResult",
]
