"""Typed wire contracts for Hochschild complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.hochschild_complexes._bar import bar_differential_entries

MAX_ALGEBRA_DIM = 8
MAX_MODULE_DIM = 8
MAX_HOCHSCHILD_DEGREE = 4
MAX_HOCHSCHILD_TENSOR_ELEMENTS = 20_000
# A dense boundary matrix d_k holds n^(k-1) * n^k entries; Gaussian elimination
# copies it and performs O(pivots * entries) field work. The entry budget keeps
# both the matrix and its elimination copy small alongside the tensor budget.
MAX_HOCHSCHILD_MATRIX_ENTRIES = 131_072


class AlgebraStructure(StrictModel):
    """A finite-dimensional unital associative algebra over GF(p).

    The multiplication is specified by structure constants:
    e_i * e_j = sum_k c_{ij}^k * e_k
    """

    prime: int = Field(ge=2, le=10_000)
    dimension: int = Field(ge=1, le=MAX_ALGEBRA_DIM)
    structure_constants: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=MAX_ALGEBRA_DIM
    )

    def _require_canonical_residues(self) -> None:
        """Reject noncanonical constants: each must already lie in 0..p-1 to
        avoid implicit field coercion and unbounded input size."""
        prime = self.prime
        dimension = self.dimension
        for i in range(dimension):
            for j in range(dimension):
                if any(not 0 <= c < prime for c in self.structure_constants[i][j]):
                    raise ValueError(
                        "structure constants must be canonical residues in 0..p-1"
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
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        self._require_canonical_residues()
        self._require_associative()
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
    """Compute the reduced bar (Hochschild) chain complex with trivial coefficients.

    With trivial bimodule action (K where a·k = 0 for non-unit basis and k·a = 0),
    the chain groups are C_n = A^⊗n and the differential is the bar differential
    b'(a1⊗...⊗an) = Σ_{i} (-1)^{i} ...⊗ a_i·a_{i+1} ⊗... (adjacent multiplications only).
    This is the normalized bar complex, which squares to zero for any associative
    algebra. For a unital augmented algebra, the usual Hochschild differential
    includes endpoint terms via the augmentation ε: the full b includes
    ε(a1)a2⊗... and ...⊗an-1 ε(an). Those terms are zero under the trivial
    (zero) action used here, so the operation is the bar complex with trivial
    coefficients. Rename or add an augmentation parameter to obtain the full
    Hochschild complex with non-zero bimodule actions.
    """

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)

    @model_validator(mode="after")
    def require_within_budget(self) -> Self:
        dimension = self.algebra.dimension
        if dimension ** (self.max_degree + 1) > MAX_HOCHSCHILD_TENSOR_ELEMENTS:
            raise ValueError(
                "requested max_degree exceeds the supported tensor-element budget "
                f"(dimension^{self.max_degree+1} > {MAX_HOCHSCHILD_TENSOR_ELEMENTS})"
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
    """One Hochschild differential matrix."""

    degree: int = Field(ge=1)
    source_dim: int = Field(ge=1)
    target_dim: int = Field(ge=1)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=1)


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
                differential.source_dim != dimension**degree
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
                algebra.structure_constants, algebra.prime, degree
            )
            if differential.entries != expected_entries:
                raise ValueError(
                    "differential entries must be the exact bar differential "
                    "of the retained algebra"
                )
        return self


class HochschildHomologyRequest(StrictModel):
    """Compute the reduced bar homology (trivial-coefficient Hochschild) of an algebra."""

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
