"""Typed wire contracts for Hochschild complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_ALGEBRA_DIM = 8
MAX_MODULE_DIM = 8
MAX_HOCHSCHILD_DEGREE = 4
MAX_HOCHSCHILD_TENSOR_ELEMENTS = 20_000


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
        # Reject noncanonical residues: each constant must already be in 0..p-1
        # to avoid implicit field coercion and unbounded input size.
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    c = self.structure_constants[i][j][k]
                    if not 0 <= c < self.prime:
                        raise ValueError(
                            "structure constants must be canonical residues in 0..p-1"
                        )
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
        # The homology computation needs d_{max_degree+1}, so check that degree.
        if self.algebra.dimension ** (self.max_degree + 1) > MAX_HOCHSCHILD_TENSOR_ELEMENTS:
            raise ValueError(
                "requested max_degree exceeds the supported tensor-element budget "
                f"(dimension^{self.max_degree+1} > {MAX_HOCHSCHILD_TENSOR_ELEMENTS})"
            )
        return self


class HochschildDifferential(StrictModel):
    """One Hochschild differential matrix."""

    degree: int = Field(ge=1)
    source_dim: int = Field(ge=1)
    target_dim: int = Field(ge=1)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=1)


class HochschildChainComplexResult(StrictModel):
    """The Hochschild chain complex prefix."""

    algebra_dimension: int = Field(ge=1)
    group_dimensions: tuple[int, ...] = Field(min_length=1)
    differentials: tuple[HochschildDifferential, ...] = Field(min_length=0)
    prime: int = Field(ge=2, le=10_000)

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        # group_dimensions length is max_degree + 1, not algebra_dimension + 1
        return self


class HochschildHomologyRequest(StrictModel):
    """Compute the reduced bar homology (trivial-coefficient Hochschild) of an algebra."""

    algebra: AlgebraStructure
    max_degree: int = Field(ge=1, le=MAX_HOCHSCHILD_DEGREE)

    @model_validator(mode="after")
    def require_within_budget(self) -> Self:
        if self.algebra.dimension ** (self.max_degree + 1) > MAX_HOCHSCHILD_TENSOR_ELEMENTS:
            raise ValueError(
                "requested max_degree exceeds the supported tensor-element budget "
                f"(dimension^{self.max_degree+1} > {MAX_HOCHSCHILD_TENSOR_ELEMENTS})"
            )
        return self


class HochschildHomologyGroup(StrictModel):
    """One Hochschild homology group."""

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)


class HochschildHomologyResult(StrictModel):
    """Hochschild homology groups with trivial coefficients."""

    groups: tuple[HochschildHomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=10_000)


__all__ = [
    "AlgebraStructure",
    "HochschildChainComplexRequest",
    "HochschildChainComplexResult",
    "HochschildDifferential",
    "HochschildHomologyGroup",
    "HochschildHomologyRequest",
    "HochschildHomologyResult",
]
