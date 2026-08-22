"""Typed wire contracts for Lie algebra homology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


def _require_prime(prime: int) -> None:
    from sympy import isprime

    if not isprime(prime):
        raise ValueError("prime must be a prime integer")


def _require_alternating(c, n: int, p: int) -> None:
    for i in range(n):
        if any(value % p != 0 for value in c[i][i]):
            raise ValueError(
                "structure constants must define an alternating bracket: "
                "[e_i, e_i] = 0"
            )


def _require_antisymmetric(c, n: int, p: int) -> None:
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if c[i][j][k] % p != (-c[j][i][k]) % p:
                    raise ValueError(
                        "structure constants must define an antisymmetric "
                        "bracket: [e_i, e_j] = -[e_j, e_i]"
                    )


def _require_jacobi(c, n: int, p: int) -> None:
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for component in range(n):
                    jacobi_sum = sum(
                        c[i][j][m] * c[m][k][component]
                        + c[j][k][m] * c[m][i][component]
                        + c[k][i][m] * c[m][j][component]
                        for m in range(n)
                    )
                    if jacobi_sum % p != 0:
                        raise ValueError(
                            "structure constants must satisfy the Jacobi identity"
                        )


class LieAlgebra(StrictModel):
    """A finite-dimensional Lie algebra over a prime field GF(p).

    The Lie bracket is specified by structure constants: for basis
    elements e_i, e_j, the bracket [e_i, e_j] = sum_k c_{ij}^k * e_k.
    The tensor must define a genuine Lie bracket: it is alternating,
    antisymmetric modulo p, and satisfies the Jacobi identity exactly;
    all three are established at this request boundary because the
    Chevalley-Eilenberg differential squares to zero only for such
    brackets.
    """

    prime: int = Field(ge=2, le=10_000)
    dimension: int = Field(ge=1, le=8)
    structure_constants: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def require_valid_structure(self) -> Self:
        n = self.dimension
        if len(self.structure_constants) != n:
            raise ValueError("structure_constants must be dimension x dimension x dimension")
        for i in range(n):
            if len(self.structure_constants[i]) != n:
                raise ValueError(
                    "each structure_constants[i] must have dimension rows"
                )
        for i in range(n):
            for j in range(n):
                if len(self.structure_constants[i][j]) != n:
                    raise ValueError("structure constant entry must have dimension components")

        p = self.prime
        c = self.structure_constants
        _require_prime(p)
        _require_alternating(c, n, p)
        _require_antisymmetric(c, n, p)
        _require_jacobi(c, n, p)
        return self


class ChevalleyEilenbergComplexRequest(StrictModel):
    """Compute the Chevalley-Eilenberg chain complex for a Lie algebra with trivial coefficients."""

    lie_algebra: LieAlgebra


class DifferentialMatrix(StrictModel):
    """One differential matrix in the Chevalley-Eilenberg complex."""

    degree: int = Field(ge=0)
    source_dim: int = Field(ge=1)
    target_dim: int = Field(ge=1)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=1)


class ChevalleyEilenbergComplexResult(StrictModel):
    """The Chevalley-Eilenberg chain complex with trivial coefficients."""

    dimension: int = Field(ge=1)
    group_dimensions: tuple[int, ...] = Field(min_length=1)
    differentials: tuple[DifferentialMatrix, ...] = Field(min_length=0)
    prime: int = Field(ge=2, le=10_000)

    @model_validator(mode="after")
    def require_consistent_dimensions(self) -> Self:
        if len(self.group_dimensions) != self.dimension + 1:
            raise ValueError("group_dimensions must have dimension+1 entries")
        return self


class LieHomologyRequest(StrictModel):
    """Compute Lie algebra homology with trivial coefficients."""

    lie_algebra: LieAlgebra


class LieHomologyGroup(StrictModel):
    """One Lie homology group."""

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)
    dimension: int = Field(ge=1)


class LieHomologyResult(StrictModel):
    """Lie algebra homology groups with trivial coefficients."""

    groups: tuple[LieHomologyGroup, ...] = Field(min_length=1)
    dimension: int = Field(ge=1)
    prime: int = Field(ge=2, le=10_000)

    @model_validator(mode="after")
    def require_consistent_groups(self) -> Self:
        if len(self.groups) != self.dimension + 1:
            raise ValueError("homology groups must cover degrees 0..dimension")
        return self


__all__ = [
    "ChevalleyEilenbergComplexRequest",
    "ChevalleyEilenbergComplexResult",
    "DifferentialMatrix",
    "LieAlgebra",
    "LieHomologyGroup",
    "LieHomologyRequest",
    "LieHomologyResult",
]
