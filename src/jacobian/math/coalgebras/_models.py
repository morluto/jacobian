"""Typed wire contracts for coalgebra and Hopf algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DIM = 8

#: Group-like enumeration scans every element of GF(p)^dimension, so the
#: admitted request space is bounded by this many candidate vectors.
GROUP_LIKE_ENUMERATION_BUDGET = 65_536


class Coalgebra(StrictModel):
    """A finite-dimensional coalgebra over a prime field GF(p).

    The comultiplication is specified by structure constants:
    Delta(c_i) = sum_{j,k} d_{i}^{jk} * c_j �otimes c_k
    The counit is epsilon(c_i) = e_i.
    """

    prime: int = Field(ge=2, le=10_000)
    dimension: int = Field(ge=1, le=MAX_DIM)
    comultiplication: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=MAX_DIM
    )
    counit: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.comultiplication) != self.dimension:
            raise ValueError("comultiplication must have dimension entries")
        for row in self.comultiplication:
            if len(row) != self.dimension:
                raise ValueError("comultiplication entry must be dimension x dimension")
            for v in row:
                if len(v) != self.dimension:
                    raise ValueError("comultiplication tensor must be 3D")
        if len(self.counit) != self.dimension:
            raise ValueError("counit must have dimension entries")
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        self._require_coalgebra_axioms()
        return self

    def _require_coalgebra_axioms(self) -> None:
        """Validate coassociativity and both counit identities modulo p.

        Group-like conclusions presuppose a coalgebra: (Delta tensor id) o
        Delta = (id tensor Delta) o Delta, (epsilon tensor id) o Delta = id,
        and (id tensor epsilon) o Delta = id. Arbitrary linear maps do not
        satisfy these and must not be admitted.
        """
        p = self.prime
        d = self.comultiplication
        n = self.dimension
        e = self.counit

        # Coassociativity per basis element i:
        # (Delta tensor id) Delta(c_i) = (id tensor Delta) Delta(c_i)
        # Coefficient of c_j tensor c_k tensor c_ell:
        #   sum_t d[i][t][ell] * d[t][j][k] == sum_t d[i][j][t] * d[t][k][ell]  (mod p)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for ell in range(n):
                        left = sum(d[i][t][ell] * d[t][j][k] for t in range(n)) % p
                        right = sum(d[i][j][t] * d[t][k][ell] for t in range(n)) % p
                        if left != right:
                            raise ValueError(
                                "comultiplication must be coassociative"
                            )

        # Counit identities: both (epsilon tensor id)Delta = id and
        # (id tensor epsilon)Delta = id must hold.
        #   sum_t e[t] * d[i][t][j] == delta_{i,j}   ((epsilon tensor id))
        #   sum_t e[t] * d[i][j][t] == delta_{i,j}   ((id tensor epsilon))
        for i in range(n):
            for j in range(n):
                left_counit = sum(e[t] * d[i][t][j] for t in range(n)) % p
                right_counit = sum(e[t] * d[i][j][t] for t in range(n)) % p
                expected = 1 if i == j else 0
                if left_counit != expected or right_counit != expected:
                    raise ValueError(
                        "counit identities (epsilon x id)Delta = id and "
                        "(id x epsilon)Delta = id must hold modulo p"
                    )


class ComultiplicationRequest(StrictModel):
    """Compute the comultiplication Delta applied to a basis element."""

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.element_index >= self.coalgebra.dimension:
            raise ValueError("element_index must be in 0..dimension-1")
        return self


class ComultiplicationResult(StrictModel):
    """The comultiplication Delta(c_i) as a matrix of coefficients."""

    element_index: int = Field(ge=0)
    coefficients: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)


class CounitRequest(StrictModel):
    """Compute the counit epsilon applied to a basis element."""

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.element_index >= self.coalgebra.dimension:
            raise ValueError("element_index must be in 0..dimension-1")
        return self


class CounitResult(StrictModel):
    """The counit value epsilon(c_i)."""

    element_index: int = Field(ge=0)
    value: int


class GroupLikeElementsRequest(StrictModel):
    """Find all group-like elements in a coalgebra.

    The operation enumerates every element of GF(p)^dimension, so requests
    are admitted only when prime**dimension is within the documented
    enumeration budget.
    """

    coalgebra: Coalgebra

    @model_validator(mode="after")
    def require_enumerable(self) -> Self:
        if (
            self.coalgebra.prime ** self.coalgebra.dimension
            > GROUP_LIKE_ENUMERATION_BUDGET
        ):
            raise ValueError(
                "group-like enumeration requires prime**dimension <= "
                f"{GROUP_LIKE_ENUMERATION_BUDGET}"
            )
        return self


class GroupLikeElement(StrictModel):
    """One group-like element with its coefficients."""

    coefficients: tuple[int, ...]


class GroupLikeElementsResult(StrictModel):
    """All group-like elements of a coalgebra."""

    elements: tuple[GroupLikeElement, ...]
    count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if self.count != len(self.elements):
            raise ValueError("count must match element count")
        return self


__all__ = [
    "Coalgebra",
    "ComultiplicationRequest",
    "ComultiplicationResult",
    "CounitRequest",
    "CounitResult",
    "GroupLikeElement",
    "GroupLikeElementsRequest",
    "GroupLikeElementsResult",
]
