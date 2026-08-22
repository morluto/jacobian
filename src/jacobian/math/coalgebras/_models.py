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
        self._require_canonical_residues()
        self._require_coalgebra_axioms()
        return self

    def _require_canonical_residues(self) -> None:
        """Reject noncanonical entries: each must already lie in 0..p-1 to
        avoid implicit field coercion and nonunique serialized identities."""
        for row in self.comultiplication:
            for v in row:
                for value in v:
                    if not 0 <= value < self.prime:
                        raise ValueError(
                            "structure constants must be canonical residues "
                            f"in 0..{self.prime - 1}"
                        )
        for value in self.counit:
            if not 0 <= value < self.prime:
                raise ValueError(
                    "counit entries must be canonical residues in "
                    f"0..{self.prime - 1}"
                )

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

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)
    coefficients: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)

    @model_validator(mode="after")
    def bind_comultiplication_to_source(self) -> Self:
        """Replay Delta(c_i) against the retained canonical coalgebra."""
        ca = self.coalgebra
        n = ca.dimension
        p = ca.prime
        if self.element_index >= n:
            raise ValueError("element_index must be in 0..dimension-1")
        if self.dimension != n:
            raise ValueError("dimension must match the retained coalgebra")
        expected = tuple(
            tuple(ca.comultiplication[self.element_index][j][k] % p for k in range(n))
            for j in range(n)
        )
        if self.coefficients != expected:
            raise ValueError(
                "coefficients must be the exact comultiplication of the "
                "retained coalgebra basis element"
            )
        return self


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

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)
    value: int

    @model_validator(mode="after")
    def bind_counit_to_source(self) -> Self:
        """Replay epsilon(c_i) against the retained canonical coalgebra."""
        ca = self.coalgebra
        if self.element_index >= ca.dimension:
            raise ValueError("element_index must be in 0..dimension-1")
        if self.value != ca.counit[self.element_index] % ca.prime:
            raise ValueError(
                "value must be the exact counit of the retained coalgebra "
                "basis element"
            )
        return self


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

    coalgebra: Coalgebra
    elements: tuple[GroupLikeElement, ...]
    count: int = Field(ge=0)

    @model_validator(mode="after")
    def bind_group_like_to_source(self) -> Self:
        """Replay the exhaustive enumeration against the retained coalgebra.

        The request model bounds prime**dimension within the documented
        enumeration budget, so replaying the defining relations
        Delta(g) = g (x) g and epsilon(g) = 1 over the whole element space is
        deterministic bounded work; the retained conclusion must be exactly
        that enumeration.
        """
        from jacobian.math.coalgebras._operations import _group_like_coefficients

        if self.count != len(self.elements):
            raise ValueError("count must match element count")
        # Reapply the enumeration admission bound before replaying: when a
        # serialized result is validated directly, its coalgebra is parsed as
        # a Coalgebra, not as a GroupLikeElementsRequest, so the request
        # guard alone never runs.
        if (
            self.coalgebra.prime ** self.coalgebra.dimension
            > GROUP_LIKE_ENUMERATION_BUDGET
        ):
            raise ValueError(
                "group-like enumeration requires prime**dimension <= "
                f"{GROUP_LIKE_ENUMERATION_BUDGET}"
            )
        n = self.coalgebra.dimension
        seen = set()
        for element in self.elements:
            if len(element.coefficients) != n:
                raise ValueError(
                    "element coefficients must match the coalgebra dimension"
                )
            key = tuple(element.coefficients)
            if key in seen:
                raise ValueError("group-like elements must be distinct")
            seen.add(key)
        expected = _group_like_coefficients(self.coalgebra)
        if tuple(element.coefficients for element in self.elements) != expected:
            raise ValueError(
                "elements must be the exact group-like set of the retained "
                "coalgebra"
            )
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
