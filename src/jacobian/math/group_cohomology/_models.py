"""Typed wire contracts for group cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_GROUP_ORDER = 64
MAX_COCHAIN_DEGREE = 6
# The bar differential over C^n = GF(p)^{|G|^n} is materialized densely;
# bounding |G|^{max_degree+1} bounds every allocated matrix exactly.
MAX_COCHAIN_TENSOR_ELEMENTS = 4_096


class PermutationGroup(StrictModel):
    """A finite permutation group given by generator permutations."""

    degree: int = Field(ge=1, le=16)
    generators: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy.combinatorics import Permutation, PermutationGroup

        for perm in self.generators:
            if len(perm) != self.degree:
                raise ValueError("each generator must have length equal to degree")
            if sorted(perm) != list(range(self.degree)):
                raise ValueError("each generator must be a permutation of 0..n-1")
        # The degree bound alone admits S16 whose enumeration is factorial;
        # the ENUMERATED order is what the kernel allocates over.
        group = PermutationGroup(*(Permutation(list(g)) for g in self.generators))
        if int(group.order()) > MAX_GROUP_ORDER:
            raise ValueError(
                f"enumerated group order exceeds the bounded maximum {MAX_GROUP_ORDER}"
            )
        return self


def _enumerated_group_order(group: PermutationGroup) -> int:
    from sympy.combinatorics import Permutation, PermutationGroup

    pg = PermutationGroup(*(Permutation(list(g)) for g in group.generators))
    return int(pg.order())


class GroupCohomologyRequest(StrictModel):
    """Compute group cohomology with trivial coefficients over GF(p)."""

    group: PermutationGroup
    prime: int = Field(ge=2, le=10_000)
    max_degree: int = Field(ge=0, le=MAX_COCHAIN_DEGREE)

    @model_validator(mode="after")
    def require_admissible_domain(self) -> Self:
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        order = _enumerated_group_order(self.group)
        if order ** (self.max_degree + 1) > MAX_COCHAIN_TENSOR_ELEMENTS:
            raise ValueError(
                "cochain dimensions |G|^n exceed the supported "
                f"{MAX_COCHAIN_TENSOR_ELEMENTS}-element budget; reduce the "
                "group order or max_degree"
            )
        return self


class CohomologyGroup(StrictModel):
    """One group cohomology group."""

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)
    dimension: int = Field(ge=1)


class GroupCohomologyResult(StrictModel):
    """Group cohomology groups with trivial coefficients."""

    groups: tuple[CohomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=10_000)
    group_order: int = Field(ge=1)

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        if not self.groups:
            raise ValueError("at least one cohomology group is required")
        return self


__all__ = [
    "CohomologyGroup",
    "GroupCohomologyRequest",
    "GroupCohomologyResult",
    "PermutationGroup",
]
