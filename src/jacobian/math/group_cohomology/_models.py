"""Typed wire contracts for group cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.group._models import PermutationGroupRequest

MAX_GROUP_ORDER = 64
MAX_COCHAIN_DEGREE = 6
# The bar differential over C^n = GF(p)^{|G|^n} is materialized densely;
# bounding |G|^{max_degree+1} bounds every cochain vector and matrix row count.
MAX_COCHAIN_TENSOR_ELEMENTS = 4_096
# Each coboundary delta^n: C^n -> C^{n+1} is a dense |G|^(n+1)-by-|G|^n matrix,
# so bounding |G|^(2*max_degree+1) bounds the full allocation and the Gaussian
# elimination work (cells times pivot-row length) for every degree at once.
MAX_BAR_MATRIX_CELLS = 65_536

# Canonical permutation-group value owned by the ``group`` domain.  Re-export
# under the local name for backward compatibility; new code should import
# ``PermutationGroupRequest`` directly.
PermutationGroup = PermutationGroupRequest


def _enumerated_group_order(group: PermutationGroupRequest) -> int:
    from sympy.combinatorics import Permutation, PermutationGroup

    pg = PermutationGroup(*(Permutation(list(g)) for g in group.generators))
    return int(pg.order())


class GroupCohomologyRequest(StrictModel):
    """Compute group cohomology with trivial coefficients over GF(p)."""

    group: PermutationGroupRequest = Field(
        description=(
            "Permutation group as the canonical group-domain value "
            "(degree + generators); pass a previous stabilizer result's "
            "`stabilizer` or `source` group unchanged."
        )
    )
    prime: int = Field(ge=2, le=10_000)
    max_degree: int = Field(ge=0, le=MAX_COCHAIN_DEGREE)

    @model_validator(mode="after")
    def require_admissible_domain(self) -> Self:
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        order = _enumerated_group_order(self.group)
        if order > MAX_GROUP_ORDER:
            raise ValueError(
                f"enumerated group order {order} exceeds the bounded maximum "
                f"{MAX_GROUP_ORDER}"
            )
        if order ** (self.max_degree + 1) > MAX_COCHAIN_TENSOR_ELEMENTS:
            raise ValueError(
                "cochain dimensions |G|^n exceed the supported "
                f"{MAX_COCHAIN_TENSOR_ELEMENTS}-element budget; reduce the "
                "group order or max_degree"
            )
        if order ** (2 * self.max_degree + 1) > MAX_BAR_MATRIX_CELLS:
            raise ValueError(
                "dense bar matrices |G|^(2n+1) exceed the supported "
                f"{MAX_BAR_MATRIX_CELLS}-cell budget; reduce the group "
                "order or max_degree"
            )
        return self


class CohomologyGroup(StrictModel):
    """One group cohomology group.

    ``betti`` is dim H^n(G, GF(p)); ``cochain_dimension`` is the dimension
    |G|^n of the ambient cochain space C^n, not the cohomology dimension.
    """

    degree: int = Field(ge=0)
    betti: int = Field(ge=0)
    cochain_dimension: int = Field(ge=1)


class GroupCohomologyResult(StrictModel):
    """Group cohomology groups with trivial coefficients.

    Retains the source request so validation replays the exact bar-complex
    relation instead of trusting an independently authored table.
    """

    request: GroupCohomologyRequest
    groups: tuple[CohomologyGroup, ...] = Field(min_length=1)
    group_order: int = Field(ge=1)

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        from jacobian.math.group_cohomology._operations import _cohomology_profile

        if not self.groups:
            raise ValueError("at least one cohomology group is required")
        if tuple(g.degree for g in self.groups) != tuple(
            range(self.request.max_degree + 1)
        ):
            raise ValueError(
                "groups must cover degrees 0..max_degree exactly once in order"
            )
        replay_groups, replay_order = _cohomology_profile(self.request)
        if self.groups != replay_groups or self.group_order != replay_order:
            raise ValueError(
                "groups must be the exact cohomology of the retained source request"
            )
        return self


__all__ = [
    "CohomologyGroup",
    "GroupCohomologyRequest",
    "GroupCohomologyResult",
    "PermutationGroup",
]
