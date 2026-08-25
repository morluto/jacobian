"""Typed wire contracts for group cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.group._models import PermutationGroupRequest

MAX_GROUP_ORDER = 64
# Degree n allocates an ``order**(n+1)``-element cochain space C^n and a dense
# ``order**(n+1)``-by-``order**n`` coboundary, so these two coupled budgets
# bound every cochain vector, matrix allocation, and Gaussian-elimination
# workload (cells times pivot-row length) at every admitted degree.
MAX_COCHAIN_TENSOR_ELEMENTS = 4_096
MAX_BAR_MATRIX_CELLS = 65_536
# The admitted top degree is *derived* from those budgets per enumerated group
# order (see ``_admitted_max_degree``), not fixed here. This ceiling binds only
# when the coupled budgets cannot: for the order-1 group every cochain space,
# coboundary, tensor count, and cell count is identically one for any degree,
# so the kernel performs one O(1) rank computation per degree and emits one
# small record per degree. There this fallback caps exactly the result tuple
# length and the linear number of kernel steps; every group of order >= 2 is
# bounded far below it by the work-derived envelope.
MAX_COCHAIN_DEGREE = 64
"""Conservative fallback on max_degree bounding result size and kernel step
count when the coupled work budgets cannot bind (order-1 groups)."""

# Prime modulus: the kernel supports arbitrary primes but primality testing
# and each modular inverse pow(a, p-2, p) scale with digit length / log p,
# not the prime's numeric value alone. Bar matrices are at most
# 65_536 cells (cochain budget 4_096), so even the densest admitted
# Gaussian elimination performs at most ~2M field ops. Bounding
# p < 2**31 (10 decimal digits) keeps sympy.isprime deterministic and
# cheap and each inverse to <=31 modular multiplies, so predicted
# modular-arithmetic work stays bounded for every admitted matrix shape.
# This is a documented conservative fallback per AGENTS.md:157-164 and
# matches prime_field_matrix.MAX_PRIME: the semantic domain is all
# primes, the admitted envelope is digit-length / log p work, so
# p=10_007 for the trivial group (1x1 matrix) is admitted with
# identical work to p=9_973 rather than being rejected by a small
# hard ceiling on p.
MAX_PRIME = 2_147_483_647
"""Conservative 31-bit prime bound before primality testing; bounds digit
length and predicted modular-arithmetic work, not a mathematical restriction
to small primes."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by group-cohomology contracts."""

    return PydanticCustomError(f"group_cohomology.{reason}", message)


def _enumerated_group_order(group: PermutationGroupRequest) -> int:
    from sympy.combinatorics import Permutation, PermutationGroup

    pg = PermutationGroup(*(Permutation(list(g)) for g in group.generators))
    return int(pg.order())


def _admitted_max_degree(order: int) -> int:
    """Largest degree whose predicted work stays inside the coupled budgets.

    Derived from the actual work the kernel performs, not a fixed cap: at
    degree n it allocates an ``order**(n+1)``-element cochain space and an
    ``order**(2n+1)``-cell dense coboundary matrix, so a top degree d is
    admissible exactly when both quantities stay within their budgets for
    every n <= d. The exact integer search terminates in O(log order) steps
    because both quantities grow geometrically. For ``order == 1`` every
    quantity is identically one at any degree, so the budgets can never bind
    and only the conservative fallback ``MAX_COCHAIN_DEGREE`` applies.
    """
    if order == 1:
        return MAX_COCHAIN_DEGREE
    degree = 0
    while (
        order ** (degree + 1) <= MAX_COCHAIN_TENSOR_ELEMENTS
        and order ** (2 * degree + 1) <= MAX_BAR_MATRIX_CELLS
    ):
        degree += 1
    return degree - 1


class GroupCohomologyRequest(StrictModel):
    """Compute group cohomology with trivial coefficients over GF(p).

    ``max_degree`` is admitted against the work the kernel actually performs:
    the largest degree whose cochain spaces and dense bar coboundaries stay
    within the coupled tensor/cell budgets for this group's enumerated order,
    or the conservative fallback ceiling for the degenerate order-1 group.
    """

    group: PermutationGroupRequest = Field(
        description=(
            "Permutation group as the canonical group-domain value "
            "(degree + generators); pass a previous stabilizer result's "
            "`stabilizer` or `source` group unchanged."
        )
    )
    prime: int = Field(ge=2, le=MAX_PRIME)
    max_degree: int = Field(ge=0, le=MAX_COCHAIN_DEGREE)

    @model_validator(mode="after")
    def require_admissible_domain(self) -> Self:
        from sympy import isprime

        if not isprime(self.prime):
            raise _validation_error("prime_not_prime", "prime must be a prime integer")
        order = _enumerated_group_order(self.group)
        if order > MAX_GROUP_ORDER:
            raise _validation_error(
                "group_order_exceeds_bound",
                f"enumerated group order {order} exceeds the bounded maximum "
                f"{MAX_GROUP_ORDER}",
            )
        admitted_degree = _admitted_max_degree(order)
        if self.max_degree > admitted_degree:
            raise _validation_error(
                "degree_exceeds_work_budget",
                f"max_degree {self.max_degree} exceeds the work-derived "
                f"degree budget {admitted_degree} for enumerated group order "
                f"{order}: cochain spaces |G|^(n+1) must stay within the "
                f"{MAX_COCHAIN_TENSOR_ELEMENTS}-element budget and dense bar "
                f"coboundaries |G|^(2n+1) within the "
                f"{MAX_BAR_MATRIX_CELLS}-cell budget",
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
            raise _validation_error(
                "groups_empty", "at least one cohomology group is required"
            )
        if tuple(g.degree for g in self.groups) != tuple(
            range(self.request.max_degree + 1)
        ):
            raise _validation_error(
                "degrees_not_contiguous",
                "groups must cover degrees 0..max_degree exactly once in order",
            )
        replay_groups, replay_order = _cohomology_profile(self.request)
        if self.groups != replay_groups or self.group_order != replay_order:
            raise _validation_error(
                "groups_do_not_match_replay",
                "groups must be the exact cohomology of the retained source request",
            )
        return self


__all__ = [
    "MAX_BAR_MATRIX_CELLS",
    "MAX_COCHAIN_DEGREE",
    "MAX_COCHAIN_TENSOR_ELEMENTS",
    "MAX_GROUP_ORDER",
    "MAX_PRIME",
    "CohomologyGroup",
    "GroupCohomologyRequest",
    "GroupCohomologyResult",
]
