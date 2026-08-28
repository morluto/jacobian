"""Typed wire contracts for group cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.groups._models import PermutationGroup

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


class GroupCohomologyRequest(StrictModel):
    """Compute group cohomology with trivial coefficients over GF(p).

    ``max_degree`` is admitted against the work the kernel actually performs:
    the largest degree whose cochain spaces and dense bar coboundaries stay
    within the coupled tensor/cell budgets for this group's enumerated order,
    or the conservative fallback ceiling for the degenerate order-1 group.
    """

    group: PermutationGroup = Field(
        description=(
            "Permutation group as the canonical group-domain value "
            "(degree + generators); pass a previous stabilizer result's "
            "`stabilizer` or `source` group unchanged."
        )
    )
    prime: int = Field(ge=2, le=MAX_PRIME)
    max_degree: int = Field(ge=0, le=MAX_COCHAIN_DEGREE)


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

    The source request and exact table stay structurally aligned here.
    Deserializing a result checks only this structural binding; the bar
    computation runs in the owner operation.
    """

    group: PermutationGroup
    prime: int = Field(ge=2, le=MAX_PRIME)
    max_degree: int = Field(ge=0, le=MAX_COCHAIN_DEGREE)
    groups: tuple[CohomologyGroup, ...] = Field(min_length=1)
    group_order: int = Field(ge=1)

    @model_validator(mode="after")
    def require_consistent(self) -> Self:
        if len(self.groups) != self.max_degree + 1 or tuple(
            group.degree for group in self.groups
        ) != tuple(range(self.max_degree + 1)):
            raise _validation_error(
                "degrees_not_contiguous",
                "groups must cover degrees 0..max_degree exactly once in order",
            )
        if self.group_order > MAX_GROUP_ORDER:
            raise _validation_error(
                "group_order_exceeds_bound",
                f"group_order must not exceed {MAX_GROUP_ORDER}",
            )
        for group in self.groups:
            expected_dimension = self.group_order**group.degree
            if group.cochain_dimension != expected_dimension:
                raise _validation_error(
                    "cochain_dimension",
                    "cochain_dimension must equal group_order**degree",
                )
            if group.betti > group.cochain_dimension:
                raise _validation_error(
                    "betti_bound", "betti cannot exceed cochain_dimension"
                )
        if self.groups[0].betti != 1:
            raise _validation_error("degree_zero_betti", "H^0 has betti one")
        return self

    @classmethod
    def _from_kernel(
        cls,
        group: PermutationGroup,
        prime: int,
        max_degree: int,
        groups: tuple[CohomologyGroup, ...],
        group_order: int,
    ) -> Self:
        """Construct a trusted result emitted by the owner-local kernel."""

        return cls.model_construct(
            group=group,
            prime=prime,
            max_degree=max_degree,
            groups=groups,
            group_order=group_order,
        )


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
