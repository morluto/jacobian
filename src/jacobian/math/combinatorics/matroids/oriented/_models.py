"""Typed contracts for complete uniform rank-3 chirotope checks."""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_GROUND_SIZE = 10
"""The B2 bound: ``10**6 == 1_000_000`` ordered triple pairs."""

MAX_CHIROTOPE_ENTRIES = 120
"""``binomial(MAX_GROUND_SIZE, 3)`` materialized increasing triples."""

MAX_B2_EXCHANGE_INSTANCES = MAX_GROUND_SIZE**6
"""The complete B2 negative-case work bound for one mathematical scan."""

MAX_EXECUTION_B2_EXCHANGE_INSTANCES = MAX_B2_EXCHANGE_INSTANCES
"""The one producer-scan work envelope charged by the public checker."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by oriented-matroid contracts."""

    return PydanticCustomError(f"oriented_matroid.{reason}", message)


class Rank3ChirotopeEntry(StrictModel):
    """One sign of a complete rank-3 table on an increasing triple."""

    triple: tuple[StrictInt, StrictInt, StrictInt] = Field(
        description=(
            "An increasing triple 0 <= i < j < k < ground_size. Entries must "
            "be the complete lexicographic list of such triples; ordered or "
            "nonalternating presentations are not admitted."
        )
    )
    sign: Literal[-1, 1] = Field(
        description="The nonzero chirotope sign on this increasing triple."
    )


class UniformRank3Chirotope(StrictModel):
    """A canonical, complete uniform rank-3 chirotope source table.

    Ground elements are the ordered indices ``0, ..., ground_size - 1``.  The
    table contains exactly one nonzero sign for each increasing triple, in
    lexicographic order. Values on all ordered triples are its alternating
    extension, so nonalternating or zero presentations are rejected at request
    admission rather than reported as checker results. It is materialized
    rather than an oracle or a generated table.
    """

    ground_size: StrictInt = Field(
        ge=3,
        le=MAX_GROUND_SIZE,
        description=(
            "Number n of indexed ground elements 0..n-1. The checker admits "
            "n <= 10 because it reserves one B2 scan of n^6 pairs: at most "
            "1,000,000 pair checks per public invocation."
        ),
    )
    entries: tuple[Rank3ChirotopeEntry, ...] = Field(
        min_length=1,
        max_length=MAX_CHIROTOPE_ENTRIES,
        description=(
            "The complete lexicographically ordered list of increasing triples; "
            "its length is exactly binomial(ground_size, 3)."
        ),
    )

    @model_validator(mode="after")
    def require_complete_canonical_table(self) -> Self:
        if self.ground_size**6 > MAX_B2_EXCHANGE_INSTANCES:
            raise _validation_error(
                "canonical_table.work_bound",
                "ground_size exceeds the one-scan B2 exchange work envelope",
            )
        expected = tuple(combinations(range(self.ground_size), 3))
        actual = tuple(entry.triple for entry in self.entries)
        if actual != expected:
            raise _validation_error(
                "canonical_table.entries",
                "entries must be the complete lexicographic sequence of increasing "
                "triples for ground_size",
            )
        return self


class ChirotopeCheckStatus(StrEnum):
    VALID = "VALID"
    B2_OBSTRUCTION = "B2_OBSTRUCTION"


class B2Obstruction(StrictModel):
    """One ordered rank-3 B2 exchange instance whose implication fails."""

    kind: Literal["B2"] = "B2"
    x: tuple[StrictInt, StrictInt, StrictInt]
    y: tuple[StrictInt, StrictInt, StrictInt]
    premise_factors: tuple[
        tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
        tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
        tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
    ]
    premise_products: tuple[StrictInt, StrictInt, StrictInt]
    conclusion_factors: tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]
    conclusion_product: StrictInt

    @model_validator(mode="after")
    def require_recorded_products(self) -> Self:
        if self.premise_products != tuple(
            left * right for left, right in self.premise_factors
        ):
            raise _validation_error(
                "b2_obstruction.premise_products",
                "premise_products must match premise_factors",
            )
        if self.conclusion_product != (
            self.conclusion_factors[0] * self.conclusion_factors[1]
        ):
            raise _validation_error(
                "b2_obstruction.conclusion_product",
                "conclusion_product must match conclusion_factors",
            )
        return self


class ChirotopeCheckRequest(StrictModel):
    """Check the complete rank-3 B2 chirotope axiom of one canonical table."""

    chirotope: UniformRank3Chirotope


class ChirotopeCheckResult(StrictModel):
    """A structurally bounded validity result or B2-obstruction claim.

    Kernel-produced results use :meth:`_from_kernel`; model validation
    deliberately does not execute the B2 enumeration.
    """

    chirotope: UniformRank3Chirotope
    status: ChirotopeCheckStatus = Field(
        description=(
            "VALID after every B2 ordered-triple pair passes, or the first B2 "
            "obstruction in lexicographic order."
        )
    )
    b2_exchange_instances_checked: StrictInt = Field(
        ge=0,
        le=MAX_B2_EXCHANGE_INSTANCES,
        description=("Pairs checked in the mathematical producer scan."),
    )
    obstruction: B2Obstruction | None = None

    @model_validator(mode="after")
    def require_bounded_result_shape(self) -> Self:
        if self.b2_exchange_instances_checked > self.chirotope.ground_size**6:
            raise _validation_error(
                "result.checked_instances",
                "b2_exchange_instances_checked exceeds the retained table's "
                "bounded B2 envelope",
            )
        if (self.status is ChirotopeCheckStatus.VALID) != (self.obstruction is None):
            raise _validation_error(
                "result.status_obstruction",
                "VALID results have no obstruction and B2_OBSTRUCTION results "
                "retain one obstruction",
            )
        if self.obstruction is not None and any(
            not 0 <= index < self.chirotope.ground_size
            for triple in (self.obstruction.x, self.obstruction.y)
            for index in triple
        ):
            raise _validation_error(
                "result.obstruction_indices",
                "obstruction triples must use indices in the retained ground set",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        chirotope: UniformRank3Chirotope,
        status: ChirotopeCheckStatus,
        b2_exchange_instances_checked: int,
        obstruction: B2Obstruction | None,
    ) -> Self:
        """Construct a result from the trusted bounded checker kernel."""

        return cls.model_construct(
            chirotope=chirotope,
            status=status,
            b2_exchange_instances_checked=b2_exchange_instances_checked,
            obstruction=obstruction,
        )


__all__ = [
    "MAX_B2_EXCHANGE_INSTANCES",
    "MAX_CHIROTOPE_ENTRIES",
    "MAX_EXECUTION_B2_EXCHANGE_INSTANCES",
    "MAX_GROUND_SIZE",
    "B2Obstruction",
    "ChirotopeCheckRequest",
    "ChirotopeCheckResult",
    "ChirotopeCheckStatus",
    "Rank3ChirotopeEntry",
    "UniformRank3Chirotope",
]
