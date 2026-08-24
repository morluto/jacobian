"""Typed contracts for complete uniform rank-3 chirotope checks."""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel

MAX_GROUND_SIZE = 10
"""The B2 bound: ``10**6 == 1_000_000`` ordered triple pairs."""

MAX_CHIROTOPE_ENTRIES = 120
"""``binomial(MAX_GROUND_SIZE, 3)`` materialized increasing triples."""

MAX_B2_EXCHANGE_INSTANCES = MAX_GROUND_SIZE**6
"""The complete B2 negative-case work bound for one mathematical scan."""

SOURCE_BOUND_REPLAY_PASSES = 2
"""One producer scan plus mandatory result-validation replay."""

MAX_EXECUTION_B2_EXCHANGE_INSTANCES = (
    SOURCE_BOUND_REPLAY_PASSES * MAX_B2_EXCHANGE_INSTANCES
)
"""The work envelope charged by one public checker invocation."""


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
            "n <= 10 because it reserves two B2 scans of n^6 pairs: at most "
            "2,000,000 pair checks per public invocation."
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
            raise ValueError(
                "ground_size exceeds the one-scan B2 exchange work envelope"
            )
        expected = tuple(combinations(range(self.ground_size), 3))
        actual = tuple(entry.triple for entry in self.entries)
        if actual != expected:
            raise ValueError(
                "entries must be the complete lexicographic sequence of increasing "
                "triples for ground_size"
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
            raise ValueError("premise_products must match premise_factors")
        if self.conclusion_product != (
            self.conclusion_factors[0] * self.conclusion_factors[1]
        ):
            raise ValueError("conclusion_product must match conclusion_factors")
        return self


class ChirotopeCheckRequest(StrictModel):
    """Check the complete rank-3 B2 chirotope axiom of one canonical table."""

    chirotope: UniformRank3Chirotope


class ChirotopeCheckResult(StrictModel):
    """A source-bound exact validity result or deterministic first obstruction."""

    chirotope: UniformRank3Chirotope
    status: ChirotopeCheckStatus = Field(
        description=(
            "VALID after every B2 ordered-triple pair passes, or the first B2 "
            "obstruction in lexicographic order."
        )
    )
    b2_exchange_instances_checked: StrictInt = Field(
        ge=0,
        description=(
            "Pairs checked in one mathematical producer scan. The mandatory "
            "source-bound replay performs the same scan again but is not added "
            "to this mathematical count."
        ),
    )
    obstruction: B2Obstruction | None = None

    @model_validator(mode="after")
    def replay_complete_check(self) -> Self:
        from jacobian.math.oriented_matroids._operations import _expected_result

        expected = _expected_result(ChirotopeCheckRequest(chirotope=self.chirotope))
        if self != expected:
            raise ValueError(
                "result must be the exact axiom replay of the retained chirotope"
            )
        return self


__all__ = [
    "MAX_B2_EXCHANGE_INSTANCES",
    "MAX_CHIROTOPE_ENTRIES",
    "MAX_EXECUTION_B2_EXCHANGE_INSTANCES",
    "MAX_GROUND_SIZE",
    "SOURCE_BOUND_REPLAY_PASSES",
    "B2Obstruction",
    "ChirotopeCheckRequest",
    "ChirotopeCheckResult",
    "ChirotopeCheckStatus",
    "Rank3ChirotopeEntry",
    "UniformRank3Chirotope",
]
