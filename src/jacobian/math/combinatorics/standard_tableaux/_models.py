"""Wire contracts for bounded standard-tableau enumeration."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    StandardYoungTableau,
)

MAX_STANDARD_TABLEAUX = 4_096
MAX_ENUMERATED_CELLS = 100_000
MAX_CONSTRUCTION_WORK_CELLS = 25_000_000
MAX_RESULT_CELLS = 200_000


class StandardTableauEnumerationRequest(StrictModel):
    """The exact partition shape whose standard tableaux are enumerated."""

    partition: IntegerPartition


class StandardTableauEnumerationResult(StrictModel):
    """The complete lexicographically ordered family for one partition."""

    partition: IntegerPartition
    tableaux: tuple[StandardYoungTableau, ...] = Field(max_length=MAX_STANDARD_TABLEAUX)


__all__ = [
    "MAX_CONSTRUCTION_WORK_CELLS",
    "MAX_ENUMERATED_CELLS",
    "MAX_RESULT_CELLS",
    "MAX_STANDARD_TABLEAUX",
    "StandardTableauEnumerationRequest",
    "StandardTableauEnumerationResult",
]
