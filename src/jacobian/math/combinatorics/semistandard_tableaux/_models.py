"""Wire contracts for bounded semistandard-tableau enumeration."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)

MAX_SEMISTANDARD_TABLEAUX = 4_096
MAX_ENUMERATED_CELLS = 100_000
MAX_ENUMERATION_WORK = 25_000_000
MAX_RESULT_BYTES = 2_000_000
MAX_TABLEAU_ALPHABET = 4_096


class SemistandardTableauEnumerationRequest(StrictModel):
    """A straight partition shape and positive-label alphabet size."""

    partition: IntegerPartition
    max_entry: StrictInt = Field(ge=0, le=MAX_TABLEAU_ALPHABET)


class SemistandardTableauEnumerationResult(StrictModel):
    """All semistandard tableaux with entries in ``1..max_entry``."""

    partition: IntegerPartition
    max_entry: StrictInt = Field(ge=0, le=MAX_TABLEAU_ALPHABET)
    tableaux: tuple[SemistandardYoungTableau, ...] = Field(
        max_length=MAX_SEMISTANDARD_TABLEAUX
    )


__all__ = [
    "MAX_ENUMERATED_CELLS",
    "MAX_ENUMERATION_WORK",
    "MAX_RESULT_BYTES",
    "MAX_SEMISTANDARD_TABLEAUX",
    "MAX_TABLEAU_ALPHABET",
    "SemistandardTableauEnumerationRequest",
    "SemistandardTableauEnumerationResult",
]
