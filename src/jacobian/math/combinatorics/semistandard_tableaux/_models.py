"""Wire contracts for bounded semistandard-tableau enumeration."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StrictInt, WithJsonSchema

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    TableauContent,
)

MAX_SEMISTANDARD_TABLEAUX = 4_096
MAX_ENUMERATED_CELLS = 100_000
MAX_ENUMERATION_WORK = 25_000_000
MAX_RESULT_BYTES = 2_000_000
MAX_TABLEAU_ALPHABET = 4_096
MAX_KOSTKA_SEARCH_WORK = 5_000_000
MAX_KOSTKA_RESULT_BYTES = 65_536

_NonnegativeExactInteger = Annotated[
    ExactInteger,
    Field(ge=0),
    WithJsonSchema(
        {
            "type": "string",
            "pattern": rf"^(?:0|[1-9][0-9]{{0,{MAX_CANONICAL_INTEGER_DIGITS - 1}}})(?![\s\S])",
            "maxLength": MAX_CANONICAL_INTEGER_DIGITS,
            "examples": ["0", "1"],
        }
    ),
]


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


class FixedContentCountRequest(StrictModel):
    """Count tableaux of a shape with one sparse, exact content map."""

    partition: IntegerPartition
    content: TableauContent


class FixedContentCountResult(StrictModel):
    """Exact fixed-content count bound to the input shape and content."""

    partition: IntegerPartition
    content: TableauContent
    count: _NonnegativeExactInteger


__all__ = [
    "MAX_ENUMERATED_CELLS",
    "MAX_ENUMERATION_WORK",
    "MAX_KOSTKA_RESULT_BYTES",
    "MAX_KOSTKA_SEARCH_WORK",
    "MAX_RESULT_BYTES",
    "MAX_SEMISTANDARD_TABLEAUX",
    "MAX_TABLEAU_ALPHABET",
    "FixedContentCountRequest",
    "FixedContentCountResult",
    "SemistandardTableauEnumerationRequest",
    "SemistandardTableauEnumerationResult",
]
