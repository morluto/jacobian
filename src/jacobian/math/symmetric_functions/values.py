"""Canonical partition and Young-tableau values."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_PARTITION_SIZE = 500
# A positive partition of size at most N has at most N parts, so admitting one
# part per unit of size keeps the canonical domain closed under Ferrers
# transpose (conjugation) and under tableau shape construction.
MAX_PARTITION_PARTS = MAX_PARTITION_SIZE
# Positive-integer tableau labels are mathematical values, not cell indices.
# Keep them exactly interoperable as JSON numbers while bounding comparison and
# serialized-output work independently of the tableau cell-count envelope.
MAX_TABLEAU_ENTRY = 2**53 - 1

TableauEntry = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        le=MAX_TABLEAU_ENTRY,
        description=(
            "Positive JSON-safe integer tableau label; its magnitude is "
            "independent of the tableau cell count."
        ),
    ),
]
TableauRow = Annotated[
    tuple[TableauEntry, ...],
    Field(min_length=1, max_length=MAX_PARTITION_SIZE),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by symmetric-function values."""

    return PydanticCustomError(f"symmetric_function.{reason}", message)


class IntegerPartition(StrictModel):
    """A partition as a weakly decreasing tuple of positive integers.

    There are at most 500 parts and their sum is at most 500, so the domain is
    closed under conjugation.  The empty tuple is the unique partition of
    zero.
    """

    parts: tuple[StrictInt, ...] = Field(
        min_length=0,
        max_length=MAX_PARTITION_PARTS,
        description=(
            "Positive weakly-decreasing parts with a total size (sum) of at "
            f"most {MAX_PARTITION_SIZE}; at most {MAX_PARTITION_PARTS} parts."
        ),
    )

    @model_validator(mode="after")
    def require_valid_partition(self) -> Self:
        if not self.parts:
            return self
        if any(part <= 0 for part in self.parts):
            raise _validation_error(
                "partition_parts_not_positive", "partition parts must be positive"
            )
        if any(
            self.parts[index] < self.parts[index + 1]
            for index in range(len(self.parts) - 1)
        ):
            raise _validation_error(
                "partition_not_weakly_decreasing",
                "partition parts must be weakly decreasing",
            )
        if sum(self.parts) > MAX_PARTITION_SIZE:
            raise _validation_error(
                "partition_size_exceeded", "partition size exceeds the supported bound"
            )
        return self


def _shape(rows: tuple[TableauRow, ...]) -> IntegerPartition:
    return IntegerPartition(parts=tuple(len(row) for row in rows))


def _require_strict_columns(rows: tuple[TableauRow, ...]) -> None:
    for row_index in range(len(rows) - 1):
        upper = rows[row_index]
        lower = rows[row_index + 1]
        for column, lower_entry in enumerate(lower):
            if upper[column] >= lower_entry:
                raise _validation_error(
                    "tableau_columns_not_strict",
                    "tableau columns must be strictly increasing",
                )


class SemistandardYoungTableau(StrictModel):
    """A bounded semistandard tableau over positive integer entries.

    Rows are weakly increasing, columns are strictly increasing, and row
    lengths form an ``IntegerPartition``.  The empty tableau is represented by
    ``rows=()``.
    """

    rows: tuple[TableauRow, ...] = Field(
        max_length=MAX_PARTITION_PARTS,
        description=(
            "Weakly increasing nonempty rows of positive integers; row lengths "
            "form a partition, columns increase strictly, and the total cell "
            f"count is at most {MAX_PARTITION_SIZE}."
        ),
    )

    @model_validator(mode="after")
    def require_semistandard(self) -> Self:
        _shape(self.rows)
        for row in self.rows:
            if any(row[index] > row[index + 1] for index in range(len(row) - 1)):
                raise _validation_error(
                    "semistandard_rows_not_weakly_increasing",
                    "semistandard tableau rows must be weakly increasing",
                )
        _require_strict_columns(self.rows)
        return self

    @property
    def shape(self) -> IntegerPartition:
        """Return the tableau shape derived from its row lengths."""
        return _shape(self.rows)


class StandardYoungTableau(StrictModel):
    """A bounded standard tableau with entries exactly ``1, ..., n``.

    Rows and columns are strictly increasing.  The empty tableau is represented
    by ``rows=()`` and has size zero.
    """

    rows: tuple[TableauRow, ...] = Field(
        max_length=MAX_PARTITION_PARTS,
        description=(
            "Strictly increasing nonempty rows whose lengths form a partition; "
            "columns increase strictly and entries are exactly 1 through the "
            f"cell count, which is at most {MAX_PARTITION_SIZE}."
        ),
    )

    @model_validator(mode="after")
    def require_standard(self) -> Self:
        shape = _shape(self.rows)
        for row in self.rows:
            if any(row[index] >= row[index + 1] for index in range(len(row) - 1)):
                raise _validation_error(
                    "standard_rows_not_strictly_increasing",
                    "standard tableau rows must be strictly increasing",
                )
        _require_strict_columns(self.rows)
        entries = sorted(entry for row in self.rows for entry in row)
        if entries != list(range(1, sum(shape.parts) + 1)):
            raise _validation_error(
                "standard_entries_not_consecutive",
                "standard tableau entries must be exactly 1 through n",
            )
        return self

    @property
    def shape(self) -> IntegerPartition:
        """Return the tableau shape derived from its row lengths."""
        return _shape(self.rows)


__all__ = [
    "MAX_PARTITION_PARTS",
    "MAX_PARTITION_SIZE",
    "MAX_TABLEAU_ENTRY",
    "IntegerPartition",
    "SemistandardYoungTableau",
    "StandardYoungTableau",
    "TableauEntry",
    "TableauRow",
]
