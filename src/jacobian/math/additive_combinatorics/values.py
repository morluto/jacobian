"""Canonical values for bounded exact additive combinatorics."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic.json_schema import WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer

MAX_SUBSET_SUM_ITEMS = 4_095
MAX_INDEXED_INTEGER_SEQUENCE_ITEMS = 500_000
MAX_SUBSET_SUM_ITEM_DIGITS = 32_768
MAX_SUBSET_SUM_SUM_DIGITS = MAX_SUBSET_SUM_ITEM_DIGITS + len(str(MAX_SUBSET_SUM_ITEMS))
MAX_SUBSET_SUM_MULTIPLICITY_DIGITS = len(str(1 << MAX_SUBSET_SUM_ITEMS))
MAX_SUBSET_SUM_PROFILE_ENTRIES = 50_000

_CANONICAL_INTEGER_PATTERN = r"^(?:0|-?[1-9][0-9]*)$"


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)


IndexedInteger = Annotated[
    str,
    StringConstraints(
        pattern=_CANONICAL_INTEGER_PATTERN,
        max_length=MAX_SUBSET_SUM_ITEM_DIGITS + 1,
        strict=True,
    ),
]
SubsetSumInteger = Annotated[
    str,
    StringConstraints(
        pattern=_CANONICAL_INTEGER_PATTERN,
        max_length=MAX_SUBSET_SUM_SUM_DIGITS + 1,
        strict=True,
    ),
]
SubsetSumMultiplicity = Annotated[
    str,
    StringConstraints(
        pattern=_CANONICAL_INTEGER_PATTERN,
        max_length=MAX_SUBSET_SUM_MULTIPLICITY_DIGITS,
        strict=True,
    ),
]


class IndexedIntegerSequence(StrictModel):
    """A finite integer sequence whose positions are distinct selectable items.

    Equal values and zeros are intentionally retained as separate positions.
    The listed order is the stable index axis used by subset witnesses and
    other downstream indexed operations. The item ceiling is the widest
    consumer envelope; each consuming operation enforces its own tighter
    item-count and work bounds in request-level admission.
    """

    items: tuple[IndexedInteger, ...] = Field(
        max_length=MAX_INDEXED_INTEGER_SEQUENCE_ITEMS,
        description=(
            f"An ordered tuple of at most "
            f"{MAX_INDEXED_INTEGER_SEQUENCE_ITEMS:,} canonical integers. "
            "Repeated values and zeros remain distinct indexed items; each "
            f"integer has at most {MAX_SUBSET_SUM_ITEM_DIGITS:,} decimal "
            "digits, excluding its optional sign."
        ),
        examples=[("1", "1", "3")],
    )

    @model_validator(mode="after")
    def require_bounded_items(self) -> Self:
        for item in self.items:
            if len(item.lstrip("-")) > MAX_SUBSET_SUM_ITEM_DIGITS:
                raise _validation_error(
                    "require_bounded_items",
                    f"indexed integer exceeds the "
                    f"{MAX_SUBSET_SUM_ITEM_DIGITS:,}-digit source-item bound",
                )
        return self

    def as_int_tuple(self) -> tuple[int, ...]:
        """Return the indexed values as exact Python integers."""

        return tuple(parse_canonical_integer(item) for item in self.items)


def indexed_sequence_item_ceiling(
    maximum_items: int,
    *,
    maximum_item_digits: int | None = None,
) -> dict[str, Any]:
    """Return the shared sequence definition tightened to ``maximum_items``.

    The canonical sequence value advertises the widest consumer envelope, so a
    verbatim shared definition would make every consumer publish
    ``maxItems: 500000`` even when its validator admits fewer items. Consumers
    attach the returned schema through ``WithJsonSchema`` so their own schema
    documents advertise exactly the item ceiling their validators enforce;
    validation itself stays with the canonical value.

    ``maximum_item_digits`` additionally publishes a consumer-specific
    per-item digit ceiling: the item ``pattern`` encodes the absolute digit
    bound and ``maxLength`` keeps only its sign character, so schema-driven
    clients never submit an item that request admission would reject.
    """

    schema: dict[str, Any] = IndexedIntegerSequence.model_json_schema()
    schema.pop("$defs", None)
    items_schema = schema["properties"]["items"]
    items_schema["maxItems"] = maximum_items
    item_digits = (
        MAX_SUBSET_SUM_ITEM_DIGITS
        if maximum_item_digits is None
        else maximum_item_digits
    )
    items_schema["description"] = (
        f"An ordered tuple of at most {maximum_items:,} canonical integers. "
        "Repeated values and zeros remain distinct indexed items; each "
        f"integer has at most {item_digits:,} decimal digits, "
        "excluding its optional sign."
    )
    if maximum_item_digits is not None:
        raw_item_schema = items_schema["items"]
        raw_item_schema["maxLength"] = maximum_item_digits + 1
        raw_item_schema["pattern"] = (
            rf"^(?:0|-?[1-9][0-9]{{0,{maximum_item_digits - 1}}})$"
        )
    return schema


class SubsetSumProfileEntry(StrictModel):
    """One attainable sum and its positive indexed-subset multiplicity."""

    sum: SubsetSumInteger
    multiplicity: SubsetSumMultiplicity

    @model_validator(mode="after")
    def require_positive_multiplicity(self) -> Self:
        if len(self.sum.lstrip("-")) > MAX_SUBSET_SUM_SUM_DIGITS:
            raise _validation_error(
                "require_positive_multiplicity",
                "subset sum exceeds the derived source-sum digit bound",
            )
        if parse_canonical_integer(self.multiplicity) <= 0:
            raise _validation_error(
                "require_positive_multiplicity",
                "subset-sum multiplicity must be positive",
            )
        return self


class SubsetSumProfile(StrictModel):
    """The complete exact multiplicity profile of one indexed sequence.

    The empty subset is always included. Validation replays the complete
    bounded dynamic program, binding every row and multiplicity to ``source``.
    """

    source: Annotated[
        IndexedIntegerSequence,
        WithJsonSchema(indexed_sequence_item_ceiling(MAX_SUBSET_SUM_ITEMS)),
    ]
    entries: tuple[SubsetSumProfileEntry, ...] = Field(
        min_length=1,
        max_length=MAX_SUBSET_SUM_PROFILE_ENTRIES,
    )
    support_size: StrictInt = Field(ge=1, le=MAX_SUBSET_SUM_PROFILE_ENTRIES)
    total_subsets: SubsetSumMultiplicity

    @model_validator(mode="after")
    def bind_complete_profile(self) -> Self:
        sums = tuple(parse_canonical_integer(entry.sum) for entry in self.entries)
        if sums != tuple(sorted(sums)) or len(sums) != len(set(sums)):
            raise _validation_error(
                "bind_complete_profile",
                "subset-sum profile entries must have unique sorted sums",
            )
        if self.support_size != len(self.entries):
            raise _validation_error(
                "bind_complete_profile",
                "support_size must equal the number of profile entries",
            )

        expected_total = 1 << len(self.source.items)
        if parse_canonical_integer(self.total_subsets) != expected_total:
            raise _validation_error(
                "bind_complete_profile", "total_subsets must equal 2^len(source.items)"
            )

        from jacobian.math.additive_combinatorics.operations import (
            _subset_sum_profile_counts,
            _subset_sum_profile_envelope,
        )

        _subset_sum_profile_envelope(self.source)
        expected = tuple(sorted(_subset_sum_profile_counts(self.source).items()))
        actual = tuple(
            (
                parse_canonical_integer(entry.sum),
                parse_canonical_integer(entry.multiplicity),
            )
            for entry in self.entries
        )
        if actual != expected:
            raise _validation_error(
                "bind_complete_profile",
                "entries must be the complete exact subset-sum profile of source",
            )
        if sum(multiplicity for _, multiplicity in actual) != expected_total:
            raise _validation_error(
                "bind_complete_profile",
                "profile multiplicities must sum to total_subsets",
            )
        return self

    @classmethod
    def from_counts(
        cls,
        source: IndexedIntegerSequence,
        counts: dict[int, int],
    ) -> SubsetSumProfile:
        """Construct the canonical source-bound profile from exact counts."""

        entries = tuple(
            SubsetSumProfileEntry(
                sum=format_canonical_integer(subtotal),
                multiplicity=format_canonical_integer(count),
            )
            for subtotal, count in sorted(counts.items())
        )
        return cls(
            source=source,
            entries=entries,
            support_size=len(entries),
            total_subsets=format_canonical_integer(1 << len(source.items)),
        )


class IndexSubset(StrictModel):
    """A finite subset of nonnegative indices in canonical increasing order."""

    indices: tuple[Annotated[StrictInt, Field(ge=0)], ...] = Field(
        description="Strictly increasing nonnegative indices.",
        examples=[(0, 2)],
    )

    @model_validator(mode="after")
    def require_strictly_increasing_indices(self) -> Self:
        if any(left >= right for left, right in pairwise(self.indices)):
            raise _validation_error(
                "require_strictly_increasing_indices",
                "subset indices must be strictly increasing",
            )
        return self


__all__ = [
    "IndexSubset",
    "IndexedIntegerSequence",
    "SubsetSumProfile",
    "SubsetSumProfileEntry",
]
