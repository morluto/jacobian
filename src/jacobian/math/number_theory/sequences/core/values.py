"""Canonical finite integer-sequence values.

The sequence is an ordered value: repeated entries and their positions are
meaningful.  Request models re-export this type so native callers and the
``math.run`` boundary use the same canonical source value.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

MAX_SEQUENCE_LENGTH = 100_000
MAX_INTEGER_SEQUENCE_ITEM_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
MAX_SEQUENCE_WIRE_BYTES = CanonicalLimits().max_output_bytes // 2


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"sequences.{reason}", message)


class IntegerSequence(StrictModel):
    """One nonempty finite sequence of canonical integers.

    These are structural limits on the reusable value.  Operation-specific
    work and result budgets remain owned by each native operation.
    """

    values: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    @model_validator(mode="after")
    def require_bounded_representation(self) -> Self:
        if any(
            len(value.lstrip("-")) > MAX_INTEGER_SEQUENCE_ITEM_DIGITS
            for value in self.values
        ):
            raise _validation_error(
                "item_too_large",
                "sequence item exceeds the "
                f"{MAX_INTEGER_SEQUENCE_ITEM_DIGITS}-digit bound",
            )
        estimated = sum(len(value) + 3 for value in self.values) + 64
        if estimated > MAX_SEQUENCE_WIRE_BYTES:
            raise _validation_error(
                "transport_too_large",
                "sequence request exceeds the "
                f"{MAX_SEQUENCE_WIRE_BYTES}-byte transport envelope; "
                "chunk the sequence into <=10MiB pieces and compose via typed values",
            )
        return self


__all__ = [
    "MAX_INTEGER_SEQUENCE_ITEM_DIGITS",
    "MAX_SEQUENCE_LENGTH",
    "MAX_SEQUENCE_WIRE_BYTES",
    "IntegerSequence",
]
