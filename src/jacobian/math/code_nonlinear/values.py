"""Canonical values for explicit nonlinear binary codes."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

# A materialized code carries exactly ``length * cardinality`` binary entries.
# Keeping that quantity below 2**19 bounds source parsing, canonical sorting,
# the packed-integer conversion, and retained-source memory independently of
# any profile's pairwise work or output obligation.  The documented
# standard A(23,6,10) construction (length 23, minimum distance 6,
# constant weight 10) with cardinality 2992 uses 68,816 entries.
MAX_EXPLICIT_CODE_BITS = 1 << 19
MAX_EXPLICIT_CODE_LENGTH = MAX_EXPLICIT_CODE_BITS

BinaryBit = Annotated[int, Field(strict=True, ge=0, le=1)]


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


type BinaryWord = Annotated[
    tuple[BinaryBit, ...],
    Field(max_length=MAX_EXPLICIT_CODE_LENGTH),
]


class ExplicitBinaryCode(StrictModel):
    """A canonical finite set of distinct equal-length binary words.

    ``length`` retains the ambient coordinate axis, including for the empty
    code.  Input word order is not mathematical and is normalized to
    lexicographic order; duplicates are rejected before normalization.
    The complete materialized source is bounded by
    ``length * cardinality <= 524288`` bits.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "A finite set of distinct equal-length binary words. `length` "
                "is the coordinate-axis size in [0, 524288]; every word has "
                "exactly that length and contains strict integer bits 0 or 1; "
                "length*cardinality is at most 524288. Word order is normalized "
                "lexicographically, and duplicates are rejected. At length 0, "
                "the code is either empty or contains the sole empty word."
            )
        },
    )

    length: Annotated[
        int,
        Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH),
    ]
    codewords: tuple[BinaryWord, ...] = Field(
        default=(),
        max_length=MAX_EXPLICIT_CODE_BITS,
        description=(
            "Distinct binary words, each of exactly `length` entries; input "
            "order is normalized lexicographically and the total number of "
            "materialized entries is at most 524288."
        ),
        examples=[((0, 0, 0), (0, 1, 1), (1, 1, 0))],
    )

    @model_validator(mode="before")
    @classmethod
    def require_bounded_containers(cls, data: Any) -> Any:
        """Bound raw containers before Pydantic materializes nested bits.

        ``require_canonical_code`` checks ``length * cardinality`` only after
        every supplied bit has been converted, so raw shapes such as millions
        of empty words or one enormous word could otherwise consume memory and
        CPU far beyond the retained-source bound before rejection. Every
        accepted code keeps at most ``MAX_EXPLICIT_CODE_BITS`` codewords (a
        zero-length code later admits only the sole empty word) and at most
        ``MAX_EXPLICIT_CODE_BITS`` materialized entries; both are checked on
        the raw containers here.
        """
        if not isinstance(data, dict):
            return data
        words = data.get("codewords")
        if not isinstance(words, (list, tuple)):
            return data
        if len(words) > MAX_EXPLICIT_CODE_BITS:
            raise _validation_error(
                "nonlinear_code.container_bound",
                "explicit binary code declares "
                f"{len(words)} codewords, exceeding the "
                f"{MAX_EXPLICIT_CODE_BITS}-word container bound",
            )
        total_bits = 0
        for word in words:
            if isinstance(word, (list, tuple)):
                total_bits += len(word)
                if total_bits > MAX_EXPLICIT_CODE_BITS:
                    raise _validation_error(
                        "nonlinear_code.source_bound",
                        "explicit binary code materializes "
                        f"{total_bits} bits, exceeding the "
                        f"{MAX_EXPLICIT_CODE_BITS}-bit source bound",
                    )
        # Running a before validator moves field validation into Python
        # mode, where decoded JSON arrays no longer coerce to the declared
        # tuple shapes; normalize the raw containers on a copied path so
        # JSON invocation keeps working while stored values stay canonical.
        return {
            **data,
            "codewords": tuple(
                tuple(word) if isinstance(word, list) else word for word in words
            ),
        }

    @model_validator(mode="after")
    def require_canonical_code(self) -> Self:
        cardinality = len(self.codewords)
        if self.length == 0 and cardinality > 1:
            raise _validation_error(
                "nonlinear_code.empty_codeword_constraint",
                "a zero-coordinate code can contain only the sole empty word",
            )
        total_bits = self.length * cardinality
        if total_bits > MAX_EXPLICIT_CODE_BITS:
            raise _validation_error(
                "nonlinear_code.source_bound",
                "explicit binary code materializes "
                f"{total_bits} bits, exceeding the "
                f"{MAX_EXPLICIT_CODE_BITS}-bit source bound",
            )
        if any(len(word) != self.length for word in self.codewords):
            raise _validation_error(
                "nonlinear_code.length_mismatch",
                "every codeword must have exactly the declared length",
            )
        if len(set(self.codewords)) != cardinality:
            raise _validation_error(
                "nonlinear_code.duplicate_codewords", "codewords must be distinct"
            )
        object.__setattr__(self, "codewords", tuple(sorted(self.codewords)))
        return self


__all__ = [
    "MAX_EXPLICIT_CODE_BITS",
    "MAX_EXPLICIT_CODE_LENGTH",
    "ExplicitBinaryCode",
]
