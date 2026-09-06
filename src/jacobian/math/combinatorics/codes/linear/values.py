"""Canonical values for bounded prime-field linear encoders."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_LINEAR_CODE_LENGTH = (
    64  # rowspace operations are O(k^2 n) and remain cheap; raised from 32
)
MAX_LINEAR_CODE_DIMENSION = MAX_LINEAR_CODE_LENGTH
MAX_LINEAR_FIELD_ORDER = 251


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"code_linear.{code}", message)


class PrimeFieldLinearEncoder(StrictModel):
    """A full-rank linear encoder with explicit message and coordinate axes."""

    field_order: StrictInt = Field(
        ge=2,
        le=MAX_LINEAR_FIELD_ORDER,
        description="Prime order p of the encoder's field GF(p).",
    )
    message_axis: tuple[OpaqueLabel, ...] = Field(
        default=(),
        max_length=MAX_LINEAR_CODE_DIMENSION,
        description=(
            "Ordered unique labels for message coordinates and generator rows."
        ),
    )
    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_LINEAR_CODE_LENGTH,
        description=(
            "Ordered unique labels for encoded-word coordinates; the empty axis "
            "represents the unique length-zero word."
        ),
    )
    generator_matrix: tuple[tuple[StrictInt, ...], ...] = Field(
        default=(),
        max_length=MAX_LINEAR_CODE_DIMENSION,
        description=(
            "Full-row-rank generator matrix over GF(p); rows follow message_axis "
            "and columns follow coordinate_axis. Entries are canonical residues."
        ),
    )

    @model_validator(mode="after")
    def require_structural_encoder(self) -> Self:
        if len(set(self.message_axis)) != len(self.message_axis):
            raise _validation_error(
                "message_axis_labels_must_be_unique",
                "message-axis labels must be unique",
            )
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise _validation_error(
                "coordinate_axis_labels_must_be_unique",
                "coordinate-axis labels must be unique",
            )
        if len(self.generator_matrix) != len(self.message_axis):
            raise _validation_error(
                "generator_rows_must_match_the_message_axis",
                "generator rows must match the message axis",
            )
        if any(len(row) != len(self.coordinate_axis) for row in self.generator_matrix):
            raise _validation_error(
                "generator_columns_must_match_the_coordinate_axis",
                "generator columns must match the coordinate axis",
            )
        if any(
            not 0 <= entry < self.field_order
            for row in self.generator_matrix
            for entry in row
        ):
            raise _validation_error(
                "generator_entries_must_be_canonical_field_residues",
                "generator entries must be canonical field residues",
            )

        return self

    @property
    def codeword_count(self) -> int:
        """Return the number of distinct words in the encoder image."""

        count = 1
        for _ in self.message_axis:
            count *= self.field_order
        return count


__all__ = [
    "MAX_LINEAR_CODE_DIMENSION",
    "MAX_LINEAR_CODE_LENGTH",
    "MAX_LINEAR_FIELD_ORDER",
    "PrimeFieldLinearEncoder",
]
