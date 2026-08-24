"""Canonical values for bounded prime-field linear encoders."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix, rank

MAX_LINEAR_CODE_LENGTH = (
    64  # rowspace operations are O(k^2 n) and remain cheap; raised from 32
)
MAX_LINEAR_CODE_DIMENSION = MAX_LINEAR_CODE_LENGTH


class PrimeFieldLinearEncoder(StrictModel):
    """A full-rank linear encoder with explicit message and coordinate axes."""

    field_order: StrictInt = Field(
        ge=2,
        le=251,
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
    def require_bounded_full_rank_encoder(self) -> Self:
        if len(set(self.message_axis)) != len(self.message_axis):
            raise ValueError("message-axis labels must be unique")
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise ValueError("coordinate-axis labels must be unique")
        if len(self.generator_matrix) != len(self.message_axis):
            raise ValueError("generator rows must match the message axis")

        matrix = PrimeFieldMatrix(
            prime=self.field_order,
            entries=self.generator_matrix,
            columns=len(self.coordinate_axis),
        )
        if rank(matrix) != len(self.generator_matrix):
            raise ValueError("generator matrix must have full row rank")

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
    "PrimeFieldLinearEncoder",
]
