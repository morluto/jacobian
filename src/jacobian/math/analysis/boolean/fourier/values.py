"""Sparse rational functions in the character basis of a Boolean cube."""

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_CHARACTER_AXES = 4096
MAX_CHARACTER_TERMS = 4096
MAX_CHARACTER_INCIDENCES = 65536


class WalshTerm(StrictModel):
    """coefficient * (-1) ** sum(x[i] for i in character)."""

    character: tuple[int, ...] = Field(max_length=MAX_CHARACTER_AXES)
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_character(self) -> Self:
        if self.character != tuple(sorted(set(self.character))) or any(
            i < 0 or i >= MAX_CHARACTER_AXES for i in self.character
        ):
            raise ValueError(
                "character coordinates must be distinct, increasing nonnegative indices"
            )
        if self.coefficient.num == "0":
            raise ValueError("zero character terms must be omitted")
        return self


class RationalWalshPolynomial(StrictModel):
    """Canonical sparse QQ character expansion, retaining the ambient cube.

    Terms use increasing lexicographic character order, with no duplicates or
    zeros. Coordinate i is bit i of the natural-order Boolean assignment.
    """

    variable_count: int = Field(ge=0, le=MAX_CHARACTER_AXES)
    terms: tuple[WalshTerm, ...] = Field(default=(), max_length=MAX_CHARACTER_TERMS)
    convention: Literal["BOOLEAN_CHARACTERS"] = "BOOLEAN_CHARACTERS"

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        characters = tuple(term.character for term in self.terms)
        if characters != tuple(sorted(set(characters))):
            raise ValueError(
                "character terms must be unique in increasing lexicographic order"
            )
        if any(i >= self.variable_count for character in characters for i in character):
            raise ValueError("character coordinate must belong to the ambient cube")
        if sum(map(len, characters)) > MAX_CHARACTER_INCIDENCES:
            raise ValueError(
                "character incidence count exceeds the representation envelope"
            )
        return self


class BooleanAffineMap(StrictModel):
    """Map y to x with x[i] = sum(y[j] for j in rows[i]) + offset[i] over GF(2).

    Rows are sparse, sorted unique target-coordinate indices. Their count is
    the source polynomial's dimension; target_dimension retains inactive axes.
    """

    target_dimension: int = Field(ge=0, le=MAX_CHARACTER_AXES)
    rows: tuple[tuple[int, ...], ...] = Field(max_length=MAX_CHARACTER_AXES)
    offset: tuple[Literal[0, 1], ...] = Field(max_length=MAX_CHARACTER_AXES)

    @model_validator(mode="after")
    def require_map_shape(self) -> Self:
        if len(self.offset) != len(self.rows):
            raise ValueError("offset must have one bit per source coordinate")
        if sum(map(len, self.rows)) > MAX_CHARACTER_INCIDENCES:
            raise ValueError(
                "affine map incidence count exceeds the representation envelope"
            )
        if any(
            row != tuple(sorted(set(row)))
            or any(j < 0 or j >= self.target_dimension for j in row)
            for row in self.rows
        ):
            raise ValueError(
                "affine rows must have distinct increasing target coordinates"
            )
        return self
