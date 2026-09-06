"""Parent-bound values for finite abelian group computations."""

from typing import Self

from pydantic import model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianProductGroup,
)

MAX_GROUP_INTEGER_DIGITS = 32_768
GroupCoordinates = tuple[ExactInteger, ...]


def _require_coordinates(
    group: FiniteAbelianProductGroup, coordinates: tuple[int, ...]
) -> None:
    if len(coordinates) != len(group.moduli):
        raise ValueError("coordinates must match the ordered group axes")
    if any(
        not 0 <= value < modulus
        for value, modulus in zip(coordinates, group.moduli, strict=True)
    ):
        raise ValueError("group coordinates must be canonical residues")


class FiniteAbelianElement(StrictModel):
    """Canonical residues in one ordered cyclic-product parent."""

    group: FiniteAbelianProductGroup
    coordinates: GroupCoordinates

    @model_validator(mode="after")
    def require_coordinates(self) -> Self:
        _require_coordinates(self.group, self.coordinates)
        return self


class FiniteAbelianSubgroup(StrictModel):
    """A subgroup specified by ordered canonical generators in its ambient group.

    Generators need not be independent or minimal. This representation asserts
    no index, quotient decomposition, or enumeration result.
    """

    group: FiniteAbelianProductGroup
    generators: tuple[GroupCoordinates, ...]

    @model_validator(mode="after")
    def require_coordinates(self) -> Self:
        for generator in self.generators:
            _require_coordinates(self.group, generator)
        return self
