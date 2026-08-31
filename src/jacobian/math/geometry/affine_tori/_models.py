"""Public request and source-bound result contracts for affine-torus fixed loci."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.geometry.affine_tori.values import (
    IntegralTorusCharacter,
    RationalAffineTorusMap,
    RationalTorusCosetFamily,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"affine_torus.{reason}", message)


class AffineTorusFixedLocusRequest(StrictModel):
    """One bounded affine self-map of a standard real torus."""

    affine_map: RationalAffineTorusMap


class NonemptyAffineTorusFixedLocus(StrictModel):
    """An exact finite-coset presentation of a nonempty fixed locus."""

    status: Literal["NONEMPTY"] = "NONEMPTY"
    fixed_locus: RationalTorusCosetFamily


class EmptyAffineTorusFixedLocus(StrictModel):
    """A primitive invariant character proving that the fixed locus is empty."""

    status: Literal["EMPTY"] = "EMPTY"
    obstruction: IntegralTorusCharacter
    obstruction_pairing: CanonicalRational = Field(
        description="The nonzero value phi(b) in the canonical interval [0,1)."
    )

    @model_validator(mode="after")
    def require_nonzero_canonical_pairing(self) -> Self:
        pairing = self.obstruction_pairing.as_fraction()
        if not 0 < pairing < 1:
            raise _validation_error(
                "obstruction_pairing",
                "empty-locus obstruction pairing must lie in (0,1)",
            )
        return self


AffineTorusFixedLocusOutcome = Annotated[
    NonemptyAffineTorusFixedLocus | EmptyAffineTorusFixedLocus,
    Field(discriminator="status"),
]


class AffineTorusFixedLocusResult(StrictModel):
    """The source map and its exact, discriminated fixed-locus conclusion."""

    source: RationalAffineTorusMap
    outcome: AffineTorusFixedLocusOutcome

    @model_validator(mode="after")
    def require_source_ambient(self) -> Self:
        torus = self.source.torus
        if isinstance(self.outcome, NonemptyAffineTorusFixedLocus):
            family = self.outcome.fixed_locus
            if family.ambient_torus != torus:
                raise _validation_error(
                    "source_mismatch", "fixed locus must belong to the source torus"
                )
            dimension = torus.dimension
            from flint import fmpz_mat

            displacement = fmpz_mat(
                [
                    [
                        parse_canonical_integer(self.source.linear_part.entries[row][column])
                        - int(row == column)
                        for column in range(dimension)
                    ]
                    for row in range(dimension)
                ]
            )
            displacement_rank = int(displacement.rank())
            expected_parameter_dimension = dimension - displacement_rank
            if (
                family.identity_component.parameter_dimension
                != expected_parameter_dimension
            ):
                raise _validation_error(
                    "source_fixed_locus",
                    "identity-component dimension must match the source kernel",
                )
            embedding = fmpz_mat(
                [
                    [parse_canonical_integer(value) for value in row]
                    for row in family.identity_component.embedding.entries
                ]
            )
            displaced_embedding = displacement * embedding
            if any(
                int(displaced_embedding[row, column]) != 0
                for row in range(dimension)
                for column in range(expected_parameter_dimension)
            ):
                raise _validation_error(
                    "source_fixed_locus",
                    "identity-component embedding must lie in the source kernel",
                )
            smith = displacement.snf()
            expected_component_count = 1
            for index in range(displacement_rank):
                expected_component_count *= abs(int(smith[index, index]))
            if parse_canonical_integer(
                family.finite_components.component_count
            ) != expected_component_count:
                raise _validation_error(
                    "source_fixed_locus",
                    "component count must match the source fixed locus",
                )

            translation = tuple(
                coordinate.as_fraction()
                for coordinate in self.source.translation.coordinates
            )
            base_point = tuple(
                coordinate.as_fraction()
                for coordinate in family.base_point.coordinates
            )
            for row in range(dimension):
                base_displacement = translation[row] + sum(
                    int(displacement[row, column]) * base_point[column]
                    for column in range(dimension)
                )
                if base_displacement.denominator != 1:
                    raise _validation_error(
                        "source_fixed_locus",
                        "base point must satisfy the source fixed-point equation",
                    )
            for generator in family.component_generators:
                coordinates = tuple(
                    coordinate.as_fraction() for coordinate in generator.coordinates
                )
                if any(
                    (
                        sum(
                            int(displacement[row, column]) * coordinates[column]
                            for column in range(dimension)
                        ).denominator
                        != 1
                    )
                    for row in range(dimension)
                ):
                    raise _validation_error(
                        "source_fixed_locus",
                        "component generators must satisfy the source fixed-point equation",
                    )
        else:
            if self.outcome.obstruction.torus != torus:
                raise _validation_error(
                    "source_mismatch",
                    "obstruction must be a character of the source torus",
                )
            character = tuple(
                parse_canonical_integer(value)
                for value in self.outcome.obstruction.coefficients
            )
            linear = self.source.linear_part.entries
            for column in range(torus.dimension):
                if (
                    sum(
                        character[row]
                        * (
                            parse_canonical_integer(linear[row][column])
                            - int(row == column)
                        )
                        for row in range(torus.dimension)
                    )
                    != 0
                ):
                    raise _validation_error(
                        "obstruction_invariant",
                        "empty-locus obstruction must annihilate the linear displacement",
                    )
            expected_pairing = (
                sum(
                    character[index]
                    * self.source.translation.coordinates[index].as_fraction()
                    for index in range(torus.dimension)
                )
                % 1
            )
            if expected_pairing != self.outcome.obstruction_pairing.as_fraction():
                raise _validation_error(
                    "obstruction_pairing_source",
                    "empty-locus obstruction pairing must match the source translation",
                )
        return self


__all__ = [
    "AffineTorusFixedLocusOutcome",
    "AffineTorusFixedLocusRequest",
    "AffineTorusFixedLocusResult",
    "EmptyAffineTorusFixedLocus",
    "NonemptyAffineTorusFixedLocus",
]
