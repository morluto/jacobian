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
            if self.outcome.fixed_locus.ambient_torus != torus:
                raise _validation_error(
                    "source_mismatch", "fixed locus must belong to the source torus"
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
                if sum(
                    character[row]
                    * (
                        parse_canonical_integer(linear[row][column])
                        - int(row == column)
                    )
                    for row in range(torus.dimension)
                ) != 0:
                    raise _validation_error(
                        "obstruction_invariant",
                        "empty-locus obstruction must annihilate the linear displacement",
                    )
            expected_pairing = sum(
                character[index]
                * self.source.translation.coordinates[index].as_fraction()
                for index in range(torus.dimension)
            ) % 1
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
