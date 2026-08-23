"""Typed contracts for finite unions of congruence classes."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictBool, StringConstraints, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer

MAX_PERIODIC_FAMILY_SIZE = 64
MAX_PERIODIC_SOURCE_ROWS = 4_096
MAX_PERIODIC_INTEGER_DIGITS = 256
MAX_MATERIALIZED_RESIDUES = 65_536

PeriodicSignedInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]{0,255})$",
        max_length=MAX_PERIODIC_INTEGER_DIGITS + 1,
        strict=True,
    ),
]
PeriodicNonnegativeInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]{0,255})$",
        max_length=MAX_PERIODIC_INTEGER_DIGITS,
        strict=True,
    ),
]
PeriodicPositiveInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^[1-9][0-9]{0,255}$",
        max_length=MAX_PERIODIC_INTEGER_DIGITS,
        strict=True,
    ),
]


class PeriodicCongruenceSubsetInput(StrictModel):
    """One finite residue subset modulo a positive integer.

    Residue representatives may be signed or outside the canonical interval;
    the operation reduces them modulo ``modulus`` before merging equal moduli.
    """

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"modulus": "6", "residues": ["-1", "1"]}]}
    )

    modulus: PeriodicPositiveInteger = Field(
        description=("Positive canonical decimal modulus with at most 256 digits."),
        examples=["6"],
    )
    residues: tuple[PeriodicSignedInteger, ...] = Field(
        max_length=MAX_PERIODIC_SOURCE_ROWS,
        description=(
            "Finite residue representatives as canonical decimal integers. "
            "They are reduced modulo the declared modulus, sorted, and deduplicated."
        ),
        examples=[["-1", "1"]],
    )


class PeriodicCongruenceSubset(StrictModel):
    """One canonical residue subset modulo a positive integer."""

    modulus: PeriodicPositiveInteger = Field(
        description="Positive canonical decimal modulus of this residue subset."
    )
    residues: tuple[PeriodicNonnegativeInteger, ...] = Field(
        max_length=MAX_PERIODIC_SOURCE_ROWS,
        description=("Strictly increasing canonical representatives in [0, modulus)."),
    )

    @model_validator(mode="after")
    def require_canonical_residues(self) -> Self:
        modulus = parse_canonical_integer(self.modulus)
        residues = tuple(map(parse_canonical_integer, self.residues))
        if residues != tuple(sorted(set(residues))):
            raise ValueError("canonical residues must be strictly increasing")
        if any(residue < 0 or residue >= modulus for residue in residues):
            raise ValueError("canonical residues must lie in [0, modulus)")
        return self


class PeriodicCongruenceUnionSource(StrictModel):
    """Canonical source for a union, optionally complemented in its common period."""

    subsets: tuple[PeriodicCongruenceSubset, ...] = Field(
        max_length=MAX_PERIODIC_FAMILY_SIZE
    )
    complement: StrictBool

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        moduli = tuple(
            parse_canonical_integer(subset.modulus) for subset in self.subsets
        )
        if moduli != tuple(sorted(set(moduli))):
            raise ValueError("canonical source moduli must be strictly increasing")
        if (
            sum(len(subset.residues) for subset in self.subsets)
            > MAX_PERIODIC_SOURCE_ROWS
        ):
            raise ValueError(
                f"normalized source exceeds the {MAX_PERIODIC_SOURCE_ROWS}-residue-row bound"
            )
        return self


class PeriodicCongruenceUnionRequest(StrictModel):
    """A bounded family of residue subsets interpreted by union and complement.

    The family may be empty and may repeat moduli or residue representatives.
    Validation normalizes representatives modulo their positive modulus, merges
    repeated moduli, and admits either a bounded one-period lift or a bounded
    generalized-CRT inclusion-exclusion computation.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "subsets": [
                        {"modulus": "4", "residues": ["0", "1"]},
                        {"modulus": "6", "residues": ["-1", "1"]},
                    ],
                    "complement": False,
                }
            ]
        }
    )

    subsets: tuple[PeriodicCongruenceSubsetInput, ...] = Field(
        max_length=MAX_PERIODIC_FAMILY_SIZE,
        description=(
            "At most 64 finite residue subsets. Equal moduli are merged; the empty "
            "family has common period 1 and denotes the empty union."
        ),
    )
    complement: StrictBool = Field(
        default=False,
        description=(
            "If true, return the complement of the union inside residues "
            "0 through L-1, where L is the lcm of the normalized source moduli."
        ),
        examples=[False],
    )

    @model_validator(mode="after")
    def require_bounded_exact_execution(self) -> Self:
        raw_rows = sum(len(subset.residues) for subset in self.subsets)
        if raw_rows > MAX_PERIODIC_SOURCE_ROWS:
            raise ValueError(
                f"source exceeds the {MAX_PERIODIC_SOURCE_ROWS}-residue-row bound"
            )
        from jacobian.math.number_theory._periodic_kernel import (
            require_admitted_periodic_source,
        )

        require_admitted_periodic_source(self.normalized_source())
        return self

    def normalized_source(self) -> PeriodicCongruenceUnionSource:
        """Return the canonical union source represented by this request."""

        merged: dict[int, set[int]] = {}
        for subset in self.subsets:
            modulus = parse_canonical_integer(subset.modulus)
            residues = merged.setdefault(modulus, set())
            residues.update(
                parse_canonical_integer(residue) % modulus
                for residue in subset.residues
            )
        return PeriodicCongruenceUnionSource(
            subsets=tuple(
                PeriodicCongruenceSubset(
                    modulus=format_canonical_integer(modulus),
                    residues=tuple(
                        format_canonical_integer(residue)
                        for residue in sorted(residues)
                    ),
                )
                for modulus, residues in sorted(merged.items())
            ),
            complement=self.complement,
        )


class PeriodicCongruenceUnionProfileRequest(PeriodicCongruenceUnionRequest):
    """A periodic-congruence union whose complete common-period set fits output."""

    @model_validator(mode="after")
    def require_materializable_profile(self) -> Self:
        from jacobian.math.number_theory._periodic_kernel import (
            require_materializable_periodic_source,
        )

        require_materializable_periodic_source(self.normalized_source())
        return self


class PeriodicCongruenceUnionMeasureResult(StrictModel):
    """Exact occupied count and density, bound to the normalized union source."""

    semantics_version: Literal["periodic-congruence-union.v1"]
    source: PeriodicCongruenceUnionSource
    common_period: PeriodicPositiveInteger = Field(
        description="Least common multiple of every normalized source modulus."
    )
    occupied_count: PeriodicNonnegativeInteger = Field(
        description="Exact number of occupied representatives in the common period."
    )
    density: CanonicalRational = Field(
        description="Reduced exact ratio occupied_count/common_period."
    )

    @model_validator(mode="after")
    def bind_measure_to_source(self) -> Self:
        from jacobian.math.number_theory._periodic_kernel import (
            common_period,
            measure_periodic_union,
        )

        period = common_period(self.source)
        occupied_count = measure_periodic_union(self.source)
        if self.common_period != format_canonical_integer(period):
            raise ValueError("common period must be the lcm of the source moduli")
        if self.occupied_count != format_canonical_integer(occupied_count):
            raise ValueError("occupied count does not match the normalized source")
        expected_density = CanonicalRational.from_fraction(
            Fraction(occupied_count, period)
        )
        if self.density != expected_density:
            raise ValueError("density must equal occupied_count/common_period")
        return self


class PeriodicCongruenceUnionProfileResult(PeriodicCongruenceUnionMeasureResult):
    """Complete canonical occupied residues in the source's common period."""

    occupied_residues: tuple[PeriodicNonnegativeInteger, ...] = Field(
        max_length=MAX_MATERIALIZED_RESIDUES,
        description=(
            "Every occupied representative in [0, common_period), in increasing order."
        ),
    )

    @model_validator(mode="after")
    def bind_profile_to_source(self) -> Self:
        from jacobian.math.number_theory._periodic_kernel import (
            materialize_periodic_union,
        )

        expected = materialize_periodic_union(self.source)
        actual = tuple(map(parse_canonical_integer, self.occupied_residues))
        if actual != expected:
            raise ValueError(
                "occupied residues must be every satisfying residue in canonical order"
            )
        return self


__all__ = [
    "MAX_MATERIALIZED_RESIDUES",
    "MAX_PERIODIC_FAMILY_SIZE",
    "MAX_PERIODIC_INTEGER_DIGITS",
    "MAX_PERIODIC_SOURCE_ROWS",
    "PeriodicCongruenceSubset",
    "PeriodicCongruenceSubsetInput",
    "PeriodicCongruenceUnionMeasureResult",
    "PeriodicCongruenceUnionProfileRequest",
    "PeriodicCongruenceUnionProfileResult",
    "PeriodicCongruenceUnionRequest",
    "PeriodicCongruenceUnionSource",
]
