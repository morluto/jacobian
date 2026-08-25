"""Typed contracts for finite unions of congruence classes."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictBool, StringConstraints, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    parse_canonical_integer,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable semantic error owned by the number-theory domain."""

    return PydanticCustomError(f"number_theory.{reason}", message)


MAX_PERIODIC_FAMILY_SIZE = 64
MAX_PERIODIC_SOURCE_ROWS = 4_096
MAX_PERIODIC_INTEGER_DIGITS = 256
MAX_MATERIALIZED_RESIDUES = 65_536
MAX_PERIOD_SCAN = 1_000_000
MAX_PERIOD_LIFT_WORK = 2_000_000
# Sparse lifting performs one bounded integer-set insertion per lifted source
# row and retains no more states than rows visited. This conservative cap is
# independent of the common-period size and keeps both quantities bounded.
MAX_SPARSE_LIFTED_ROWS = 65_536
MAX_INTERSECTION_STATES = 65_535
MAX_INTERSECTION_MERGES = 100_000
PERIODIC_EXECUTION_PASSES_PER_CALL = 2
MAX_PERIODIC_RESULT_BYTES = CanonicalLimits().max_output_bytes
PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES = 4_096

_PERIODIC_REQUEST_EXAMPLE = {
    "subsets": [
        {"modulus": "4", "residues": ["0", "1"]},
        {"modulus": "6", "residues": ["-1", "1"]},
    ],
    "complement": False,
}
_PERIODIC_REQUEST_DESCRIPTION = (
    "Normalize and merge at most 64 residue subsets with at most 4,096 raw "
    "and 4,096 normalized residue rows. Each modulus and their lcm L have at "
    "most 256 decimal digits. With W = sum(|R_i| * L/m_i), exact execution "
    "uses a full-subset shortcut; a period lift when L <= 1,000,000 and "
    "L + W <= 2,000,000; a sparse lift when W <= 65,536; or generalized-CRT "
    "inclusion-exclusion with at most 65,535 retained states and 100,000 merges "
    "per pass. Every call performs one producer pass and one source-bound "
    "result replay, so whole-call lift and merge work is at most twice the "
    "per-pass limit."
)


def _periodic_request_schema_extra(*, profile: bool) -> JsonSchemaValue:
    description = _PERIODIC_REQUEST_DESCRIPTION
    extra: JsonSchemaValue = {
        "description": description,
        "examples": [_PERIODIC_REQUEST_EXAMPLE],
        "aggregate_raw_residue_row_limit": MAX_PERIODIC_SOURCE_ROWS,
        "aggregate_normalized_residue_row_limit": MAX_PERIODIC_SOURCE_ROWS,
        "common_period_digit_limit": MAX_PERIODIC_INTEGER_DIGITS,
        "execution_passes_per_call": PERIODIC_EXECUTION_PASSES_PER_CALL,
        "execution_regime_limits": {
            "period_lift": {
                "max_common_period": MAX_PERIOD_SCAN,
                "max_period_plus_lifted_rows_per_pass": MAX_PERIOD_LIFT_WORK,
                "max_period_plus_lifted_rows_per_call": (
                    PERIODIC_EXECUTION_PASSES_PER_CALL * MAX_PERIOD_LIFT_WORK
                ),
            },
            "sparse_lift": {
                "max_lifted_rows_per_pass": MAX_SPARSE_LIFTED_ROWS,
                "max_lifted_rows_per_call": (
                    PERIODIC_EXECUTION_PASSES_PER_CALL * MAX_SPARSE_LIFTED_ROWS
                ),
                "max_retained_states_per_pass": MAX_SPARSE_LIFTED_ROWS,
            },
            "inclusion_exclusion": {
                "max_retained_states_per_pass": MAX_INTERSECTION_STATES,
                "max_merges_per_pass": MAX_INTERSECTION_MERGES,
                "max_merges_per_call": (
                    PERIODIC_EXECUTION_PASSES_PER_CALL * MAX_INTERSECTION_MERGES
                ),
            },
        },
    }
    if profile:
        extra.update(
            {
                "description": (
                    description
                    + " A materialized profile returns at most 65,536 residues. "
                    "A full-subset shortcut returns L union rows (so L <= 65,536) "
                    "or an empty complement. Otherwise, non-complements require "
                    "W <= 65,536; complements require L <= 65,536 and "
                    "L + W <= 2,000,000. The retained source, residue list, and "
                    "4,096-byte result envelope must fit the 10,485,760-byte "
                    "canonical output limit."
                ),
                "profile_materialized_residue_limit": MAX_MATERIALIZED_RESIDUES,
                "profile_full_union_period_limit": MAX_MATERIALIZED_RESIDUES,
                "profile_general_noncomplement_lifted_row_limit": (
                    MAX_MATERIALIZED_RESIDUES
                ),
                "profile_nontrivial_complement_period_limit": (
                    MAX_MATERIALIZED_RESIDUES
                ),
                "profile_materialization_work_limit": MAX_PERIOD_LIFT_WORK,
                "profile_result_envelope_bytes": PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES,
                "profile_result_byte_limit": MAX_PERIODIC_RESULT_BYTES,
            }
        )
    return extra


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
            raise _validation_error(
                "canonical_residues_must_be_strictly_increasing",
                "canonical residues must be strictly increasing",
            )
        if any(residue < 0 or residue >= modulus for residue in residues):
            raise _validation_error(
                "canonical_residues_must_lie_in_0_modulus",
                "canonical residues must lie in [0, modulus)",
            )
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
            raise _validation_error(
                "canonical_source_moduli_must_be_strictly_increasing",
                "canonical source moduli must be strictly increasing",
            )
        if (
            sum(len(subset.residues) for subset in self.subsets)
            > MAX_PERIODIC_SOURCE_ROWS
        ):
            raise _validation_error(
                "f_normalized_source_exceeds_the_max_periodic_source",
                f"normalized source exceeds the {MAX_PERIODIC_SOURCE_ROWS}-residue-row bound",
            )
        return self


class PeriodicCongruenceUnionRequest(StrictModel):
    """A bounded family of residue subsets interpreted by union and complement.

    The family may be empty and may repeat moduli or residue representatives.
    Validation normalizes representatives modulo their positive modulus, merges
    repeated moduli, and admits a bounded period lift, sparse lift, or
    generalized-CRT inclusion-exclusion computation.
    """

    model_config = ConfigDict(
        json_schema_extra=_periodic_request_schema_extra(profile=False)
    )

    subsets: tuple[PeriodicCongruenceSubsetInput, ...] = Field(
        max_length=MAX_PERIODIC_FAMILY_SIZE,
        description=(
            "At most 64 finite residue subsets and 4,096 raw residue rows in "
            "aggregate. Equal moduli are merged, with at most 4,096 normalized "
            "rows; the empty family has common period 1 and denotes the empty union."
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

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source_rows(cls, data: object) -> object:
        """Reject oversized raw families before constructing every row model."""

        if not isinstance(data, Mapping):
            return data
        subsets = data.get("subsets")
        if not isinstance(subsets, (list, tuple)):
            return data
        if len(subsets) > MAX_PERIODIC_FAMILY_SIZE:
            raise _validation_error(
                "f_the_family_exceeds_the_max_periodic_family",
                f"the family exceeds the {MAX_PERIODIC_FAMILY_SIZE}-subset bound",
            )
        total = 0
        normalized_subsets: list[object] = []
        for subset in subsets:
            if isinstance(subset, PeriodicCongruenceSubsetInput):
                total += len(subset.residues)
                if total > MAX_PERIODIC_SOURCE_ROWS:
                    raise _validation_error(
                        "f_source_exceeds_the_max_periodic_source_rows",
                        f"source exceeds the {MAX_PERIODIC_SOURCE_ROWS}-residue-row "
                        "bound",
                    )
                normalized_subsets.append(subset)
            elif isinstance(subset, Mapping):
                residues = subset.get("residues")
                if not isinstance(residues, (list, tuple)):
                    normalized_subsets.append(subset)
                    continue
                total += len(residues)
                if total > MAX_PERIODIC_SOURCE_ROWS:
                    raise _validation_error(
                        "f_source_exceeds_the_max_periodic_source_rows",
                        f"source exceeds the {MAX_PERIODIC_SOURCE_ROWS}-residue-row "
                        "bound",
                    )
                normalized_subset = dict(subset)
                normalized_subset["residues"] = tuple(residues)
                normalized_subsets.append(normalized_subset)
            else:
                normalized_subsets.append(subset)
        normalized_data = dict(data)
        normalized_data["subsets"] = tuple(normalized_subsets)
        return normalized_data

    @model_validator(mode="after")
    def require_bounded_exact_execution(self) -> Self:
        raw_rows = sum(len(subset.residues) for subset in self.subsets)
        if raw_rows > MAX_PERIODIC_SOURCE_ROWS:
            raise _validation_error(
                "f_source_exceeds_the_max_periodic_source_rows",
                f"source exceeds the {MAX_PERIODIC_SOURCE_ROWS}-residue-row bound",
            )
        from jacobian.math.number_theory._periodic_kernel import (
            require_admitted_periodic_source,
        )

        try:
            require_admitted_periodic_source(self.normalized_source())
        except ValueError as error:
            raise _validation_error("backend_admission", str(error)) from error
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

    model_config = ConfigDict(
        json_schema_extra=_periodic_request_schema_extra(profile=True)
    )

    @model_validator(mode="after")
    def require_materializable_profile(self) -> Self:
        from jacobian.math.number_theory._periodic_kernel import (
            require_materializable_periodic_source,
        )

        try:
            require_materializable_periodic_source(self.normalized_source())
        except ValueError as error:
            raise _validation_error("backend_admission", str(error)) from error
        return self


def _require_measure_binding(
    common_period: str,
    occupied_count_text: str,
    density: CanonicalRational,
    *,
    period: int,
    occupied_count: int,
) -> None:
    if common_period != format_canonical_integer(period):
        raise _validation_error(
            "common_period_must_be_the_lcm_of_the_source_moduli",
            "common period must be the lcm of the source moduli",
        )
    if occupied_count_text != format_canonical_integer(occupied_count):
        raise _validation_error(
            "occupied_count_does_not_match_the_normalized_source",
            "occupied count does not match the normalized source",
        )
    expected_density = CanonicalRational.from_fraction(Fraction(occupied_count, period))
    if density != expected_density:
        raise _validation_error(
            "density_must_equal_occupied_count_common_period",
            "density must equal occupied_count/common_period",
        )


class PeriodicCongruenceUnionMeasureResult(StrictModel):
    """Exact occupied count and density, bound to the normalized union source."""

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
        _require_measure_binding(
            self.common_period,
            self.occupied_count,
            self.density,
            period=period,
            occupied_count=occupied_count,
        )
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
    def bind_measure_to_source(self) -> Self:
        """Override measure replay with one complete profile replay."""

        from jacobian.math.number_theory._periodic_kernel import (
            common_period,
            materialize_periodic_union,
        )

        period = common_period(self.source)
        expected = materialize_periodic_union(self.source)
        actual = tuple(map(parse_canonical_integer, self.occupied_residues))
        _require_measure_binding(
            self.common_period,
            self.occupied_count,
            self.density,
            period=period,
            occupied_count=len(expected),
        )
        if actual != expected:
            raise _validation_error(
                "occupied_residues_must_be_every_satisfying_residue_in_canonical_order",
                "occupied residues must be every satisfying residue in canonical order",
            )
        return self


__all__ = [
    "MAX_INTERSECTION_MERGES",
    "MAX_INTERSECTION_STATES",
    "MAX_MATERIALIZED_RESIDUES",
    "MAX_PERIODIC_FAMILY_SIZE",
    "MAX_PERIODIC_INTEGER_DIGITS",
    "MAX_PERIODIC_RESULT_BYTES",
    "MAX_PERIODIC_SOURCE_ROWS",
    "MAX_PERIOD_LIFT_WORK",
    "MAX_PERIOD_SCAN",
    "MAX_SPARSE_LIFTED_ROWS",
    "PERIODIC_EXECUTION_PASSES_PER_CALL",
    "PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES",
    "PeriodicCongruenceSubset",
    "PeriodicCongruenceSubsetInput",
    "PeriodicCongruenceUnionMeasureResult",
    "PeriodicCongruenceUnionProfileRequest",
    "PeriodicCongruenceUnionProfileResult",
    "PeriodicCongruenceUnionRequest",
    "PeriodicCongruenceUnionSource",
]
