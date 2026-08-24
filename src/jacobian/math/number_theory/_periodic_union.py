"""Exact finite profiles of unions of periodic residue classes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.math.number_theory._support import number_theory_operation

# The kernel stores one byte per common-period residue and marks each lifted
# source residue once. These bounds separately cap source size, temporary
# memory, marking work, and the optional exact output. Construction and
# independent result replay each use one marking pass and one period scan.
MAX_PERIODIC_FAMILY_SIZE = 64
MAX_PERIODIC_MATERIALIZED_RESIDUES = 32_768
MAX_PERIODIC_SOURCE_RESIDUES = MAX_PERIODIC_MATERIALIZED_RESIDUES
MAX_PERIODIC_MODULUS = 1_000_000
MAX_PERIODIC_COMMON_PERIOD = 1_000_000
MAX_PERIODIC_MARK_WRITES_PER_PASS = 8_000_000


class PeriodicResidueSubset(StrictModel):
    """One canonical materialized subset of residues modulo a positive integer."""

    modulus: StrictInt = Field(
        ge=1,
        le=MAX_PERIODIC_MODULUS,
        description="The positive modulus, which is the subset's ambient parent.",
    )
    residues: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_PERIODIC_MATERIALIZED_RESIDUES,
        description=(
            "Canonical residue representatives in [0, modulus), strictly increasing "
            "and unique. The empty tuple retains its declared modulus."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_residues(self) -> Self:
        if self.residues != tuple(sorted(set(self.residues))):
            raise ValueError("residues must be strictly increasing and unique")
        if any(residue < 0 or residue >= self.modulus for residue in self.residues):
            raise ValueError("residues must be canonical for their modulus")
        return self


def _family_key(subset: PeriodicResidueSubset) -> tuple[int, tuple[int, ...]]:
    return subset.modulus, subset.residues


def _aggregate_same_modulus(
    subsets: tuple[PeriodicResidueSubset, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Union source rows with the same modulus without changing the source value."""

    residues_by_modulus: dict[int, set[int]] = {}
    for subset in subsets:
        residues_by_modulus.setdefault(subset.modulus, set()).update(subset.residues)
    return tuple(
        (modulus, tuple(sorted(residues)))
        for modulus, residues in sorted(residues_by_modulus.items())
    )


def _request_envelope(
    subsets: tuple[PeriodicResidueSubset, ...],
    *,
    complement: bool,
) -> tuple[int, int, int]:
    """Return common period, mark writes, and a safe output-cardinality bound."""

    common_period = 1
    for subset in subsets:
        common_period = math.lcm(common_period, subset.modulus)
        if common_period > MAX_PERIODIC_COMMON_PERIOD:
            raise ValueError(
                "the least common period exceeds the 1,000,000-residue bound"
            )

    aggregated = _aggregate_same_modulus(subsets)
    lifted_counts = tuple(
        len(residues) * (common_period // modulus) for modulus, residues in aggregated
    )
    mark_writes = sum(lifted_counts)
    if mark_writes > MAX_PERIODIC_MARK_WRITES_PER_PASS:
        raise ValueError(
            "lifting the aggregated source residues exceeds the 8,000,000-write "
            "per-pass work bound"
        )

    if complement:
        # Each aggregated same-modulus subset is contained in the union, so its
        # largest lifted cardinality is a certified lower bound on coverage.
        output_bound = common_period - max(lifted_counts, default=0)
    else:
        # The union cardinality is no larger than the sum of its lifted rows.
        output_bound = min(common_period, mark_writes)
    return common_period, mark_writes, output_bound


class PeriodicUnionProfileRequest(StrictModel):
    """A canonical bounded family whose exact periodic union is profiled.

    Same-modulus subsets are unioned before lifting, while the ordered source
    family remains unchanged in the result. Construction and independent replay
    each perform at most 8,000,000 lifted-mark writes and one scan of at most
    1,000,000 residues; together they perform at most 16,000,000 writes and two
    such scans, plus bounded aggregation of the materialized source rows.
    """

    subsets: tuple[PeriodicResidueSubset, ...] = Field(
        default=(),
        max_length=MAX_PERIODIC_FAMILY_SIZE,
        description=(
            "Canonical family ordered strictly by (modulus, residues), with no "
            "exact duplicates. Repeated moduli are allowed when their residue "
            "tuples differ; they are unioned for lifting but retained in source. "
            "The family may contain at most 32,768 residue rows in total."
        ),
    )
    complement: StrictBool = Field(
        default=False,
        description=(
            "If false, profile the union. If true, profile its complement inside "
            "[0, L), where L is the least common period. The empty family has "
            "L = 1, empty union, and complement {0}."
        ),
    )
    result_mode: Literal["count_only", "materialize_residues"] = Field(
        default="count_only",
        description=(
            "count_only returns no occupied subset. materialize_residues returns "
            "the canonical occupied subset only when admission proves at most "
            "32,768 output residues."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source_rows(cls, data: object) -> object:
        """Reject oversized nested families before constructing every row model."""

        if not isinstance(data, Mapping):
            return data
        subsets = data.get("subsets")
        if not isinstance(subsets, (list, tuple)):
            return data
        if len(subsets) > MAX_PERIODIC_FAMILY_SIZE:
            raise ValueError(
                f"the family exceeds the {MAX_PERIODIC_FAMILY_SIZE}-subset bound"
            )
        total = 0
        normalized_subsets: list[object] = []
        for subset in subsets:
            if isinstance(subset, PeriodicResidueSubset):
                total += len(subset.residues)
                if total > MAX_PERIODIC_SOURCE_RESIDUES:
                    raise ValueError(
                        "the family exceeds the 32,768-source-residue bound"
                    )
                normalized_subsets.append(subset)
            elif isinstance(subset, Mapping):
                residues = subset.get("residues")
                if isinstance(residues, (list, tuple)):
                    total += len(residues)
                    if total > MAX_PERIODIC_SOURCE_RESIDUES:
                        raise ValueError(
                            "the family exceeds the 32,768-source-residue bound"
                        )
                    normalized_subset = dict(subset)
                    normalized_subset["residues"] = tuple(residues)
                    normalized_subsets.append(normalized_subset)
                else:
                    normalized_subsets.append(subset)
            else:
                normalized_subsets.append(subset)
        normalized_data = dict(data)
        normalized_data["subsets"] = tuple(normalized_subsets)
        return normalized_data

    @model_validator(mode="after")
    def require_canonical_bounded_family(self) -> Self:
        keys = tuple(_family_key(subset) for subset in self.subsets)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("residue subsets must be unique and canonically ordered")
        if sum(len(subset.residues) for subset in self.subsets) > (
            MAX_PERIODIC_SOURCE_RESIDUES
        ):
            raise ValueError("the family exceeds the 32,768-source-residue bound")
        _period, _writes, output_bound = _request_envelope(
            self.subsets, complement=self.complement
        )
        if (
            self.result_mode == "materialize_residues"
            and output_bound > MAX_PERIODIC_MATERIALIZED_RESIDUES
        ):
            raise ValueError(
                "the materialized result may exceed the 32,768-residue output bound; "
                "use count_only"
            )
        return self


def _union_occupancy(
    subsets: tuple[PeriodicResidueSubset, ...], common_period: int
) -> bytearray:
    """Mark the complete union in one admitted common period."""

    occupancy = bytearray(common_period)
    for modulus, residues in _aggregate_same_modulus(subsets):
        lift_count = common_period // modulus
        marks = b"\x01" * lift_count
        for residue in residues:
            occupancy[residue:common_period:modulus] = marks
    return occupancy


def _profile_values(
    request: PeriodicUnionProfileRequest,
) -> tuple[int, int, PeriodicResidueSubset | None]:
    common_period, _mark_writes, _output_bound = _request_envelope(
        request.subsets, complement=request.complement
    )
    union = _union_occupancy(request.subsets, common_period)
    occupied_subset: PeriodicResidueSubset | None = None
    if request.result_mode == "materialize_residues":
        target = 0 if request.complement else 1
        occupied_residues = tuple(
            residue for residue, membership in enumerate(union) if membership == target
        )
        occupied_count = len(occupied_residues)
        occupied_subset = PeriodicResidueSubset(
            modulus=common_period,
            residues=occupied_residues,
        )
    else:
        union_count = union.count(1)
        occupied_count = (
            common_period - union_count if request.complement else union_count
        )
    return common_period, occupied_count, occupied_subset


class PeriodicUnionProfileResult(StrictModel):
    """The source-bound complete exact profile over one common period."""

    semantics_version: Literal["periodic-congruence-union.v1"]
    source: PeriodicUnionProfileRequest = Field(
        description=(
            "The exact ordered source family, complement choice, and result mode "
            "retained for deterministic replay."
        )
    )
    common_period: StrictInt = Field(
        ge=1,
        le=MAX_PERIODIC_COMMON_PERIOD,
        description="The least common multiple of all declared source moduli.",
    )
    occupied_count: StrictInt = Field(
        ge=0,
        le=MAX_PERIODIC_COMMON_PERIOD,
        description="Exact occupied cardinality inside [0, common_period).",
    )
    density: CanonicalRational = Field(
        description="The reduced exact ratio occupied_count / common_period."
    )
    occupied_subset: PeriodicResidueSubset | None = Field(
        default=None,
        description=(
            "Canonical occupied subset with modulus equal to common_period in "
            "materialize_residues mode; null exactly in count_only mode."
        ),
    )
    completeness: Literal["COMPLETE_COMMON_PERIOD"]

    @model_validator(mode="after")
    def bind_complete_profile_to_source(self) -> Self:
        if self.occupied_count > self.common_period:
            raise ValueError("occupied count cannot exceed the common period")
        expected_period, expected_count, expected_subset = _profile_values(self.source)
        if self.common_period != expected_period:
            raise ValueError("common period must be the lcm of the source moduli")
        if self.occupied_count != expected_count:
            raise ValueError("occupied count must match the complete source union")
        expected_density = CanonicalRational.from_fraction(
            Fraction(self.occupied_count, self.common_period)
        )
        if self.density != expected_density:
            raise ValueError("density must be the reduced occupied-count ratio")
        if self.occupied_subset != expected_subset:
            raise ValueError(
                "occupied subset must exactly match the complete source union"
            )
        return self


def compute_periodic_union_profile(
    request: PeriodicUnionProfileRequest,
) -> PeriodicUnionProfileResult:
    """Compute one complete exact union or complement profile."""

    common_period, occupied_count, occupied_subset = _profile_values(request)
    return PeriodicUnionProfileResult(
        semantics_version="periodic-congruence-union.v1",
        source=request,
        common_period=common_period,
        occupied_count=occupied_count,
        density=CanonicalRational.from_fraction(
            Fraction(occupied_count, common_period)
        ),
        occupied_subset=occupied_subset,
        completeness="COMPLETE_COMMON_PERIOD",
    )


PERIODIC_UNION_OPERATION = number_theory_operation(
    "congruence.periodic_union.profile.compute",
    "Compute a finite periodic congruence-union profile",
    (
        "Lift a canonical finite family of residue subsets to its least common "
        "period and return the complete exact union or complement count, density, "
        "and optionally its canonical occupied subset. Same-modulus subsets are "
        "unioned before the bounded bytearray marking pass."
    ),
    PeriodicUnionProfileRequest,
    PeriodicUnionProfileResult,
    compute_periodic_union_profile,
    "number-theory",
    "congruence",
    "periodic",
    "union",
    "density",
    examples=(
        example(
            "overlapping_periodic_classes",
            (
                "Profile the union of three overlapping residue cylinders modulo "
                "6, 10, and 15; each residue list must be canonical, the family "
                "must be ordered by (modulus, residues), and materialization "
                "requires a proven output bound of at most 32,768 residues."
            ),
            {
                "subsets": [
                    {"modulus": 6, "residues": [1]},
                    {"modulus": 10, "residues": [4]},
                    {"modulus": 15, "residues": [9]},
                ],
                "result_mode": "materialize_residues",
            },
        ),
    ),
)


__all__ = [
    "MAX_PERIODIC_COMMON_PERIOD",
    "MAX_PERIODIC_FAMILY_SIZE",
    "MAX_PERIODIC_MARK_WRITES_PER_PASS",
    "MAX_PERIODIC_MATERIALIZED_RESIDUES",
    "MAX_PERIODIC_MODULUS",
    "MAX_PERIODIC_SOURCE_RESIDUES",
    "PERIODIC_UNION_OPERATION",
    "PeriodicResidueSubset",
    "PeriodicUnionProfileRequest",
    "PeriodicUnionProfileResult",
    "compute_periodic_union_profile",
]
