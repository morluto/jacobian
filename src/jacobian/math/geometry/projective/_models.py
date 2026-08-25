"""Exact contracts for labelled rational projective-line arrangements."""

from __future__ import annotations

from math import comb
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.geometry.projective.values import (
    PrimitiveProjectiveTriple,
    ProjectiveLabel,
    RationalProjectiveLine,
    _primitive_integer_triple,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


class ProjectiveLineArrangementRequest(StrictModel):
    """A bounded labelled arrangement in the rational projective plane."""

    lines: tuple[RationalProjectiveLine, ...] = Field(
        min_length=2,
        max_length=64,
    )

    @model_validator(mode="after")
    def require_unique_labels_and_projective_lines(self) -> Self:
        labels = tuple(line.label for line in self.lines)
        if len(labels) != len(set(labels)):
            raise _validation_error(
                "projective_line_labels_unique", "projective line labels must be unique"
            )
        normalized = tuple(
            _primitive_integer_triple(line.coefficients) for line in self.lines
        )
        if len(normalized) != len(set(normalized)):
            raise _validation_error(
                "projectively_duplicate_lines_merged_before_invocation",
                "projectively duplicate lines must be merged before invocation",
            )
        return self


class NormalizedProjectiveLine(StrictModel):
    label: ProjectiveLabel
    coefficients: PrimitiveProjectiveTriple


class ProjectiveArrangementFlat(StrictModel):
    point: PrimitiveProjectiveTriple
    incident_labels: tuple[ProjectiveLabel, ...] = Field(
        min_length=2,
        max_length=64,
    )
    multiplicity: int = Field(ge=2, le=64, strict=True)
    pair_count: int = Field(ge=1, le=2016, strict=True)

    @model_validator(mode="after")
    def bind_incidence_multiplicity(self) -> Self:
        if self.incident_labels != tuple(sorted(set(self.incident_labels))):
            raise _validation_error(
                "flat_incident_labels_unique_sorted",
                "flat incident labels must be unique and sorted",
            )
        if self.multiplicity != len(self.incident_labels):
            raise _validation_error(
                "flat_multiplicity_incident_line_count",
                "flat multiplicity must equal its incident-line count",
            )
        if self.pair_count != comb(self.multiplicity, 2):
            raise _validation_error(
                "flat_pair_count_binomial_multiplicity",
                "flat pair_count must equal binomial(multiplicity, 2)",
            )
        return self


class ProjectiveMultiplicityCount(StrictModel):
    multiplicity: int = Field(ge=2, le=64, strict=True)
    flat_count: int = Field(ge=1, le=2016, strict=True)


class ProjectiveLineArrangementResult(StrictModel):
    """Complete exact flat lattice at rank two for one labelled arrangement."""

    line_count: int = Field(ge=2, le=64, strict=True)
    normalized_lines: tuple[NormalizedProjectiveLine, ...] = Field(
        min_length=2,
        max_length=64,
    )
    flats: tuple[ProjectiveArrangementFlat, ...] = Field(
        min_length=1,
        max_length=2016,
    )
    non_double_flats: tuple[tuple[ProjectiveLabel, ...], ...]
    multiplicity_histogram: tuple[ProjectiveMultiplicityCount, ...]
    pair_count_total: int = Field(ge=1, le=2016, strict=True)
    completion: Literal["COMPLETE"] = "COMPLETE"
    arithmetic: Literal["EXACT_INTEGER"] = "EXACT_INTEGER"

    @model_validator(mode="after")
    def bind_complete_arrangement_accounting(self) -> Self:
        labels = tuple(line.label for line in self.normalized_lines)
        if len(labels) != self.line_count or labels != tuple(sorted(set(labels))):
            raise _validation_error(
                "normalized_lines_line_count_sorted_labels",
                "normalized lines must match line_count and sorted labels",
            )
        points = tuple(flat.point.coordinates for flat in self.flats)
        if len(points) != len(set(points)):
            raise _validation_error(
                "projective_flats_unique_points",
                "projective flats must have unique points",
            )
        label_set = set(labels)
        if any(not set(flat.incident_labels) <= label_set for flat in self.flats):
            raise _validation_error(
                "flat_incidences_reference_supplied_line_labels",
                "flat incidences must reference supplied line labels",
            )
        expected_pair_count = comb(self.line_count, 2)
        if (
            self.pair_count_total != expected_pair_count
            or sum(flat.pair_count for flat in self.flats) != expected_pair_count
        ):
            raise _validation_error(
                "flat_multiplicities_account_every_line_pair",
                "flat multiplicities must account for every line pair",
            )
        expected_non_double = tuple(
            sorted(flat.incident_labels for flat in self.flats if flat.multiplicity > 2)
        )
        if self.non_double_flats != expected_non_double:
            raise _validation_error(
                "non_double_flats_project_higher_flats",
                "non_double_flats must exactly project higher flats",
            )
        histogram: dict[int, int] = {}
        for flat in self.flats:
            histogram[flat.multiplicity] = histogram.get(flat.multiplicity, 0) + 1
        expected_histogram = tuple(sorted(histogram.items()))
        supplied_histogram = tuple(
            (item.multiplicity, item.flat_count) for item in self.multiplicity_histogram
        )
        if supplied_histogram != expected_histogram:
            raise _validation_error(
                "multiplicity_histogram_flat_list",
                "multiplicity histogram does not match the flat list",
            )
        return self


__all__ = [
    "NormalizedProjectiveLine",
    "ProjectiveArrangementFlat",
    "ProjectiveLineArrangementRequest",
    "ProjectiveLineArrangementResult",
    "ProjectiveMultiplicityCount",
]
