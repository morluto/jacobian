"""Exact contracts for labelled rational projective-line arrangements."""

from __future__ import annotations

from math import comb
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.geometry.projective.values import (
    PrimitiveProjectiveTriple,
    ProjectiveLabel,
    RationalProjectiveLine,
)

MAX_ARRANGEMENT_LINES = 64
MAX_ARRANGEMENT_PAIRS = comb(MAX_ARRANGEMENT_LINES, 2)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


class ProjectiveLineArrangementRequest(StrictModel):
    """A bounded labelled arrangement in the rational projective plane."""

    lines: tuple[RationalProjectiveLine, ...] = Field(
        min_length=2,
        max_length=MAX_ARRANGEMENT_LINES,
    )


class NormalizedProjectiveLine(StrictModel):
    label: ProjectiveLabel
    coefficients: PrimitiveProjectiveTriple


class ProjectiveArrangementFlat(StrictModel):
    point: PrimitiveProjectiveTriple
    incident_labels: tuple[ProjectiveLabel, ...] = Field(
        min_length=2,
        max_length=MAX_ARRANGEMENT_LINES,
    )
    multiplicity: int = Field(ge=2, le=MAX_ARRANGEMENT_LINES, strict=True)
    pair_count: int = Field(ge=1, le=MAX_ARRANGEMENT_PAIRS, strict=True)

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
    multiplicity: int = Field(ge=2, le=MAX_ARRANGEMENT_LINES, strict=True)
    flat_count: int = Field(ge=1, le=MAX_ARRANGEMENT_PAIRS, strict=True)


class ProjectiveLineArrangementResult(StrictModel):
    """Complete exact flat lattice at rank two for one labelled arrangement."""

    line_count: int = Field(ge=2, le=MAX_ARRANGEMENT_LINES, strict=True)
    normalized_lines: tuple[NormalizedProjectiveLine, ...] = Field(
        min_length=2,
        max_length=MAX_ARRANGEMENT_LINES,
    )
    flats: tuple[ProjectiveArrangementFlat, ...] = Field(
        min_length=1,
        max_length=MAX_ARRANGEMENT_PAIRS,
    )
    non_double_flats: tuple[tuple[ProjectiveLabel, ...], ...]
    multiplicity_histogram: tuple[ProjectiveMultiplicityCount, ...]
    pair_count_total: int = Field(ge=1, le=MAX_ARRANGEMENT_PAIRS, strict=True)
    arithmetic: Literal["EXACT_INTEGER"] = "EXACT_INTEGER"

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Construct output after the arrangement kernel establishes its lattice."""
        return cls.model_construct(**values)


__all__ = [
    "NormalizedProjectiveLine",
    "ProjectiveArrangementFlat",
    "ProjectiveLineArrangementRequest",
    "ProjectiveLineArrangementResult",
    "ProjectiveMultiplicityCount",
]
