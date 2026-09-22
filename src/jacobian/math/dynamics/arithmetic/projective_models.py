from __future__ import annotations

from typing import Literal

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.dynamics.arithmetic.projective._kernel import (
    HomogeneousProjectiveMap,
    ProjectiveOrbitResult,
    ProjectivePoint,
)


class ProjectiveMapRequest(StrictModel):
    map: HomogeneousProjectiveMap


class ProjectiveApplyRequest(StrictModel):
    map: HomogeneousProjectiveMap
    point: ProjectivePoint


class ProjectivePointResult(StrictModel):
    map: HomogeneousProjectiveMap
    point: ProjectivePoint
    image: ProjectivePoint


class ProjectiveComposeRequest(StrictModel):
    outer: HomogeneousProjectiveMap
    inner: HomogeneousProjectiveMap


class ProjectiveMapResult(StrictModel):
    outer: HomogeneousProjectiveMap
    inner: HomogeneousProjectiveMap
    composition: HomogeneousProjectiveMap


class ProjectiveOrbitRequest(StrictModel):
    map: HomogeneousProjectiveMap
    start: ProjectivePoint
    max_steps: int = Field(ge=0, le=256)


class CriticalOrbitRequest(StrictModel):
    map: HomogeneousProjectiveMap
    critical_points: tuple[ProjectivePoint, ...] = Field(min_length=1, max_length=64)
    max_steps: int = Field(ge=0, le=256)


class CriticalOrbitRow(StrictModel):
    point: ProjectivePoint
    orbit: tuple[ProjectivePoint, ...]
    termination: Literal["REPEAT_FOUND", "STEP_BOUND_REACHED"]
    period: int | None = None


class CriticalOrbitResult(StrictModel):
    map: HomogeneousProjectiveMap
    rows: tuple[CriticalOrbitRow, ...]


class ExactPeriodResult(ProjectiveOrbitResult):
    pass


__all__ = [
    "CriticalOrbitRequest",
    "CriticalOrbitResult",
    "CriticalOrbitRow",
    "ExactPeriodResult",
    "ProjectiveApplyRequest",
    "ProjectiveComposeRequest",
    "ProjectiveMapRequest",
    "ProjectiveMapResult",
    "ProjectiveOrbitRequest",
    "ProjectivePointResult",
]
