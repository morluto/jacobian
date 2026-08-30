"""Strict private protocol for the QEPCAD plane-component worker."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, TypeAdapter, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_SAMPLES,
    MAX_PLANE_COMPONENTS,
    IsolatedRealPlanePoint,
    PlaneComponentProfileRequest,
    _plane_point_identity_key,
)

MAX_QEPCAD_TRUE_CELLS = 512
MAX_QEPCAD_CLOSURE_CELLS = 2_048
MAX_QEPCAD_SAMPLE_CHARACTERS = 96 * 1024
# Every structurally admitted plane point fits this reservation, including two
# degree-sixteen coordinate polynomials and four 8,192-digit rational endpoint
# components. The aggregate worker response remains subject to the canonical
# transport limit and may therefore stop before all 128 component slots fill.
MAX_QEPCAD_POINT_JSON_BYTES = 96 * 1024
MAX_QEPCAD_WORKER_RESPONSE_BYTES = CanonicalLimits().max_output_bytes


class QepcadPlaneWorkerRequest(StrictModel):
    """One parent-admitted exact component-profile transaction."""

    kind: Literal["components"] = "components"
    executable: str = Field(min_length=1, max_length=4_096)
    qepcad_root: str = Field(min_length=1, max_length=4_096)
    deadline_monotonic: float = Field(gt=0, allow_inf_nan=False)
    request: PlaneComponentProfileRequest


class PlaneSampleWorkerRequest(StrictModel):
    """Recognize samples for a backend-free degenerate component profile."""

    kind: Literal["samples"] = "samples"
    samples: tuple[IsolatedRealPlanePoint, ...] = Field(
        max_length=MAX_PLANE_COMPONENT_SAMPLES
    )


PlaneWorkerRequest = Annotated[
    QepcadPlaneWorkerRequest | PlaneSampleWorkerRequest,
    Field(discriminator="kind"),
]
PLANE_WORKER_REQUEST_ADAPTER: TypeAdapter[PlaneWorkerRequest] = TypeAdapter(
    PlaneWorkerRequest
)


class QepcadPlaneCell(StrictModel):
    """One original true CAD cell and its exact backend sample description."""

    index: tuple[StrictInt, StrictInt]
    dimension: StrictInt = Field(ge=0, le=2)
    sample: str = Field(min_length=1, max_length=MAX_QEPCAD_SAMPLE_CHARACTERS)

    @model_validator(mode="after")
    def require_positive_index(self) -> Self:
        if any(index <= 0 for index in self.index):
            raise ValueError("QEPCAD cell indices must be positive")
        return self


class QepcadPlaneCellClosure(StrictModel):
    """Exact CAD cells in the closure of one original true cell."""

    cell_index: tuple[StrictInt, StrictInt]
    closure_indices: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        min_length=1,
        max_length=MAX_QEPCAD_CLOSURE_CELLS,
    )

    @model_validator(mode="after")
    def require_canonical_closure(self) -> Self:
        if any(index <= 0 for cell in self.closure_indices for index in cell):
            raise ValueError("QEPCAD closure indices must be positive")
        if self.closure_indices != tuple(sorted(set(self.closure_indices))):
            raise ValueError("QEPCAD closure indices must be unique and ordered")
        if self.cell_index not in self.closure_indices:
            raise ValueError("a CAD cell must belong to its own closure")
        return self


class QepcadPlaneWorkerComplete(StrictModel):
    """A compact exact projection bound to the retained parent request."""

    kind: Literal["complete"] = "complete"
    version: Literal["1.74"]
    representatives: tuple[IsolatedRealPlanePoint, ...] = Field(
        max_length=MAX_PLANE_COMPONENTS
    )
    sample_component_ids: tuple[StrictInt | None, ...] = Field(
        max_length=MAX_PLANE_COMPONENT_SAMPLES
    )

    @model_validator(mode="after")
    def bind_component_references(self) -> Self:
        representative_keys = tuple(
            _plane_point_identity_key(representative)
            for representative in self.representatives
        )
        if representative_keys != tuple(sorted(set(representative_keys))):
            raise ValueError(
                "worker representatives must be unique and canonically ordered"
            )
        if any(
            component_id is not None
            and not 0 <= component_id < len(self.representatives)
            for component_id in self.sample_component_ids
        ):
            raise ValueError("worker sample component ID is outside the profile")
        return self


class QepcadPlaneWorkerRejected(StrictModel):
    """A bounded worker stopped without making a topological conclusion."""

    kind: Literal["rejected"] = "rejected"
    reason: Literal[
        "UNSUPPORTED_QEPCAD_VERSION",
        "QEPCAD_DEADLINE_EXPIRED",
        "QEPCAD_OUTPUT_LIMIT",
        "QEPCAD_CELL_LIMIT",
        "QEPCAD_INVALID_OUTPUT",
        "QEPCAD_EXECUTION_FAILED",
    ]


class QepcadPlaneWorkerInvalid(StrictModel):
    """A supplied algebraic sample failed exact bounded recognition."""

    kind: Literal["invalid"] = "invalid"
    reason: Literal["SAMPLE_NOT_ISOLATED", "SAMPLE_RECOGNITION_LIMIT"]


class PlaneSamplesValid(StrictModel):
    """Every supplied structural sample denotes its declared isolated point."""

    kind: Literal["samples_valid"] = "samples_valid"


QepcadPlaneWorkerResponse = Annotated[
    QepcadPlaneWorkerComplete
    | QepcadPlaneWorkerInvalid
    | QepcadPlaneWorkerRejected
    | PlaneSamplesValid,
    Field(discriminator="kind"),
]
QEPCAD_PLANE_WORKER_RESPONSE_ADAPTER: TypeAdapter[QepcadPlaneWorkerResponse] = (
    TypeAdapter(QepcadPlaneWorkerResponse)
)


__all__ = [
    "MAX_QEPCAD_CLOSURE_CELLS",
    "MAX_QEPCAD_POINT_JSON_BYTES",
    "MAX_QEPCAD_SAMPLE_CHARACTERS",
    "MAX_QEPCAD_TRUE_CELLS",
    "MAX_QEPCAD_WORKER_RESPONSE_BYTES",
    "PLANE_WORKER_REQUEST_ADAPTER",
    "QEPCAD_PLANE_WORKER_RESPONSE_ADAPTER",
    "PlaneSampleWorkerRequest",
    "PlaneSamplesValid",
    "PlaneWorkerRequest",
    "QepcadPlaneCell",
    "QepcadPlaneCellClosure",
    "QepcadPlaneWorkerComplete",
    "QepcadPlaneWorkerInvalid",
    "QepcadPlaneWorkerRejected",
    "QepcadPlaneWorkerRequest",
    "QepcadPlaneWorkerResponse",
]
