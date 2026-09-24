"""Exact relabelling and axis transport for finite delta-matroids."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
    canonical_feasible_rows,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_envelope,
    require_delta_matroid_exchange_work,
)

MAX_DELTA_RELABEL_GROUND = MAX_DELTA_LABEL_BYTES + 1
MAX_DELTA_RELABEL_TRANSPORT_WORK = (
    MAX_DELTA_MEMBERSHIPS * ((MAX_DELTA_LABEL_BYTES + 1).bit_length() + 1)
    + 4 * MAX_DELTA_RELABEL_GROUND * ((MAX_DELTA_RELABEL_GROUND).bit_length() + 2)
    + MAX_DELTA_LABEL_BYTES
)
MAX_DELTA_RELABEL_OUTPUT_BYTES = 2_000_000
MAX_DELTA_RELABEL_WORK = (
    MAX_DELTA_RELABEL_TRANSPORT_WORK
    + 2 * MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
    + MAX_DELTA_RELABEL_OUTPUT_BYTES
)


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"delta_matroid.relabel_{reason}", message)


class DeltaMatroidRelabelRequest(StrictModel):
    """Relabel the ground axis and specify its target-to-source permutation."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Relabel and reorder a complete finite delta-matroid using a "
                "bijection between ground axes. The operation admits at most "
                f"{MAX_DELTA_RELABEL_GROUND} ground elements, "
                f"{MAX_DELTA_LABEL_BYTES} UTF-8 target-label bytes, "
                f"{MAX_DELTA_RELABEL_TRANSPORT_WORK} transport work units, "
                f"{MAX_DELTA_RELABEL_WORK} total work units, and "
                f"{MAX_DELTA_RELABEL_OUTPUT_BYTES} output bytes."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_DELTA_RELABEL_GROUND,
                "max_target_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_transport_work_units": MAX_DELTA_RELABEL_TRANSPORT_WORK,
                "max_total_work_units": MAX_DELTA_RELABEL_WORK,
                "max_output_bytes": MAX_DELTA_RELABEL_OUTPUT_BYTES,
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Complete canonical source family; relabel admission also uses the "
            f"existing {MAX_DELTA_MEMBERSHIPS}-membership and "
            f"{MAX_DELTA_LABEL_BYTES}-byte source-label bounds."
        )
    )
    target_ground: tuple[str, ...] = Field(
        max_length=MAX_DELTA_RELABEL_GROUND,
        description=(
            "Unique labels in target-axis order, with at most "
            f"{MAX_DELTA_LABEL_BYTES} aggregate UTF-8 bytes."
        ),
    )
    target_to_source: tuple[int, ...] = Field(
        max_length=MAX_DELTA_RELABEL_GROUND,
        description=(
            "Permutation specifying the source-axis index at each target-axis "
            "position; must have the same length as target_ground."
        ),
    )

    @model_validator(mode="after")
    def require_bijection_and_bounded_labels(self) -> Self:
        n = len(self.delta_matroid.ground)
        if n > MAX_DELTA_RELABEL_GROUND:
            raise _error(
                "ground_limit",
                f"relabeling supports at most {MAX_DELTA_RELABEL_GROUND} ground elements",
            )
        if len(self.target_ground) != n or len(self.target_to_source) != n:
            raise _error(
                "axis_shape",
                "target labels and target-to-source map must match the source axis",
            )
        if any(type(index) is not int for index in self.target_to_source):
            raise _error("map_type", "axis-map entries must be exact integers")
        if tuple(sorted(self.target_to_source)) != tuple(range(n)):
            raise _error(
                "map_bijection", "target-to-source entries must be a permutation"
            )
        if len(set(self.target_ground)) != n:
            raise _error("ground_unique", "target ground labels must be unique")
        try:
            label_bytes = sum(
                len(label.encode("utf-8")) for label in self.target_ground
            )
        except UnicodeEncodeError:
            raise _error(
                "ground_utf8", "target labels must be UTF-8 representable"
            ) from None
        if label_bytes > MAX_DELTA_LABEL_BYTES:
            raise _error(
                "ground_bytes",
                f"target labels exceed the {MAX_DELTA_LABEL_BYTES}-byte envelope",
            )
        return self


class DeltaMatroidRelabelling(StrictModel):
    """Source, relabelled value, and inverse maps between their ground axes."""

    source: FiniteDeltaMatroid
    relabelled: FiniteDeltaMatroid
    target_to_source: tuple[int, ...]
    source_to_target: tuple[int, ...]

    @model_validator(mode="after")
    def require_inverse_axis_maps(self) -> Self:
        n = len(self.source.ground)
        if len(self.relabelled.ground) != n:
            raise _error("result_shape", "source and target ground axes must agree")
        if any(
            type(index) is not int
            for index in (*self.target_to_source, *self.source_to_target)
        ):
            raise _error("result_map_type", "result axis maps must use exact integers")
        if tuple(sorted(self.target_to_source)) != tuple(range(n)) or tuple(
            sorted(self.source_to_target)
        ) != tuple(range(n)):
            raise _error("result_bijection", "result maps must be axis permutations")
        if any(
            self.source_to_target[self.target_to_source[target]] != target
            for target in range(n)
        ):
            raise _error("result_inverse", "result axis maps must be inverses")
        return self


def _output_estimate(
    source: FiniteDeltaMatroid,
    target_ground: tuple[str, ...],
    target_to_source: tuple[int, ...],
    source_to_target: tuple[int, ...],
) -> tuple[int, int]:
    """Bound serialized output and the linear admission-encoding work."""

    memberships = sum(len(row) for row in source.feasible)
    rows = len(source.feasible)
    n = len(source.ground)
    index_digits = max(1, len(str(max(0, n - 1))))
    transported_rows = (
        2 * rows + memberships * index_digits + max(0, memberships - rows)
    )
    source_wire = {
        "ground": list(source.ground),
        "feasible": [list(row) for row in source.feasible],
    }
    source_bytes = len(encode_strict_json(source_wire))
    labels_bytes = len(encode_strict_json(list(target_ground)))
    forward_map_bytes = len(encode_strict_json(list(target_to_source)))
    inverse_map_bytes = len(encode_strict_json(list(source_to_target)))
    return (
        source_bytes
        + labels_bytes
        + forward_map_bytes
        + inverse_map_bytes
        + transported_rows
        + 256
    ), source_bytes + labels_bytes + forward_map_bytes + inverse_map_bytes


def relabel(request: DeltaMatroidRelabelRequest) -> DeltaMatroidRelabelling:
    """Transport a complete feasible family through a ground-axis bijection."""

    if type(request) is not DeltaMatroidRelabelRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="delta_matroid.relabel_request",
            message="request must be a canonical relabeling request",
        )
    try:
        request = DeltaMatroidRelabelRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="delta_matroid.relabel_request",
            message="relabeling request is malformed",
        ) from exc
    try:
        system = FiniteFeasibleSetSystem(
            ground=request.delta_matroid.ground,
            feasible=request.delta_matroid.feasible,
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message="source feasible family is malformed",
        ) from exc
    if request.delta_matroid.feasible != canonical_feasible_rows(system):
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message="source feasible rows must be canonical",
        )
    source = request.delta_matroid

    try:
        require_delta_matroid_envelope(system)
    except DeltaMatroidAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc

    n = len(source.ground)
    source_to_target_list = [0] * n
    for target, origin in enumerate(request.target_to_source):
        source_to_target_list[origin] = target
    source_to_target = tuple(source_to_target_list)

    memberships = sum(len(row) for row in source.feasible)
    transport_work = (
        memberships * (max(1, n.bit_length()) + 1)
        + 4 * n * (max(1, n.bit_length()) + 2)
        + sum(len(label.encode("utf-8")) for label in request.target_ground)
    )
    if transport_work > MAX_DELTA_RELABEL_TRANSPORT_WORK:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.relabel_work",
            message="ground-axis transport exceeds its admitted work bound",
        )
    output_bytes, admission_encoding_work = _output_estimate(
        source,
        request.target_ground,
        request.target_to_source,
        source_to_target,
    )
    # Reserve the full source-exchange envelope for both bounded admission and
    # recognition scans; the exact source request may use less, but this keeps
    # all mandatory phases within one operation-specific ceiling.
    total_work = (
        transport_work
        + admission_encoding_work
        + 2 * MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
    )
    if total_work > MAX_DELTA_RELABEL_WORK:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.relabel_work",
            message="ground-axis transport and output admission exceed their work bound",
        )
    if output_bytes > MAX_DELTA_RELABEL_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.relabel_output",
            message="relabelling result exceeds its admitted output bound",
        )

    try:
        require_delta_matroid_exchange_work(system)
    except DeltaMatroidAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
    if first_symmetric_exchange_obstruction(system) is not None:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_delta",
            message="source is not a delta-matroid",
        )

    rows = tuple(
        sorted(
            tuple(sorted(source_to_target[index] for index in row))
            for row in source.feasible
        )
    )
    result_value = FiniteDeltaMatroid.model_construct(
        ground=request.target_ground,
        feasible=rows,
    )
    return DeltaMatroidRelabelling(
        source=source,
        relabelled=result_value,
        target_to_source=request.target_to_source,
        source_to_target=source_to_target,
    )


__all__ = [
    "MAX_DELTA_RELABEL_GROUND",
    "MAX_DELTA_RELABEL_OUTPUT_BYTES",
    "MAX_DELTA_RELABEL_TRANSPORT_WORK",
    "MAX_DELTA_RELABEL_WORK",
    "DeltaMatroidRelabelRequest",
    "DeltaMatroidRelabelling",
    "relabel",
]
