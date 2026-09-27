"""Exact relabelling and axis transport for finite delta-matroids."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
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
# Count retained labels, map entries, feasible-set rows, and memberships in the
# source-bound result. This bounds mathematical materialization rather than a
# particular transport encoding.
MAX_DELTA_RELABEL_OUTPUT_CELLS = (
    4 * MAX_DELTA_MEMBERSHIPS + 4 * MAX_DELTA_RELABEL_GROUND + 8
)
MAX_DELTA_RELABEL_WORK = (
    MAX_DELTA_RELABEL_TRANSPORT_WORK
    + 2 * MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
    + MAX_DELTA_RELABEL_OUTPUT_CELLS
)


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"delta_matroid.relabel_{reason}", message)


def _bounded_utf8_length(label: str) -> int | None:
    """Count a bounded UTF-8 label without allocating an encoded copy."""
    if type(label) is not str or len(label) > MAX_DELTA_LABEL_BYTES:
        return MAX_DELTA_LABEL_BYTES + 1
    size = 0
    for character in label:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            return None
        size += (
            1
            if codepoint <= 0x7F
            else 2
            if codepoint <= 0x7FF
            else 3
            if codepoint <= 0xFFFF
            else 4
        )
        if size > MAX_DELTA_LABEL_BYTES:
            return MAX_DELTA_LABEL_BYTES + 1
    return size


class DeltaMatroidRelabelRequest(StrictModel):
    """Relabel the ground axis and specify its target-to-source permutation."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Relabel and reorder a complete finite delta-matroid using a "
                "bijection between ground axes. The operation admits at most "
                f"{MAX_DELTA_RELABEL_GROUND} ground elements, "
                f"{MAX_DELTA_LABEL_BYTES} UTF-8 target-label bytes, "
                f"{MAX_DELTA_RELABEL_TRANSPORT_WORK} relabelling work units, "
                f"{MAX_DELTA_RELABEL_WORK} total work units, and "
                f"{MAX_DELTA_RELABEL_OUTPUT_CELLS} materialized result cells."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_DELTA_RELABEL_GROUND,
                "max_target_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_transport_work_units": MAX_DELTA_RELABEL_TRANSPORT_WORK,
                "max_total_work_units": MAX_DELTA_RELABEL_WORK,
                "max_output_cells": MAX_DELTA_RELABEL_OUTPUT_CELLS,
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
    target_to_source: tuple[StrictInt, ...] = Field(
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
        label_sizes = tuple(_bounded_utf8_length(label) for label in self.target_ground)
        if any(size is None for size in label_sizes):
            raise _error("ground_utf8", "target labels must be UTF-8 representable")
        if (
            sum(size for size in label_sizes if size is not None)
            > MAX_DELTA_LABEL_BYTES
        ):
            raise _error(
                "ground_bytes", "target labels exceed the admitted UTF-8 byte bound"
            )
        return self


class DeltaMatroidRelabelling(StrictModel):
    """Source, relabelled value, and inverse maps between their ground axes."""

    source: FiniteDeltaMatroid
    relabelled: FiniteDeltaMatroid
    target_to_source: tuple[StrictInt, ...]
    source_to_target: tuple[StrictInt, ...]

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


def _output_cell_count(source: FiniteDeltaMatroid) -> int:
    """Count scalar and row allocations in the source-bound result."""

    memberships = sum(len(row) for row in source.feasible)
    rows = len(source.feasible)
    n = len(source.ground)
    # Both source and target retain their ground labels, feasible row
    # containers, and memberships; the result also retains two n-entry maps.
    return 2 * n + 2 * rows + 2 * memberships + 2 * n + 4


def relabel(
    delta_matroid: FiniteDeltaMatroid,
    target_ground: tuple[str, ...],
    target_to_source: tuple[int, ...],
) -> DeltaMatroidRelabelling:
    """Transport a complete feasible family through a ground-axis bijection."""

    try:
        request = DeltaMatroidRelabelRequest(
            delta_matroid=delta_matroid,
            target_ground=target_ground,
            target_to_source=target_to_source,
        )
    except Exception as exc:
        if isinstance(exc, ValidationError) and any(
            error["type"] == "delta_matroid.relabel_ground_bytes"
            for error in exc.errors()
        ):
            raise OperationResourceAdmissionError(
                location=("target_ground",),
                code="delta_matroid.relabel_target_bytes",
                message="target labels exceed the admitted UTF-8 byte bound",
            ) from exc
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
    # The request may carry a forged ``FiniteDeltaMatroid`` instance whose
    # fields bypassed validation (for example a list ground). Retain the
    # canonical carrier reconstructed from the validated feasible system so
    # the returned source cannot expose noncanonical mutable fields.
    source = FiniteDeltaMatroid._from_kernel(system)

    try:
        require_delta_matroid_envelope(system)
    except DeltaMatroidAdmissionError as exc:
        error_type = (
            OperationResourceAdmissionError
            if exc.reason in {"memberships_exceeded", "label_bytes_exceeded"}
            else OperationDomainValidationError
        )
        raise error_type(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc

    # The request validator already admitted the bounded target-label bytes.
    # Recount only to price the exact transport work, not to classify input.
    target_bytes = sum(
        _bounded_utf8_length(label) or 0 for label in request.target_ground
    )
    n = len(source.ground)
    source_to_target_list = [0] * n
    for target, origin in enumerate(request.target_to_source):
        request_checkpoint("during delta-matroid relabelling")
        source_to_target_list[origin] = target
    source_to_target = tuple(source_to_target_list)

    memberships = sum(len(row) for row in source.feasible)
    transport_work = (
        memberships * (max(1, n.bit_length()) + 1)
        + 4 * n * (max(1, n.bit_length()) + 2)
        + target_bytes
    )
    if transport_work > MAX_DELTA_RELABEL_TRANSPORT_WORK:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.relabel_work",
            message="ground-axis transport exceeds its admitted work bound",
        )
    output_cells = _output_cell_count(source)
    # Reserve the full source-exchange envelope for both bounded admission and
    # recognition scans; the exact source request may use less, but this keeps
    # all mandatory phases within one operation-specific ceiling.
    total_work = transport_work + output_cells + 2 * MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
    if total_work > MAX_DELTA_RELABEL_WORK:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.relabel_work",
            message="ground-axis transport and output admission exceed their work bound",
        )
    if output_cells > MAX_DELTA_RELABEL_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.relabel_output",
            message="relabelling result exceeds its admitted allocation bound",
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
    request_checkpoint("before delta-matroid relabelling result construction")
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
    "MAX_DELTA_RELABEL_OUTPUT_CELLS",
    "MAX_DELTA_RELABEL_TRANSPORT_WORK",
    "MAX_DELTA_RELABEL_WORK",
    "DeltaMatroidRelabelRequest",
    "DeltaMatroidRelabelling",
    "relabel",
]
