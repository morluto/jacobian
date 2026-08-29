"""Exact native operations on finite families of rational boxes."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.boxes._kernel import (
    complete_intersection_ledger,
    wire_rational,
)
from jacobian.math.geometry.boxes._models import (
    MAX_BOX_UNION_RESULT_BYTES,
    MAX_BOX_UNION_RESULT_RATIONAL_DIGITS,
    MAX_INTERSECTION_CANDIDATES,
    BoxIntersectionLedgerEntry,
    BoxUnionVolumeResult,
)
from jacobian.math.geometry.boxes.values import RationalAxisAlignedBox

_BOX_UNION_RESULT_FIXED_BYTES = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"geometry.{reason}", message)


def _axis_endpoints(
    boxes: tuple[RationalAxisAlignedBox, ...], axis: int
) -> tuple[CanonicalRational, ...]:
    return tuple(
        endpoint
        for box in boxes
        if box.intervals is not None
        for endpoint in (box.intervals[axis].lower, box.intervals[axis].upper)
    )


def _digit_bounds(
    boxes: tuple[RationalAxisAlignedBox, ...], dimension: int, candidate_count: int
) -> tuple[int, int, int, int]:
    maximum_numerator_digits = 1
    maximum_denominator_digits = 1
    volume_numerator_digits = 1
    volume_denominator_digits = 1
    common_denominator_digits = 1
    for axis in range(dimension):
        endpoints = _axis_endpoints(boxes, axis)
        numerator_digits = max(
            (len(value.num.lstrip("-")) for value in endpoints), default=1
        )
        denominator_digits = max((len(value.den) for value in endpoints), default=1)
        maximum_numerator_digits = max(maximum_numerator_digits, numerator_digits)
        maximum_denominator_digits = max(maximum_denominator_digits, denominator_digits)
        volume_numerator_digits += numerator_digits + denominator_digits + 1
        volume_denominator_digits += 2 * denominator_digits
        common_denominator_digits += sum(
            len(value) for value in {endpoint.den for endpoint in endpoints}
        )
    union_numerator_digits = (
        volume_numerator_digits
        + common_denominator_digits
        + len(str(candidate_count))
        + 2
    )
    union_denominator_digits = common_denominator_digits + 1
    return (
        maximum_numerator_digits,
        maximum_denominator_digits,
        max(volume_numerator_digits, volume_denominator_digits),
        max(union_numerator_digits, union_denominator_digits),
    )


def _maximum_result_bytes(
    boxes: tuple[RationalAxisAlignedBox, ...],
    *,
    candidate_count: int,
    endpoint_numerator_digits: int,
    endpoint_denominator_digits: int,
    volume_digits: int,
    union_digits: int,
) -> int:
    endpoint = {
        "num": "-" + "9" * endpoint_numerator_digits,
        "den": "9" * endpoint_denominator_digits,
    }
    interval = {"lower": endpoint, "upper": endpoint}
    maximum_entry = {
        "box_indices": [len(boxes) - 1] * sum(not box.is_empty for box in boxes),
        "intersection": {
            "dimension": boxes[0].dimension,
            "intervals": [interval] * boxes[0].dimension,
        },
        "volume": {"num": "-" + "9" * volume_digits, "den": "9" * volume_digits},
    }
    result_header = {
        "source": [box.model_dump(mode="json") for box in boxes],
        "union_volume": {
            "num": "-" + "9" * union_digits,
            "den": "9" * union_digits,
        },
        "intersections": [],
    }
    return (
        len(encode_strict_json(result_header))
        + candidate_count * (len(encode_strict_json(maximum_entry)) + 1)
        + _BOX_UNION_RESULT_FIXED_BYTES
    )


def admit_box_union_volume(boxes: tuple[RationalAxisAlignedBox, ...]) -> None:
    if not boxes:
        raise _validation_error(
            "box_union_source_empty", "box union requires at least one source box"
        )
    dimension = boxes[0].dimension
    if any(box.dimension != dimension for box in boxes):
        raise _validation_error(
            "box_union_sources_same_dimension",
            "all box-union sources must have the same dimension",
        )
    active_box_count = sum(not box.is_empty for box in boxes)
    candidate_count = (1 << active_box_count) - 1
    if candidate_count > MAX_INTERSECTION_CANDIDATES:
        raise _validation_error(
            "active_box_count_nonempty_boxes_require",
            f"{active_box_count} nonempty boxes require {candidate_count} "
            "intersection candidates, exceeding the complete "
            f"{MAX_INTERSECTION_CANDIDATES}-candidate bound",
        )
    endpoint_num, endpoint_den, volume_digits, union_digits = _digit_bounds(
        boxes, dimension, candidate_count
    )
    result_digits = max(volume_digits, union_digits)
    if result_digits > min(
        MAX_BOX_UNION_RESULT_RATIONAL_DIGITS, MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise _validation_error(
            "box_union_exceed_exact_rational_intermediate",
            "box union can exceed the exact rational intermediate bound "
            f"({result_digits} digits > {MAX_BOX_UNION_RESULT_RATIONAL_DIGITS})",
        )
    estimated_bytes = _maximum_result_bytes(
        boxes,
        candidate_count=candidate_count,
        endpoint_numerator_digits=endpoint_num,
        endpoint_denominator_digits=endpoint_den,
        volume_digits=volume_digits,
        union_digits=union_digits,
    )
    if estimated_bytes > MAX_BOX_UNION_RESULT_BYTES:
        raise _validation_error(
            "box_union_exceed_complete_intersection_ledger",
            "box union can exceed the complete intersection-ledger result "
            f"budget ({estimated_bytes} bytes > {MAX_BOX_UNION_RESULT_BYTES})",
        )


def _admit(boxes: tuple[RationalAxisAlignedBox, ...]) -> None:
    try:
        admit_box_union_volume(boxes)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("boxes",), code=exc.type, message=exc.message()
        ) from exc


def _union_volume_from_source(
    boxes: tuple[RationalAxisAlignedBox, ...],
) -> BoxUnionVolumeResult:
    """Compute the complete ledger for one admitted box family."""
    _admit(boxes)
    records, union_volume = complete_intersection_ledger(boxes)
    return BoxUnionVolumeResult._from_kernel(
        source=boxes,
        intersections=tuple(
            BoxIntersectionLedgerEntry(
                box_indices=record.box_indices,
                intersection=record.intersection,
                volume=wire_rational(record.volume),
            )
            for record in records
        ),
        union_volume=wire_rational(union_volume),
    )


def compute_box_union_volume(
    boxes: tuple[RationalAxisAlignedBox, ...],
) -> BoxUnionVolumeResult:
    """Return exact union volume and the complete inclusion-exclusion ledger."""
    return _union_volume_from_source(boxes)


__all__ = ["compute_box_union_volume"]
