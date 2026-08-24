"""Typed contracts for exact finite unions of rational boxes."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.boxes._kernel import box_volume
from jacobian.math.geometry.boxes.values import RationalAxisAlignedBox

MAX_BOX_UNION_NONEMPTY_BOXES = 16
MAX_INTERSECTION_CANDIDATES = (1 << MAX_BOX_UNION_NONEMPTY_BOXES) - 1
MAX_BOX_UNION_REPLAY_WORK = 4_000_000
MAX_BOX_UNION_RESULT_BYTES = 8 * 1024 * 1024
MAX_BOX_UNION_RESULT_RATIONAL_DIGITS = 16_384


def _axis_endpoints(
    boxes: tuple[RationalAxisAlignedBox, ...],
    axis: int,
) -> tuple[CanonicalRational, ...]:
    return tuple(
        endpoint
        for box in boxes
        if box.intervals is not None
        for endpoint in (box.intervals[axis].lower, box.intervals[axis].upper)
    )


def _digit_bounds(
    boxes: tuple[RationalAxisAlignedBox, ...],
    dimension: int,
    candidate_count: int,
) -> tuple[int, int, int, int]:
    maximum_numerator_digits = 1
    maximum_denominator_digits = 1

    # For one width a/b-c/d, numerator and denominator carry at most
    # num_digits+den_digits+1 and 2*den_digits digits. Multiplication across
    # the declared axes sums these per-axis bounds, so one large axis cannot
    # multiply its growth onto every other axis.
    volume_numerator_digits = 1
    volume_denominator_digits = 1
    # On each axis, every width denominator divides the product of the distinct
    # source endpoint denominators on that axis. The common denominator for all
    # box-cell volumes therefore divides the product of those per-axis products.
    common_denominator_digits = 1
    for axis in range(dimension):
        endpoints = _axis_endpoints(boxes, axis)
        numerator_digits = max(
            (len(value.num.lstrip("-")) for value in endpoints),
            default=1,
        )
        denominator_digits = max(
            (len(value.den) for value in endpoints),
            default=1,
        )
        maximum_numerator_digits = max(maximum_numerator_digits, numerator_digits)
        maximum_denominator_digits = max(maximum_denominator_digits, denominator_digits)
        volume_numerator_digits += numerator_digits + denominator_digits + 1
        volume_denominator_digits += 2 * denominator_digits
        common_denominator_digits += sum(
            len(value) for value in {endpoint.den for endpoint in endpoints}
        )

    # A partial inclusion-exclusion sum has at most candidate_count terms.
    # This additionally bounds Fraction intermediates before canonical output.
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
    request: BoxUnionVolumeRequest,
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
        "box_indices": [len(request.boxes) - 1]
        * sum(not box.is_empty for box in request.boxes),
        "intersection": {
            "dimension": request.boxes[0].dimension,
            "intervals": [interval] * request.boxes[0].dimension,
        },
        "volume": {"num": "-" + "9" * volume_digits, "den": "9" * volume_digits},
    }
    result_header = {
        "source": request.model_dump(mode="json"),
        "union_volume": {
            "num": "-" + "9" * union_digits,
            "den": "9" * union_digits,
        },
        "intersections": [],
    }
    return (
        len(encode_strict_json(result_header))
        + candidate_count * (len(encode_strict_json(maximum_entry)) + 1)
        + 4_096
    )


class BoxUnionVolumeRequest(StrictModel):
    """A bounded ordered family of rational boxes in one ambient space."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "An ordered family of one or more rational boxes; the "
                "admitted source count is bounded by the serialized-result "
                "budget because every result echoes its full source family. "
                "Every box must use the same dimension in [1,64]. Each "
                "endpoint component obeys the canonical 32,768-digit "
                "rational limit, and admission rests on the coupled exact "
                "growth and serialized-byte budgets rather than a "
                "per-endpoint cap. intervals=null "
                "denotes the canonical empty box; at most 16 boxes may be "
                "nonempty, and equal interval endpoints are valid measure-zero "
                "axes."
            ),
            "examples": [
                {
                    "boxes": [
                        {
                            "dimension": 1,
                            "intervals": [
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "1", "den": "1"},
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    )

    boxes: tuple[RationalAxisAlignedBox, ...] = Field(
        min_length=1,
        description=(
            "Ordered, indexed boxes with a common dimension in [1,64]. Empty "
            "boxes use intervals=null and are pruned before subset expansion. "
            "The echoed source, the complete 2^nonempty_box_count-1 subset "
            "expansion, exact rational growth, and worst-case ledger bytes "
            "must fit the published operation budgets; at most 16 boxes may "
            "be nonempty."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_common_space(self) -> Self:
        dimension = self.boxes[0].dimension
        if any(box.dimension != dimension for box in self.boxes):
            raise ValueError("all box-union sources must have the same dimension")

        active_box_count = sum(not box.is_empty for box in self.boxes)
        candidate_count = (1 << active_box_count) - 1
        if candidate_count > MAX_INTERSECTION_CANDIDATES:
            raise ValueError(
                f"{active_box_count} nonempty boxes require {candidate_count} "
                "intersection candidates, exceeding the complete "
                f"{MAX_INTERSECTION_CANDIDATES}-candidate bound"
            )
        # Two full ledger computations (operation plus result replay), with
        # lower/upper comparisons, plus per-entry volume validation.
        replay_work = (
            6 * dimension * active_box_count * (1 << (active_box_count - 1))
            if active_box_count
            else 0
        )
        if replay_work > MAX_BOX_UNION_REPLAY_WORK:
            raise ValueError(
                "box union exceeds the complete intersection replay work bound "
                f"({replay_work} > {MAX_BOX_UNION_REPLAY_WORK})"
            )

        endpoint_num, endpoint_den, volume_digits, union_digits = _digit_bounds(
            self.boxes,
            dimension,
            candidate_count,
        )
        result_digits = max(volume_digits, union_digits)
        if result_digits > min(
            MAX_BOX_UNION_RESULT_RATIONAL_DIGITS,
            MAX_CANONICAL_RATIONAL_DIGITS,
        ):
            raise ValueError(
                "box union can exceed the exact rational intermediate bound "
                f"({result_digits} digits > "
                f"{MAX_BOX_UNION_RESULT_RATIONAL_DIGITS})"
            )
        estimated_bytes = _maximum_result_bytes(
            self,
            candidate_count=candidate_count,
            endpoint_numerator_digits=endpoint_num,
            endpoint_denominator_digits=endpoint_den,
            volume_digits=volume_digits,
            union_digits=union_digits,
        )
        if estimated_bytes > MAX_BOX_UNION_RESULT_BYTES:
            raise ValueError(
                "box union can exceed the complete intersection-ledger result "
                f"budget ({estimated_bytes} bytes > {MAX_BOX_UNION_RESULT_BYTES})"
            )
        return self


class BoxIntersectionLedgerEntry(StrictModel):
    """One nonempty source-subset intersection and its exact volume."""

    box_indices: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_BOX_UNION_NONEMPTY_BOXES,
        description="Strictly increasing zero-based source-box indices.",
    )
    intersection: RationalAxisAlignedBox
    volume: CanonicalRational

    @model_validator(mode="after")
    def require_valid_intersection_volume(self) -> Self:
        if tuple(sorted(set(self.box_indices))) != self.box_indices:
            raise ValueError("box intersection indices must be strictly increasing")
        if self.box_indices[0] < 0:
            raise ValueError("box intersection indices must be nonnegative")
        if self.intersection.is_empty:
            raise ValueError("ledger contains only nonempty box intersections")
        if self.volume.as_fraction() != box_volume(self.intersection):
            raise ValueError("box intersection volume does not match its intervals")
        return self


class BoxUnionVolumeResult(StrictModel):
    """Exact union volume with a complete source-bound intersection ledger."""

    source: BoxUnionVolumeRequest
    intersections: tuple[BoxIntersectionLedgerEntry, ...] = Field(
        max_length=MAX_INTERSECTION_CANDIDATES,
        description=(
            "Every source subset whose closed-box intersection is nonempty, "
            "ordered first by subset size and then lexicographically."
        ),
    )
    union_volume: CanonicalRational

    @model_validator(mode="after")
    def require_complete_source_bound_ledger(self) -> Self:
        from jacobian.math.geometry.boxes._kernel import complete_intersection_ledger

        expected, expected_union = complete_intersection_ledger(self.source.boxes)
        if len(self.intersections) != len(expected):
            raise ValueError("intersection ledger is not complete for its source")
        for actual, wanted in zip(self.intersections, expected, strict=True):
            if (
                actual.box_indices != wanted.box_indices
                or actual.intersection != wanted.intersection
                or actual.volume.as_fraction() != wanted.volume
            ):
                raise ValueError("intersection ledger does not match its source boxes")
        if self.union_volume.as_fraction() != expected_union:
            raise ValueError("union volume does not match inclusion-exclusion replay")
        return self


__all__ = [
    "BoxIntersectionLedgerEntry",
    "BoxUnionVolumeRequest",
    "BoxUnionVolumeResult",
]
