"""Typed contracts for exact finite unions of rational boxes."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.boxes.values import RationalAxisAlignedBox


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


MAX_BOX_UNION_NONEMPTY_BOXES = 16
MAX_INTERSECTION_CANDIDATES = (1 << MAX_BOX_UNION_NONEMPTY_BOXES) - 1
MAX_BOX_UNION_RESULT_BYTES = 8 * 1024 * 1024
MAX_BOX_UNION_RESULT_RATIONAL_DIGITS = 16_384


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
            raise _validation_error(
                "box_intersection_indices_strictly_increasing",
                "box intersection indices must be strictly increasing",
            )
        if self.box_indices[0] < 0:
            raise _validation_error(
                "box_intersection_indices_nonnegative",
                "box intersection indices must be nonnegative",
            )
        if self.intersection.is_empty:
            raise _validation_error(
                "ledger_contains_nonempty_box_intersections",
                "ledger contains only nonempty box intersections",
            )
        return self


class BoxUnionVolumeResult(StrictModel):
    """Exact union volume with a complete source-bound intersection ledger."""

    source: tuple[RationalAxisAlignedBox, ...] = Field(min_length=1)
    intersections: tuple[BoxIntersectionLedgerEntry, ...] = Field(
        max_length=MAX_INTERSECTION_CANDIDATES,
        description=(
            "Every source subset whose closed-box intersection is nonempty, "
            "ordered first by subset size and then lexicographically."
        ),
    )
    union_volume: CanonicalRational

    @model_validator(mode="after")
    def require_structural_ledger(self) -> Self:
        if any(
            index >= len(self.source)
            for entry in self.intersections
            for index in entry.box_indices
        ):
            raise _validation_error(
                "intersection_ledger_source_indices",
                "intersection ledger indices must refer to retained source boxes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: tuple[RationalAxisAlignedBox, ...],
        intersections: tuple[BoxIntersectionLedgerEntry, ...],
        union_volume: CanonicalRational,
    ) -> Self:
        """Construct a result after the owner kernel completed the ledger."""

        return cls.model_construct(
            source=source,
            intersections=intersections,
            union_volume=union_volume,
        )


__all__ = [
    "BoxIntersectionLedgerEntry",
    "BoxUnionVolumeRequest",
    "BoxUnionVolumeResult",
]
