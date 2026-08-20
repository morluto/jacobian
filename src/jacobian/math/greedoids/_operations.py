"""Domain adapter for greedoid operations."""

from __future__ import annotations

from typing import Any

from jacobian.math.greedoids._models import (
    BasesRequest,
    BasesResult,
    BasicWordProfileRequest,
    BasicWordProfileResult,
    ConvexGeometryRequest,
    ConvexGeometryResult,
    RankRequest,
    RankResult,
    RecognizeRequest,
    RecognizeResult,
)
from jacobian.math.greedoids.operations import (
    antimatroid_to_convex_geometry,
    bases,
    basic_word_profile,
    rank,
    recognize,
)

__all__ = [
    "compute_bases",
    "compute_basic_word_profile",
    "compute_convex_geometry",
    "compute_rank",
    "compute_recognize",
]


def compute_recognize(request: RecognizeRequest) -> RecognizeResult:
    result: dict[str, Any] = recognize(request.system)
    if result["status"] == "GREEDOID":
        return RecognizeResult(
            status="GREEDOID",
            rank=result["rank"],
            bases=tuple(result["bases"]),
            ground_size=result["ground_size"],
        )
    return RecognizeResult(
        status="NOT_A_GREEDOID",
        obstruction=result["obstruction"],
        larger_set=result.get("larger_set"),
        smaller_set=result.get("smaller_set"),
        feasible_set=result.get("feasible_set"),
    )


def compute_rank(request: RankRequest) -> RankResult:
    if request.subset is None:
        r = rank(request.system)
    else:
        r = rank(request.system, frozenset(request.subset))
    return RankResult(rank=r, subset=request.subset)


def compute_bases(request: BasesRequest) -> BasesResult:
    if request.subset is None:
        r, basis_list = bases(request.system)
    else:
        r, basis_list = bases(request.system, frozenset(request.subset))
    return BasesResult(
        rank=r,
        bases=tuple(tuple(sorted(b)) for b in basis_list),
    )


def compute_basic_word_profile(
    request: BasicWordProfileRequest,
) -> BasicWordProfileResult:
    result: dict[str, Any] = basic_word_profile(request.system, request.word)
    if result["status"] == "BASIC_WORD":
        return BasicWordProfileResult(
            status="BASIC_WORD",
            prefix_length=result["prefix_length"],
            is_full=result["is_full"],
            rank=result["rank"],
        )
    return BasicWordProfileResult(
        status="NOT_A_BASIC_WORD",
        obstruction=result["obstruction"],
        prefix_index=result.get("prefix_index"),
        prefix_set=result.get("prefix_set"),
    )


def compute_convex_geometry(
    request: ConvexGeometryRequest,
) -> ConvexGeometryResult:
    closed_family, complement_map = antimatroid_to_convex_geometry(request.system)
    return ConvexGeometryResult(
        closed_family=tuple(closed_family),
        complement_map=tuple(sorted(complement_map.items())),
    )
