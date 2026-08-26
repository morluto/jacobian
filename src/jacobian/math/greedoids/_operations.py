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
    _antimatroid_to_convex_geometry_unchecked,
    _bases_unchecked,
    _basic_word_profile_unchecked,
    _rank_unchecked,
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
    recognized = recognize(request.system)
    if recognized["status"] != "GREEDOID":
        return RankResult(
            status="NOT_A_GREEDOID",
            obstruction=str(recognized["obstruction"]),
            subset=request.subset,
        )
    if request.subset is None:
        r = _rank_unchecked(request.system, None)
    else:
        r = _rank_unchecked(request.system, frozenset(request.subset))
    return RankResult(rank=r, subset=request.subset)


def compute_bases(request: BasesRequest) -> BasesResult:
    recognized = recognize(request.system)
    if recognized["status"] != "GREEDOID":
        return BasesResult(
            status="NOT_A_GREEDOID",
            bases=(),
            obstruction=str(recognized["obstruction"]),
        )
    if request.subset is None:
        r, basis_list = _bases_unchecked(request.system, None)
    else:
        r, basis_list = _bases_unchecked(request.system, frozenset(request.subset))
    return BasesResult(
        rank=r,
        bases=tuple(tuple(sorted(b)) for b in basis_list),
    )


def compute_basic_word_profile(
    request: BasicWordProfileRequest,
) -> BasicWordProfileResult:
    recognized = recognize(request.system)
    if recognized["status"] != "GREEDOID":
        return BasicWordProfileResult(
            status="NOT_A_BASIC_WORD",
            obstruction="not_a_greedoid",
        )
    result: dict[str, Any] = _basic_word_profile_unchecked(request.system, request.word)
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
    recognized = recognize(request.system)
    if recognized["status"] != "GREEDOID":
        return ConvexGeometryResult(
            status="NOT_AN_ANTIMATROID",
            obstruction=str(recognized["obstruction"]),
        )
    from jacobian.math.greedoids.operations import union_closed

    full_ground = tuple(range(len(request.system.ground)))
    if full_ground not in request.system.feasible_index() or not union_closed(
        request.system
    ):
        return ConvexGeometryResult(
            status="NOT_AN_ANTIMATROID", obstruction="not_full_support_or_union_closed"
        )
    closed_family, complement_map = _antimatroid_to_convex_geometry_unchecked(
        request.system
    )
    return ConvexGeometryResult(
        closed_family=tuple(closed_family),
        complement_map=tuple(sorted(complement_map.items())),
    )
