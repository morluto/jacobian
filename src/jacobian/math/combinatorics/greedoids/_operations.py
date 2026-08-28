"""Domain adapter for greedoid operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.greedoids._models import (
    BasesRequest,
    BasesResult,
    BasicWordProfileRequest,
    BasicWordProfileResult,
    ConvexGeometryRequest,
    ConvexGeometryResult,
    GreedoidAdmissionError,
    RankRequest,
    RankResult,
    RecognizeRequest,
    RecognizeResult,
    require_bounded_carrier,
)
from jacobian.math.combinatorics.greedoids.operations import (
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


def _reject(reason: str, message: str, *location: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"greedoid.{reason}",
        message=message,
    )


def _admit_system(
    request: RecognizeRequest | BasicWordProfileRequest | ConvexGeometryRequest,
) -> None:
    try:
        require_bounded_carrier(request.system)
    except GreedoidAdmissionError as exc:
        _reject(exc.reason, str(exc), "system")


def _admit_subset(request: RankRequest | BasesRequest) -> None:
    try:
        require_bounded_carrier(request.system)
    except GreedoidAdmissionError as exc:
        _reject(exc.reason, str(exc), "system")
    if request.subset is None:
        return
    if len(set(request.subset)) != len(request.subset):
        _reject("subset_duplicate", "subset must not contain duplicates", "subset")
    if any(not 0 <= index < len(request.system.ground) for index in request.subset):
        _reject(
            "subset_index_out_of_range", "subset indices must be in range", "subset"
        )


def compute_recognize(request: RecognizeRequest) -> RecognizeResult:
    _admit_system(request)
    return recognize(request.system)


def compute_rank(request: RankRequest) -> RankResult:
    _admit_subset(request)
    recognized = recognize(request.system)
    if recognized.status != "GREEDOID":
        return RankResult(
            status="NOT_A_GREEDOID",
            obstruction=recognized.obstruction,
            subset=request.subset,
        )
    if request.subset is None:
        r = _rank_unchecked(request.system, None)
    else:
        r = _rank_unchecked(request.system, frozenset(request.subset))
    return RankResult(rank=r, subset=request.subset)


def compute_bases(request: BasesRequest) -> BasesResult:
    _admit_subset(request)
    recognized = recognize(request.system)
    if recognized.status != "GREEDOID":
        return BasesResult(
            status="NOT_A_GREEDOID",
            bases=(),
            obstruction=recognized.obstruction,
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
    _admit_system(request)
    recognized = recognize(request.system)
    if recognized.status != "GREEDOID":
        return BasicWordProfileResult(
            status="NOT_A_BASIC_WORD",
            obstruction="not_a_greedoid",
        )
    return _basic_word_profile_unchecked(request.system, request.word)


def compute_convex_geometry(
    request: ConvexGeometryRequest,
) -> ConvexGeometryResult:
    _admit_system(request)
    recognized = recognize(request.system)
    if recognized.status != "GREEDOID":
        return ConvexGeometryResult(
            status="NOT_AN_ANTIMATROID",
            obstruction=recognized.obstruction,
        )
    from jacobian.math.combinatorics.greedoids.operations import union_closed

    full_ground = tuple(range(len(request.system.ground)))
    if full_ground not in request.system.feasible_index() or not union_closed(
        request.system
    ):
        return ConvexGeometryResult(
            status="NOT_AN_ANTIMATROID", obstruction="not_full_support_or_union_closed"
        )
    return _antimatroid_to_convex_geometry_unchecked(request.system)
