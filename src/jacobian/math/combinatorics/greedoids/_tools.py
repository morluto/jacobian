"""Greedoid operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.greedoids._models import (
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
from jacobian.math.combinatorics.greedoids.operations import (
    bases_profile,
    basic_word_outcome,
    convex_geometry_profile,
    rank_profile,
    recognize,
)


def _recognize(request: RecognizeRequest) -> RecognizeResult:
    return recognize(request.system)


def _rank(request: RankRequest) -> RankResult:
    return rank_profile(request.system, request.subset)


def _bases(request: BasesRequest) -> BasesResult:
    return bases_profile(request.system, request.subset)


def _basic_word_profile(request: BasicWordProfileRequest) -> BasicWordProfileResult:
    return basic_word_outcome(request.system, request.word)


def _convex_geometry(request: ConvexGeometryRequest) -> ConvexGeometryResult:
    return convex_geometry_profile(request.system)


# Minimal full-support antimatroid on two elements {a, b}:
# feasible family = {empty, {a}, {b}, {a,b}} (union-closed and accessible).
_SYSTEM = {
    "ground": ["a", "b"],
    "feasible": [[], [0], [1], [0, 1]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="greedoid.recognize.compute",
        title="Recognize a feasible-set family as a greedoid",
        description="Exhaust the accessibility and exchange axioms over the complete "
        "feasible-set family. Return GREEDOID with rank and bases, or "
        "NOT_A_GREEDOID with the first exact obstruction under deterministic "
        "order. A sample of exchange pairs cannot return GREEDOID.",
        request_type=RecognizeRequest,
        result_type=RecognizeResult,
        run=_recognize,
        tags=("greedoid", "recognition", "exact"),
        examples=(
            OperationExample(
                name="two_element_antimatroid",
                description="A two-element full-support antimatroid is a greedoid.",
                input={"system": _SYSTEM},
            ),
        ),
    ),
    MathTool(
        operation_id="greedoid.rank.compute",
        title="Compute the greedoid rank of an optional ground subset",
        description="Return r(X) = max{|F| : F feasible and F subseteq X}. If no subset "
        "is supplied, return the whole-greedoid rank (the common size of its "
        "bases).",
        request_type=RankRequest,
        result_type=RankResult,
        run=_rank,
        tags=("greedoid", "rank", "exact"),
        examples=(
            OperationExample(
                name="rank_of_full_ground",
                description="Rank of the full ground set of a two-element antimatroid.",
                input={"system": _SYSTEM},
            ),
        ),
    ),
    MathTool(
        operation_id="greedoid.bases.compute",
        title="Compute the maximal feasible subsets (bases)",
        description="Return the complete maximal feasible-set family and the common rank. "
        "For a subset-local variant, return all bases of the supplied subset.",
        request_type=BasesRequest,
        result_type=BasesResult,
        run=_bases,
        tags=("greedoid", "bases", "exact"),
        examples=(
            OperationExample(
                name="bases_of_full_ground",
                description="Bases of a two-element antimatroid.",
                input={"system": _SYSTEM},
            ),
        ),
    ),
    MathTool(
        operation_id="greedoid.basic_word.profile.compute",
        title="Profile a candidate basic word",
        description="Return BASIC_WORD if every prefix set of the distinct-element word "
        "is feasible, with final feasible-set/basis status; otherwise return "
        "NOT_A_BASIC_WORD with the first infeasible prefix. Repeated or foreign "
        "elements are boundary-invalid.",
        request_type=BasicWordProfileRequest,
        result_type=BasicWordProfileResult,
        run=_basic_word_profile,
        tags=("greedoid", "basic-word", "exact"),
        examples=(
            OperationExample(
                name="basic_word_01",
                description="Word (0, 1) is a full basic word of the two-element antimatroid.",
                input={"system": _SYSTEM, "word": [0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="greedoid.convex_geometry.compute",
        title="Compute the complementary closed-set family of a full-support antimatroid",
        description="Return the complementary closed-set family C = {E\\F : F in F}, an "
        "intersection-closed finite closure system satisfying anti-exchange, "
        "plus the feasible->closed complement map.",
        request_type=ConvexGeometryRequest,
        result_type=ConvexGeometryResult,
        run=_convex_geometry,
        tags=("greedoid", "convex-geometry", "exact"),
        examples=(
            OperationExample(
                name="two_element_convex_geometry",
                description="Complementary convex geometry of a two-element antimatroid.",
                input={"system": _SYSTEM},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
