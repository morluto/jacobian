"""Greedoid operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# Minimal full-support antimatroid on two elements {a, b}:
# feasible family = {empty, {a}, {b}, {a,b}} (union-closed and accessible).
_SYSTEM = {
    "ground": ["a", "b"],
    "feasible": [[], [0], [1], [0, 1]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "greedoid.recognize.compute",
        "Recognize a feasible-set family as a greedoid",
        "Exhaust the accessibility and exchange axioms over the complete "
        "feasible-set family. Return GREEDOID with rank and bases, or "
        "NOT_A_GREEDOID with the first exact obstruction under deterministic "
        "order. A sample of exchange pairs cannot return GREEDOID.",
        RecognizeRequest,
        RecognizeResult,
        _recognize,
        "greedoid",
        "recognition",
        "exact",
        examples=(
            example(
                "two_element_antimatroid",
                "A two-element full-support antimatroid is a greedoid.",
                {"system": _SYSTEM},
            ),
        ),
    ),
    _op(
        "greedoid.rank.compute",
        "Compute the greedoid rank of an optional ground subset",
        "Return r(X) = max{|F| : F feasible and F subseteq X}. If no subset "
        "is supplied, return the whole-greedoid rank (the common size of its "
        "bases).",
        RankRequest,
        RankResult,
        _rank,
        "greedoid",
        "rank",
        "exact",
        examples=(
            example(
                "rank_of_full_ground",
                "Rank of the full ground set of a two-element antimatroid.",
                {"system": _SYSTEM},
            ),
        ),
    ),
    _op(
        "greedoid.bases.compute",
        "Compute the maximal feasible subsets (bases)",
        "Return the complete maximal feasible-set family and the common rank. "
        "For a subset-local variant, return all bases of the supplied subset.",
        BasesRequest,
        BasesResult,
        _bases,
        "greedoid",
        "bases",
        "exact",
        examples=(
            example(
                "bases_of_full_ground",
                "Bases of a two-element antimatroid.",
                {"system": _SYSTEM},
            ),
        ),
    ),
    _op(
        "greedoid.basic_word.profile.compute",
        "Profile a candidate basic word",
        "Return BASIC_WORD if every prefix set of the distinct-element word "
        "is feasible, with final feasible-set/basis status; otherwise return "
        "NOT_A_BASIC_WORD with the first infeasible prefix. Repeated or foreign "
        "elements are boundary-invalid.",
        BasicWordProfileRequest,
        BasicWordProfileResult,
        _basic_word_profile,
        "greedoid",
        "basic-word",
        "exact",
        examples=(
            example(
                "basic_word_01",
                "Word (0, 1) is a full basic word of the two-element antimatroid.",
                {"system": _SYSTEM, "word": [0, 1]},
            ),
        ),
    ),
    _op(
        "greedoid.convex_geometry.compute",
        "Compute the complementary closed-set family of a full-support antimatroid",
        "Return the complementary closed-set family C = {E\\F : F in F}, an "
        "intersection-closed finite closure system satisfying anti-exchange, "
        "plus the feasible->closed complement map.",
        ConvexGeometryRequest,
        ConvexGeometryResult,
        _convex_geometry,
        "greedoid",
        "convex-geometry",
        "exact",
        examples=(
            example(
                "two_element_convex_geometry",
                "Complementary convex geometry of a two-element antimatroid.",
                {"system": _SYSTEM},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
