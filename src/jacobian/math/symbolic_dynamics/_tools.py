"""Public symbolic dynamics operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.symbolic_dynamics._models import (
    ArtinMazurZetaRequest,
    ArtinMazurZetaResult,
    BlockLanguageRequest,
    BlockLanguageResult,
    FiniteTypeShiftRequest,
    FiniteTypeShiftResult,
    HigherBlockRequest,
    HigherBlockResult,
    PeriodicPointProfileRequest,
    PeriodicPointProfileResult,
)
from jacobian.math.symbolic_dynamics._operations import (
    compute_artin_mazur_zeta,
    compute_block_language,
    compute_higher_block,
    compute_periodic_point_profile,
    construct_finite_type_shift,
)

_GOLDEN_MEAN = {
    "alphabet": ["0", "1"],
    "forbidden_blocks": [["1", "1"]],
    "two_sided": True,
}

SYMBOLIC_DYNAMICS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="symbolic_dynamics.finite_type_shift.construct",
        version="1",
        title="Construct a finite-type shift presentation",
        description=(
            "Construct the complete finite labeled De Bruijn presentation induced "
            "by a bounded forbidden-block specification."
        ),
        request_type=FiniteTypeShiftRequest,
        result_type=FiniteTypeShiftResult,
        run=construct_finite_type_shift,
        tags=("symbolic-dynamics", "shift-of-finite-type", "presentation", "exact"),
        examples=(
            example(
                "golden_mean_presentation",
                "Construct the presentation forbidding the block 11.",
                {"shift": _GOLDEN_MEAN},
            ),
        ),
    ),
    MathTool(
        operation_id="symbolic_dynamics.block_language.compute",
        version="1",
        title="Compute a bounded block language",
        description=(
            "Enumerate every block of one requested length that occurs in an "
            "infinite shift point. Oversized support or result enumerations are "
            "rejected before computation, never truncated."
        ),
        request_type=BlockLanguageRequest,
        result_type=BlockLanguageResult,
        run=compute_block_language,
        tags=("symbolic-dynamics", "block-language", "exact", "complete"),
        examples=(
            example(
                "golden_mean_blocks_3",
                "Enumerate every occurring length-three block.",
                {"shift": _GOLDEN_MEAN, "block_length": 3},
            ),
        ),
    ),
    MathTool(
        operation_id="symbolic_dynamics.periodic_point_profile.compute",
        version="1",
        title="Compute a periodic-point profile",
        description=(
            "Compute fixed-point, least-period-point, and primitive-orbit counts "
            "through a declared period by exact traces and Möbius inversion."
        ),
        request_type=PeriodicPointProfileRequest,
        result_type=PeriodicPointProfileResult,
        run=compute_periodic_point_profile,
        tags=("symbolic-dynamics", "periodic-points", "mobius-inversion", "exact"),
        examples=(
            example(
                "golden_mean_periodic_points",
                "Compute the profile through period five.",
                {
                    "shift": {"matrix": [[1, 1], [1, 0]], "two_sided": True},
                    "max_period": 5,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="symbolic_dynamics.artin_mazur_zeta.compute",
        version="1",
        title="Compute an Artin-Mazur zeta function",
        description=(
            "Compute the exact edge-shift zeta function 1/det(I-tA) as a "
            "canonical rational function, retaining det(I-tA) with constant "
            "term one and replaying -tD'(t)/D(t) against periodic-point traces."
        ),
        request_type=ArtinMazurZetaRequest,
        result_type=ArtinMazurZetaResult,
        run=compute_artin_mazur_zeta,
        tags=("symbolic-dynamics", "artin-mazur-zeta", "periodic-points", "exact"),
        examples=(
            example(
                "golden_mean_zeta",
                "Compute the Golden Mean shift zeta function through five replay coefficients.",
                {
                    "shift": {"matrix": [[1, 1], [1, 0]], "two_sided": True},
                    "replay_period": 5,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="symbolic_dynamics.higher_block.compute",
        version="1",
        title="Construct a higher-block presentation",
        description=(
            "Construct the exact overlap presentation on all occurring blocks of "
            "a declared length at least the forbidden-block presentation memory."
        ),
        request_type=HigherBlockRequest,
        result_type=HigherBlockResult,
        run=compute_higher_block,
        tags=("symbolic-dynamics", "higher-block", "presentation", "exact"),
        examples=(
            example(
                "golden_mean_two_block",
                "Construct the two-block presentation of the Golden Mean shift.",
                {"shift": _GOLDEN_MEAN, "block_length": 2},
            ),
        ),
    ),
)

TOOLS = SYMBOLIC_DYNAMICS_OPERATIONS

__all__ = ["TOOLS"]
