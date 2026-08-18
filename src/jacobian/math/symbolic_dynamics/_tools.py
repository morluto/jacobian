"""Symbolic dynamics operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample

from jacobian.math.symbolic_dynamics._models import (
    AdjacencyShiftRequest,
    AdjacencyShiftResult,
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
    compute_block_language,
    compute_higher_block,
    compute_periodic_point_profile,
    construct_adjacency_shift,
    construct_finite_type_shift,
)


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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


SYMBOLIC_DYNAMICS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "symbolic_dynamics.finite_type_shift.construct",
        "Construct a shift of finite type",
        "Build a De Bruijn-style presentation of a shift of finite type "
        "from a finite forbidden-block family over an exact alphabet.",
        FiniteTypeShiftRequest,
        FiniteTypeShiftResult,
        construct_finite_type_shift,
        "symbolic-dynamics",
        "shift-of-finite-type",
        "exact",
        examples=(
            example(
                "golden_mean_shift",
                "Construct the Golden Mean shift (forbid '11'); "
                "forbidden blocks must use only declared alphabet symbols.",
                {"alphabet": ["0", "1"], "forbidden_blocks": [["1", "1"]]},
            ),
        ),
    ),
    _op(
        "symbolic_dynamics.block_language.compute",
        "Compute allowed block language",
        "Compute the complete allowed block language of a shift at a "
        "given length, excluding blocks containing any forbidden factor.",
        BlockLanguageRequest,
        BlockLanguageResult,
        compute_block_language,
        "symbolic-dynamics",
        "block-language",
        "exact",
        examples=(
            example(
                "golden_mean_blocks_2",
                "Length-2 blocks of Golden Mean shift; "
                "forbidden blocks must use only declared alphabet symbols.",
                {"alphabet": ["0", "1"], "forbidden_blocks": [["1", "1"]], "block_length": 2},
            ),
        ),
    ),
    _op(
        "symbolic_dynamics.adjacency_shift.construct",
        "Construct a shift from an adjacency matrix",
        "Construct a shift presentation from a nonnegative integer adjacency "
        "matrix, returning essential/irreducible/period/mixing profiles.",
        AdjacencyShiftRequest,
        AdjacencyShiftResult,
        construct_adjacency_shift,
        "symbolic-dynamics",
        "adjacency-matrix",
        "exact",
        examples=(
            example(
                "golden_mean_matrix",
                "Construct shift from [[1,1],[1,0]] (Golden Mean); "
                "matrix must be square with non-negative entries.",
                {"matrix": [[1, 1], [1, 0]]},
            ),
        ),
    ),
    _op(
        "symbolic_dynamics.periodic_point_profile.compute",
        "Compute periodic point profile",
        "Compute Fix(n), Exact(n), Orbit(n), and zeta function data "
        "from the adjacency matrix using Mobius inversion.",
        PeriodicPointProfileRequest,
        PeriodicPointProfileResult,
        compute_periodic_point_profile,
        "symbolic-dynamics",
        "periodic-points",
        "zeta-function",
        "exact",
        examples=(
            example(
                "golden_mean_periodic",
                "Periodic point profile of Golden Mean shift; "
                "max_period must be at least 1.",
                {"matrix": [[1, 1], [1, 0]], "max_period": 5},
            ),
        ),
    ),
    _op(
        "symbolic_dynamics.higher_block.compute",
        "Compute higher-block presentation",
        "Compute the n-th higher-block presentation of a shift, where the "
        "new alphabet consists of (n-1)-blocks from the original alphabet.",
        HigherBlockRequest,
        HigherBlockResult,
        compute_higher_block,
        "symbolic-dynamics",
        "higher-block",
        "exact",
        examples=(
            example(
                "higher_block_2",
                "2nd higher block of Golden Mean shift; n must be at least 2.",
                {"alphabet": ["0", "1"], "forbidden_blocks": [["1", "1"]], "n": 2},
            ),
        ),
    ),
)


TOOLS = SYMBOLIC_DYNAMICS_OPERATIONS

__all__ = ["TOOLS"]
