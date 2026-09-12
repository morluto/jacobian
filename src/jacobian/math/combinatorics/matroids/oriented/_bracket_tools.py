"""Canonical bracket-algebra operation declarations (#2775)."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.matroids.oriented import _bracket_kernel as native
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    BracketPolynomial,
    BracketSyzygyResidualRequest,
    GrassmannPlueckerRelationRequest,
    GrassmannPlueckerRelationResult,
)


def _run_pluecker_relation(
    request: GrassmannPlueckerRelationRequest,
) -> GrassmannPlueckerRelationResult:
    return native.grassmann_pluecker_relation(
        request.ground_size, request.indices, request.family
    )


def _run_syzygy_residual(request: BracketSyzygyResidualRequest) -> BracketPolynomial:
    return native.bracket_syzygy_residual(request.target, request.terms)


BRACKET_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="oriented_matroid.grassmann_pluecker_relation.rank3.compute",
        title="Construct a rank-3 Grassmann-Pluecker relation",
        description=(
            "Return the canonical sparse formal bracket polynomial of one "
            "rank-3 Grassmann-Pluecker relation on distinct indices. "
            "Brackets are alternating symbols normalized to their increasing "
            "triple with an explicit parity, so the returned expression is a "
            "formal identity in bracket atoms. The operation emits one "
            "relation; it does not decide ideal membership or realizability."
        ),
        request_type=GrassmannPlueckerRelationRequest,
        result_type=GrassmannPlueckerRelationResult,
        run=_run_pluecker_relation,
        tags=("bracket", "pluecker", "formal", "exact"),
        examples=(
            OperationExample(
                name="four_term_six_indices",
                description="The four-term Pluecker relation on six columns.",
                input={
                    "ground_size": 6,
                    "indices": [0, 1, 2, 3, 4, 5],
                    "family": "FOUR_TERM",
                },
            ),
            OperationExample(
                name="shared_index_three_term",
                description="The three-term relation sharing index zero.",
                input={
                    "ground_size": 6,
                    "indices": [0, 1, 2, 3, 4],
                    "family": "SHARED_INDEX_THREE_TERM",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="bracket_polynomial.syzygy_residual.compute",
        title="Compute a formal bracket-polynomial syzygy residual",
        description=(
            "Return target minus a finite scalar and monomial combination of "
            "source-bound Grassmann-Pluecker relations. This is exact "
            "free-commutative algebra on canonical bracket atoms and makes no "
            "realizability claim."
        ),
        request_type=BracketSyzygyResidualRequest,
        result_type=BracketPolynomial,
        run=_run_syzygy_residual,
        tags=("bracket", "polynomial", "syzygy", "exact"),
        examples=(
            OperationExample(
                name="empty_combination",
                description=(
                    "Return a target unchanged when the finite relation combination "
                    "is empty; the target uses canonical bracket factors."
                ),
                input={
                    "target": {
                        "ground_size": 6,
                        "terms": [],
                    },
                    "terms": [],
                },
            ),
        ),
    ),
)

__all__ = ["BRACKET_OPERATIONS"]
