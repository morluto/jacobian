"""Finite geometric discrepancy operation declaration."""

from typing import Any

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.discrepancy.operations import squared_l2_star_discrepancy
from jacobian.math.matrices.values import RationalMatrix


class SquaredL2StarDiscrepancyRequest(StrictModel):
    points: RationalMatrix = Field(
        description="Ordered point occurrences as rows and unit-cube coordinates as columns. Repeated points count repeatedly. column_count retains the positive dimension for an empty list."
    )


def _run(request: SquaredL2StarDiscrepancyRequest) -> CanonicalRational:
    return squared_l2_star_discrepancy(request.points)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="geometry.discrepancy.squared_l2_star.compute",
        title="Compute exact unnormalized squared L2 star discrepancy",
        description="Integrate (sum_i 1[x_i in [0,u)) - n*product(u))^2 over the unit cube with Lebesgue measure. Return an exact rational scalar, retaining point multiplicities and coordinate order in the input matrix. Empty point lists have value zero. This is unnormalized squared L2 discrepancy, not supremum discrepancy. Admit (n*n+n)*d <= 1000000 and at most 50000000 height-weighted arithmetic units.",
        request_type=SquaredL2StarDiscrepancyRequest,
        result_type=CanonicalRational,
        run=_run,
        tags=("geometry", "discrepancy", "l2", "star", "rational", "unit-cube"),
        examples=(
            OperationExample(
                name="two_diagonal_points",
                description="Compute squared L2 star discrepancy for two point occurrences; each matrix row is a point in the unit square.",
                input={
                    "points": {
                        "row_count": 2,
                        "column_count": 2,
                        "entries": [
                            [{"num": "1", "den": "4"}] * 2,
                            [{"num": "3", "den": "4"}] * 2,
                        ],
                    }
                },
            ),
        ),
    ),
)
