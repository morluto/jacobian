"""Catalog declaration for finite abelian subset-sum profiles."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.additive.finite_abelian_subset_sum._models import (
    FiniteAbelianSubsetSumRequest,
    FiniteAbelianSubsetSumResult,
)
from jacobian.math.combinatorics.additive.finite_abelian_subset_sum.operations import (
    finite_abelian_subset_sum_profile,
)


def _compute(request: FiniteAbelianSubsetSumRequest) -> FiniteAbelianSubsetSumResult:
    return finite_abelian_subset_sum_profile(request.group, request.sequence)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="additive.subset_sum.finite_abelian_profile.compute",
        title="Compute a finite abelian indexed subset-sum profile",
        description=(
            "Return multiplicities for every element of a finite abelian product "
            "group, retaining repeated indexed source positions and the empty subset."
        ),
        request_type=FiniteAbelianSubsetSumRequest,
        result_type=FiniteAbelianSubsetSumResult,
        run=_compute,
        tags=("additive-combinatorics", "subset-sum", "finite-abelian", "exact"),
        examples=(
            OperationExample(
                name="klein_four_profile",
                description="The sequence (1,0),(0,1) covers C2 x C2 once.",
                input={
                    "group": {"moduli": ["2", "2"]},
                    "sequence": [
                        {"group": {"moduli": ["2", "2"]}, "coordinates": [1, 0]},
                        {"group": {"moduli": ["2", "2"]}, "coordinates": [0, 1]},
                    ],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
