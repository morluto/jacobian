"""Compound-Poisson cumulant declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability._compound_poisson import (
    CompoundPoissonCumulantRequest,
    CompoundPoissonCumulantResult,
    compound_poisson_cumulant_prefix,
)

COMPOUND_POISSON_CUMULANT_OPERATION = MathTool(
    operation_id="probability.compound_poisson.cumulant_prefix.compute",
    title="Compute an exact compound-Poisson cumulant prefix",
    description="For nonnegative rational intensity and a normalized finite rational jump law, return κ_n=λE[J^n] without materializing the compound distribution.",
    request_type=CompoundPoissonCumulantRequest,
    result_type=CompoundPoissonCumulantResult,
    run=compound_poisson_cumulant_prefix,
    tags=("probability", "compound-poisson", "cumulant", "moment", "exact"),
    examples=(
        OperationExample(
            name="bernoulli_jumps",
            description="Compute the first three cumulants for rate two and fair binary jumps.",
            input={
                "intensity": {"num": "2", "den": "1"},
                "jump_distribution": {
                    "atoms": [
                        {
                            "value": {"num": "0", "den": "1"},
                            "probability": {"num": "1", "den": "2"},
                        },
                        {
                            "value": {"num": "1", "den": "1"},
                            "probability": {"num": "1", "den": "2"},
                        },
                    ]
                },
                "max_order": 3,
            },
        ),
    ),
)

__all__ = ["COMPOUND_POISSON_CUMULANT_OPERATION"]
