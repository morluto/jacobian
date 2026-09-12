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
    description=(
        "Return the exact compound-Poisson cumulant prefix κ_n=λE[J^n] "
        "for a nonnegative rational intensity and normalized finite rational "
        "jump law, without materializing the generally infinite distribution."
    ),
    request_type=CompoundPoissonCumulantRequest,
    result_type=CompoundPoissonCumulantResult,
    run=compound_poisson_cumulant_prefix,
    tags=("probability", "compound-poisson", "cumulant", "moment", "exact"),
    examples=(
        OperationExample(
            name="bernoulli_jumps",
            description=(
                "Compute the first three cumulants for rate two and fair binary "
                "jumps; the intensity must be nonnegative and the jump masses "
                "must form a normalized canonical finite law."
            ),
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
