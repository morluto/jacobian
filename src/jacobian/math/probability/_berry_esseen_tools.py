"""Berry--Esseen operation declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability._berry_esseen import (
    BerryEsseenRequest,
    BerryEsseenResult,
    berry_esseen_bound,
)

_FAIR_BIT = {
    "atoms": [
        {"value": {"num": "0", "den": "1"}, "probability": {"num": "1", "den": "2"}},
        {"value": {"num": "1", "den": "1"}, "probability": {"num": "1", "den": "2"}},
    ]
}

BERRY_ESSEEN_OPERATION = MathTool(
    operation_id="probability.finite_distribution.berry_esseen_05600.compute",
    title="Exact independent Berry--Esseen bound (C=0.5600)",
    description=(
        "For independent, finite rational summands with positive variances, "
        "return exact total mean, variance, and third absolute central moment "
        "and the outward-rounded bound C*rho/V^(3/2) using the published "
        "general independent non-identical constant C=0.5600=14/25."
    ),
    request_type=BerryEsseenRequest,
    result_type=BerryEsseenResult,
    run=berry_esseen_bound,
    tags=("probability", "berry-esseen", "normal-approximation", "exact", "bounded"),
    discovery_terms=("Berry Esseen", "independent summands", "normal approximation"),
    examples=(
        OperationExample(
            name="two_fair_bits",
            description="Bound the normalized sum of two independent fair Bernoulli summands.",
            input={"summands": [_FAIR_BIT, _FAIR_BIT]},
        ),
    ),
)

TOOLS = (BERRY_ESSEEN_OPERATION,)

__all__ = ["BERRY_ESSEEN_OPERATION", "TOOLS"]
