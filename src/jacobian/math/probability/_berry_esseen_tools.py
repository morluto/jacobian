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
    operation_id="probability.finite_distribution.berry_esseen_iid_05600.compute",
    title="Exact i.i.d. Berry--Esseen bound (general-independent C=0.5600)",
    description=(
        "For a finite rational law with positive variance and a positive i.i.d. "
        "sample count n, return exact source mean, variance, and third absolute "
        "central moment and the outward-rounded bound C*rho/(sigma^3*sqrt(n)) "
        "using Shevtsova's published general-independent constant C=0.5600=14/25 "
        "as a valid specialization to the i.i.d. request. Admission envelope: "
        "at most 16,384 atoms, 128 decimal digits per input rational component, "
        "and 512 decimal digits per intermediate or result rational component. "
        "The sample-count field uses a 512-digit exact-integer envelope; a "
        "particular n is admitted from the reduced height of C^2*rho^2/(sigma^6*n) "
        "after cancellation, including affine rescalings of a small law."
    ),
    request_type=BerryEsseenRequest,
    result_type=BerryEsseenResult,
    run=berry_esseen_bound,
    tags=("probability", "berry-esseen", "normal-approximation", "exact", "bounded"),
    discovery_terms=("Berry Esseen", "iid summands", "normal approximation"),
    examples=(
        OperationExample(
            name="four_fair_bits",
            description="Bound the normalized sum of four i.i.d. fair Bernoulli variables.",
            input={"distribution": _FAIR_BIT, "sample_count": "4"},
        ),
    ),
)

TOOLS = (BERRY_ESSEEN_OPERATION,)

__all__ = ["BERRY_ESSEEN_OPERATION", "TOOLS"]
