"""Monic quartic cubic-resolvent declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._quartic_resolvent import (
    QuarticCubicResolventRequest,
    QuarticCubicResolventResult,
    compute_quartic_cubic_resolvent,
)


def _run(
    request: QuarticCubicResolventRequest,
) -> QuarticCubicResolventResult:
    return compute_quartic_cubic_resolvent(request.polynomial)


QUARTIC_CUBIC_RESOLVENT_OPERATION = MathTool(
    operation_id="polynomial.quartic.cubic_resolvent.compute",
    title="Compute the cubic resolvent of a monic quartic",
    description=(
        "For f(T)=T^4+aT^3+bT^2+cT+d over QQ, return the cubic whose roots "
        "are alpha1*alpha2+alpha3*alpha4, alpha1*alpha3+alpha2*alpha4, and "
        "alpha1*alpha4+alpha2*alpha3 for roots alpha1,...,alpha4 of f. Its "
        "coefficient convention is Y^3-bY^2+(ac-4d)Y+(4bd-a^2d-c^2). "
        "The result retains the exact source quartic."
    ),
    request_type=QuarticCubicResolventRequest,
    result_type=QuarticCubicResolventResult,
    run=_run,
    tags=("polynomial", "quartic", "cubic-resolvent", "exact", "QQ"),
    discovery_terms=("quartic cubic resolvent", "Galois resolvent cubic"),
    examples=(
        OperationExample(
            name="roots-1-through-4",
            description=(
                "The quartic with roots 1, 2, 3, and 4 has pair-product sums "
                "14, 11, and 10, giving Y^3-35Y^2+404Y-1540."
            ),
            input={
                "polynomial": {
                    "variables": ["t"],
                    "polynomial": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponents": [4]},
                            {
                                "coefficient": {"num": "-10", "den": "1"},
                                "exponents": [3],
                            },
                            {
                                "coefficient": {"num": "35", "den": "1"},
                                "exponents": [2],
                            },
                            {
                                "coefficient": {"num": "-50", "den": "1"},
                                "exponents": [1],
                            },
                            {
                                "coefficient": {"num": "24", "den": "1"},
                                "exponents": [0],
                            },
                        ]
                    },
                }
            },
        ),
    ),
)

__all__ = ["QUARTIC_CUBIC_RESOLVENT_OPERATION"]
