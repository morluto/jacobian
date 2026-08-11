"""Independent checker declarations owned by the arithmetic domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.rationals import RationalPairRequest

_ENTRYPOINT = "jacobian_checkers.exact_arithmetic"
_REASON = (
    "operator-authorized Python-FLINT rational replay independent of the "
    "standard-library Fraction producer"
)

ARITHMETIC_EXACT_REPLAY_CHECKERS = tuple(
    ExactReplayCheckerDeclaration(
        capability_id,
        RationalPairRequest,
        function,
        format_id,
        entrypoint_module=_ENTRYPOINT,
        replay_method="Python-FLINT exact rational replay",
        reason=_REASON,
    )
    for capability_id, function, format_id in (
        (
            "rational.compute.sum",
            "check_rational_sum",
            "rational.sum.flint-replay",
        ),
        (
            "rational.compute.difference",
            "check_rational_difference",
            "rational.difference.flint-replay",
        ),
        (
            "rational.compute.product",
            "check_rational_product",
            "rational.product.flint-replay",
        ),
        (
            "rational.compute.quotient",
            "check_rational_quotient",
            "rational.quotient.flint-replay",
        ),
    )
)

__all__ = ["ARITHMETIC_EXACT_REPLAY_CHECKERS"]
