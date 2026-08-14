"""Typed real-quadratic order operation and checker declaration."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.operations import (
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.domains._examples import example
from jacobian.domains.arithmetic._support import arithmetic_operation
from jacobian.math.real_quadratic import (
    RealQuadraticOrderRequest,
    RealQuadraticOrderValue,
    real_quadratic_order,
)
from jacobian.provider_runtime import source_provider_runtime


def _real_quadratic_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> ProviderObservation:
    return source_provider_runtime(
        "jacobian.real-quadratic-checker",
        version="1",
        entrypoint="jacobian_checkers.real_quadratic:check_real_quadratic_order",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


REAL_QUADRATIC_OPERATIONS = (
    arithmetic_operation(
        "arithmetic.real_quadratic.order.compute",
        "Compare exact real quadratic values",
        (
            "Compare two bounded values a+b*sqrt(d) in one shared real quadratic "
            "field, returning their exact difference and squared-magnitude sign data."
        ),
        RealQuadraticOrderRequest,
        RealQuadraticOrderValue,
        real_quadratic_order,
        "arithmetic",
        "real-quadratic",
        "quadratic-surd",
        "exact-order",
        examples=(
            example(
                "pang_m4_scalar_gap",
                "Compare 3*sqrt(3)/8 with 1/2+sqrt(3)/20 exactly.",
                {
                    "left": {
                        "rational_part": {"num": "0", "den": "1"},
                        "radical_coefficient": {"num": "3", "den": "8"},
                        "radicand": 3,
                    },
                    "right": {
                        "rational_part": {"num": "1", "den": "2"},
                        "radical_coefficient": {"num": "1", "den": "20"},
                        "radicand": 3,
                    },
                },
            ),
        ),
    ),
)

REAL_QUADRATIC_CHECKERS = (
    AuthorizedChecker(
        "arithmetic.real_quadratic.order.compute",
        RealQuadraticOrderRequest,
        "check_real_quadratic_order",
        "arithmetic.real-quadratic.fraction-square-replay",
        entrypoint_module="jacobian_checkers.real_quadratic",
        observation_loader=_real_quadratic_runtime,
        replay_method="standard-library Fraction squared-magnitude replay",
        reason=(
            "operator-authorized standard-library checker independently compares "
            "the exact rational and radical squared magnitudes"
        ),
        verification_operation_id="arithmetic.real_quadratic.order.verify",
        verification_title="Verify an exact real-quadratic order",
        verification_description=(
            "Independently replay the shared-field difference, sign case, squared "
            "magnitudes, and resulting order using exact rational arithmetic."
        ),
        verification_tags=(
            "verification",
            "exact",
            "arithmetic",
            "real-quadratic",
            "order",
        ),
    ),
)

__all__ = ["REAL_QUADRATIC_CHECKERS", "REAL_QUADRATIC_OPERATIONS"]
