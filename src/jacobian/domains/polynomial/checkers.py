"""Independent checker declarations owned by the polynomial domain."""

from collections.abc import Callable

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.jacobian_syzygy import GradedJacobianSyzygyRequest
from jacobian.contracts.polynomial_operations import (
    PolynomialDiscriminantRequest,
    PolynomialFactorRequest,
    PolynomialGcdRequest,
    PolynomialResultantRequest,
    PolynomialSquareFreeRequest,
)
from jacobian.providers import flint_runtime


def _flint_exact_replay_runtime(
    *, checker_ids: tuple[str, ...] = (), refresh: bool = False
) -> CapabilityProviderRuntime:
    return flint_runtime.exact_domain_checker_provider_runtime(
        checker_ids=checker_ids,
        refresh=refresh,
    )


def _graded_syzygy_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return flint_runtime.graded_syzygy_checker_provider_runtime(checker_ids=checker_ids)


def _univariate_polynomial(*fields: str) -> Callable[[object], bool]:
    def supports(payload: object) -> bool:
        return isinstance(payload, dict) and all(
            isinstance(payload.get(field), dict)
            and payload[field].get("variables")
            and len(payload[field]["variables"]) == 1
            for field in fields
        )

    return supports


POLYNOMIAL_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "polynomial.jacobian_syzygy.minimum_degree.compute",
        GradedJacobianSyzygyRequest,
        "check_graded_jacobian_syzygy",
        "polynomial.jacobian-syzygy.graded-fraction-replay",
        entrypoint_module="jacobian_checkers.jacobian_syzygy",
        replay_method="standard-library exact rational graded-map replay",
        reason=(
            "operator-authorized exact rational checker independently reconstructs "
            "the homogeneous coefficient maps without importing the SymPy producer"
        ),
        verification_capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
        verification_title="Verify a first graded Jacobian syzygy degree",
        verification_description=(
            "Independently reconstruct every bounded homogeneous coefficient map, "
            "rank ledger, nonzero minor, and first kernel from the exact producer "
            "input and the complete, unmodified producer output.result object."
        ),
        verification_tags=(
            "verification",
            "exact",
            "polynomial",
            "jacobian",
            "syzygy",
        ),
        provider_runtime_factory=_graded_syzygy_runtime,
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.gcd",
        PolynomialGcdRequest,
        "check_polynomial_gcd",
        "polynomial.gcd.flint-replay",
        provider_runtime_factory=_flint_exact_replay_runtime,
        supports_input=_univariate_polynomial("left", "right"),
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.resultant",
        PolynomialResultantRequest,
        "check_polynomial_resultant",
        "polynomial.resultant.flint-replay",
        provider_runtime_factory=_flint_exact_replay_runtime,
        supports_input=_univariate_polynomial("left", "right"),
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.discriminant",
        PolynomialDiscriminantRequest,
        "check_polynomial_discriminant",
        "polynomial.discriminant.flint-replay",
        provider_runtime_factory=_flint_exact_replay_runtime,
        supports_input=_univariate_polynomial("polynomial"),
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.compute.square_free_decomposition",
        PolynomialSquareFreeRequest,
        "check_polynomial_square_free",
        "polynomial.square-free.flint-replay",
        provider_runtime_factory=_flint_exact_replay_runtime,
        supports_input=_univariate_polynomial("polynomial"),
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.factor.compute",
        PolynomialFactorRequest,
        "check_polynomial_factorization",
        "polynomial.factorization.flint-replay",
        provider_runtime_factory=_flint_exact_replay_runtime,
        supports_input=_univariate_polynomial("polynomial"),
    ),
)

__all__ = ["POLYNOMIAL_EXACT_REPLAY_CHECKERS"]
