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


def _materialized_syzygy_supports(payload: object) -> bool:
    """Bound aggregate checker work while retaining cheap degree-zero cases."""

    if not isinstance(payload, dict) or type(payload.get("max_degree")) is not int:
        return False
    maximum_degree = payload["max_degree"]
    polynomial = payload.get("polynomial")
    factors = payload.get("linear_factors")
    if isinstance(polynomial, dict):
        body = polynomial.get("polynomial")
        terms = body.get("terms") if isinstance(body, dict) else None
        if not isinstance(terms, list):
            return False
        homogeneous_degree = max(
            (
                sum(term.get("exponents", ()))
                for term in terms
                if isinstance(term, dict)
                and isinstance(term.get("exponents"), list)
                and all(type(value) is int for value in term["exponents"])
            ),
            default=0,
        )
        term_count = len(terms)
        if terms and all(
            isinstance(term, dict)
            and isinstance(term.get("exponents"), list)
            and len(term["exponents"]) == 3
            for term in terms
        ) and any(
            all(term["exponents"][variable] == 0 for term in terms)
            for variable in range(3)
        ):
            maximum_degree = 0
    elif isinstance(factors, list):
        homogeneous_degree = len(factors)
        support: set[tuple[int, ...]] = {(0, 0, 0)}
        for factor in factors:
            coefficients = (
                factor.get("coefficients") if isinstance(factor, dict) else None
            )
            if not isinstance(coefficients, list) or len(coefficients) != 3:
                return False
            active_variables = tuple(
                index
                for index, coefficient in enumerate(coefficients)
                if isinstance(coefficient, dict) and coefficient.get("num") != "0"
            )
            if not active_variables:
                return False
            support = {
                tuple(
                    exponent + (1 if variable == index else 0)
                    for index, exponent in enumerate(monomial)
                )
                for monomial in support
                for variable in active_variables
            }
        term_count = len(support)
        if len(support) == 1 and sum(value > 0 for value in next(iter(support))) < 3:
            maximum_degree = 0
    else:
        return False
    replay_cells = sum(
        (3 * ((degree + 2) * (degree + 1) // 2))
        * ((homogeneous_degree + degree + 1) * (homogeneous_degree + degree) // 2)
        for degree in range(maximum_degree + 1)
    )
    return term_count * replay_cells <= 1_000_000


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
        supports_input=_materialized_syzygy_supports,
    ),
    ExactReplayCheckerDeclaration(
        "polynomial.jacobian_syzygy.coefficients.materialize",
        GradedJacobianSyzygyRequest,
        "check_materialized_graded_jacobian_syzygy",
        "polynomial.jacobian-syzygy.graded-fraction-replay",
        entrypoint_module="jacobian_checkers.jacobian_syzygy",
        replay_method="standard-library exact rational graded-map replay",
        reason=(
            "operator-authorized exact rational checker independently reconstructs "
            "the stored homogeneous coefficient ledger without importing the "
            "SymPy producer"
        ),
        verification_capability_id=("polynomial.jacobian_syzygy.coefficients.verify"),
        verification_title="Verify a materialized Jacobian syzygy coefficient ledger",
        verification_description=(
            "Independently reconstruct every bounded homogeneous coefficient map, "
            "sparse entry, rank ledger, nonzero minor, and first kernel from the "
            "materialized producer result."
        ),
        verification_tags=(
            "verification",
            "exact",
            "polynomial",
            "jacobian",
            "syzygy",
            "coefficient-ledger",
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
