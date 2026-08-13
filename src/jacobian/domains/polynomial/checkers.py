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

_MAX_SYZYGY_REPLAY_WORK = 10_000_000


def _rational_decimal_digits(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    numerator = value.get("num")
    denominator = value.get("den")
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or not numerator
        or not denominator
    ):
        return None
    return len(numerator.lstrip("-")) + len(denominator)


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


def _polynomial_replay_shape(
    polynomial: dict[str, object], maximum_degree: int
) -> tuple[int, int, int, int] | None:
    body = polynomial.get("polynomial")
    terms = body.get("terms") if isinstance(body, dict) else None
    if not isinstance(terms, list):
        return None
    coefficient_digits = 0
    for term in terms:
        if not isinstance(term, dict):
            return None
        digits = _rational_decimal_digits(term.get("coefficient"))
        if digits is None:
            return None
        coefficient_digits += digits
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
    if (
        terms
        and all(
            isinstance(term, dict)
            and isinstance(term.get("exponents"), list)
            and len(term["exponents"]) == 3
            for term in terms
        )
        and any(
            all(term["exponents"][variable] == 0 for term in terms)
            for variable in range(3)
        )
    ):
        maximum_degree = 0
    return maximum_degree, homogeneous_degree, len(terms), coefficient_digits


def _factor_replay_shape(
    factors: list[object], maximum_degree: int
) -> tuple[int, int, int, int] | None:
    homogeneous_degree = len(factors)
    coefficient_digits = 0
    support: set[tuple[int, ...]] = {(0, 0, 0)}
    for factor in factors:
        coefficients = factor.get("coefficients") if isinstance(factor, dict) else None
        if not isinstance(coefficients, list) or len(coefficients) != 3:
            return None
        for coefficient in coefficients:
            digits = _rational_decimal_digits(coefficient)
            if digits is None:
                return None
            coefficient_digits += digits
        active_variables = tuple(
            index
            for index, coefficient in enumerate(coefficients)
            if isinstance(coefficient, dict) and coefficient.get("num") != "0"
        )
        if not active_variables:
            return None
        support = {
            tuple(
                exponent + (1 if variable == index else 0)
                for index, exponent in enumerate(monomial)
            )
            for monomial in support
            for variable in active_variables
        }
    if len(support) == 1 and sum(value > 0 for value in next(iter(support))) < 3:
        maximum_degree = 0
    return maximum_degree, homogeneous_degree, len(support), coefficient_digits


def _materialized_syzygy_supports(payload: object) -> bool:
    """Bound aggregate checker work while retaining cheap degree-zero cases."""

    if not isinstance(payload, dict) or type(payload.get("max_degree")) is not int:
        return False
    maximum_degree = payload["max_degree"]
    shape: tuple[int, int, int, int] | None
    if isinstance(payload.get("polynomial"), dict):
        shape = _polynomial_replay_shape(payload["polynomial"], maximum_degree)
    elif isinstance(payload.get("linear_factors"), list):
        shape = _factor_replay_shape(payload["linear_factors"], maximum_degree)
    else:
        return False
    if shape is None:
        return False
    maximum_degree, homogeneous_degree, term_count, coefficient_digits = shape
    replay_cells = sum(
        (3 * ((degree + 2) * (degree + 1) // 2))
        * ((homogeneous_degree + degree + 1) * (homogeneous_degree + degree) // 2)
        for degree in range(maximum_degree + 1)
    )
    return (
        term_count * replay_cells <= 1_000_000
        and coefficient_digits * replay_cells <= _MAX_SYZYGY_REPLAY_WORK
    )


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
