"""Contracts and bounded replay for multivariate factorization."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.multivariate._models import (
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _MULTIVARIATE_MIN_VARIABLES,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

# Public output-term budget for one converted irreducible factor.  The
# operation converter and result replay use the same bound, so a typed
# output-capacity outcome can reproduce the exact limit that the producer hit.
_MAX_FACTOR_OUTPUT_TERMS = 1_024


class MultivariateFactorRequest(StrictModel):
    """Exact factorization request over ``QQ[variables]`` for nonzero multivariate polynomials."""

    polynomial: RationalPolynomial = Field(
        description=(
            "Nonzero multivariate polynomial in QQ[variables] with at least "
            "two variables (univariate factorization is owned by "
            "polynomial.factor.compute); terms, exponents, and coefficients "
            "must respect the operation's exact budget."
        )
    )

    @model_validator(mode="after")
    def require_factor_budget(self) -> Self:
        if len(self.polynomial.variables) < _MULTIVARIATE_MIN_VARIABLES:
            raise _validation_error(
                f"multivariate factorization requires at least {_MULTIVARIATE_MIN_VARIABLES} variables; "
                "univariate polynomials are handled by polynomial.factor.compute"
            )
        if not self.polynomial.polynomial.terms:
            raise _validation_error("zero polynomial has no factorization")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        _require_representable_content(self.polynomial)
        return self


def _require_representable_content(polynomial: RationalPolynomial) -> None:
    """Bound aggregate rational content before the factor backend expands it."""

    fractions = [term.coefficient.as_fraction() for term in polynomial.polynomial.terms]
    common_denominator = reduce(lcm, (value.denominator for value in fractions), 1)
    scaled = [
        value.numerator * (common_denominator // value.denominator)
        for value in fractions
    ]
    content_numerator = gcd(*scaled)
    canonical_bound = 10**MAX_CANONICAL_RATIONAL_DIGITS
    if (
        abs(content_numerator) >= canonical_bound
        or common_denominator >= canonical_bound
    ):
        raise _validation_error(
            "aggregate rational content exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit representable bound"
        )
    primitive_bound = 10**_MAX_MULTIVARIATE_COEFFICIENT_DIGITS
    for value in scaled:
        if abs(value // content_numerator) >= primitive_bound:
            raise _validation_error(
                "primitive integer coefficients exceed the "
                f"{_MAX_MULTIVARIATE_COEFFICIENT_DIGITS}-digit operation budget"
            )


class MultivariateIrreducibleFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_MULTIVARIATE_EXPONENT)


class MultivariateFactorResult(StrictModel):
    """Exact factorization outcome over ``QQ[variables]``.

    ``FACTORIZED`` carries the full content-and-monic-irreducibles
    decomposition.  ``OUTPUT_BUDGET_EXCEEDED`` reports, as a typed bounded
    outcome, that the exact factorization is beyond this operation's
    public output bounds: either an irreducible factor exceeds the public
    output-term budget or the serialized exact decomposition exceeded the
    declared transport bound.  ``EXECUTION_FAILED`` is not a mathematical
    conclusion: the worker was stopped by its deadline or cancellation,
    killed by an enforced resource cap such as its CPU or address-space
    budget, crashed, or could not be contained, so no factorization was
    obtained and callers may retry.
    For both non-FACTORIZED statuses ``reconstructed`` restates the
    requested polynomial unchanged, ``coefficient`` carries the exact
    positive rational content of that polynomial, and ``factors`` is empty.
    """

    status: Literal[
        "FACTORIZED",
        "OUTPUT_BUDGET_EXCEEDED",
        "EXECUTION_FAILED",
    ] = "FACTORIZED"
    coefficient: CanonicalRational
    factors: tuple[MultivariateIrreducibleFactor, ...] = Field(max_length=128)
    reconstructed: RationalPolynomial
    normalization: Literal["CONTENT_AND_MONIC_IRREDUCIBLES"] | None = (
        "CONTENT_AND_MONIC_IRREDUCIBLES"
    )
    product_reconstruction: Literal["EXACT"] | None = "EXACT"

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise _validation_error("factorization coefficient must be nonzero")
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if self.status != "FACTORIZED":
            from jacobian.math.polynomials.multivariate import _factor_backend

            if self.factors:
                raise _validation_error(
                    "non-FACTORIZED outcomes carry no irreducible factors"
                )
            if (
                self.normalization is not None
                or self.product_reconstruction is not None
            ):
                raise _validation_error(
                    "non-FACTORIZED outcomes declare no normalization or "
                    "product reconstruction"
                )
            if _factor_backend.primitive_content_fraction(self.reconstructed) != (
                self.coefficient.as_fraction()
            ):
                raise _validation_error(
                    "outcome coefficient does not match the exact content "
                    "of the restated polynomial"
                )
            if self.status == "OUTPUT_BUDGET_EXCEEDED":
                _verify_output_budget_exceeded_claim(
                    self.coefficient, self.reconstructed
                )
            return self
        if (
            self.normalization != "CONTENT_AND_MONIC_IRREDUCIBLES"
            or self.product_reconstruction != "EXACT"
        ):
            raise _validation_error(
                "FACTORIZED outcomes declare content-and-monic-irreducibles "
                "normalization and exact product reconstruction"
            )
        if not self.reconstructed.polynomial.terms:
            raise _validation_error("reconstructed polynomial must be nonzero")
        _check_factor_records(self.factors, self.reconstructed.variables)
        _require_aggregate_degree_consistent(self.factors, self.reconstructed)
        _require_distinct_canonical_order(self.factors)
        _verify_monic_irreducibles(self.factors)
        _verify_exact_reconstruction(
            self.coefficient,
            self.factors,
            self.reconstructed,
        )
        return self


_FactorContentKey = tuple[tuple[tuple[int, ...], str, str], ...]
_SympyFactorKey = tuple[tuple[tuple[int, ...], int, int], ...]


def _factor_content_key(record: MultivariateIrreducibleFactor) -> _FactorContentKey:
    return tuple(
        (term.exponents, term.coefficient.num, term.coefficient.den)
        for term in record.factor.polynomial.terms
    )


def _factor_total_degree(record: MultivariateIrreducibleFactor) -> int:
    return max(
        (sum(term.exponents) for term in record.factor.polynomial.terms),
        default=0,
    )


def _reconstructed_total_degree(reconstructed: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in reconstructed.polynomial.terms),
        default=0,
    )


def _require_aggregate_degree_consistent(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Reject aggregate degree mismatches before any product expansion."""

    target = _reconstructed_total_degree(reconstructed)
    aggregate = 0
    for record in factors:
        aggregate += _factor_total_degree(record) * record.multiplicity
        if aggregate > target:
            raise _validation_error(
                "aggregate irreducible degree exceeds the reconstructed "
                "total degree; the factorization product cannot match"
            )


def _monic_content_fraction(content: Any) -> Fraction:
    """Extract the exact rational content returned by ``_monic_decomposition``."""

    leading = getattr(content, "LC", None)
    value = leading() if callable(leading) else content
    return Fraction(int(value.p), int(value.q))


def _verify_output_budget_exceeded_claim(
    coefficient: CanonicalRational,
    reconstructed: RationalPolynomial,
) -> None:
    """Re-derive a claimed ``OUTPUT_BUDGET_EXCEEDED`` status from its source."""

    from jacobian.math.polynomials.multivariate import _factor_backend
    from jacobian.math.polynomials.multivariate._factor_backend import (
        FactorBackendExhaustedError,
        FactorBackendInterruptedError,
    )

    if _factor_backend.primitive_content_fraction(reconstructed) != (
        coefficient.as_fraction()
    ):
        raise _validation_error(
            "budget-exceeded outcome coefficient does not match the exact "
            "content of the restated polynomial"
        )
    try:
        decomposition = _factor_backend.run_bounded_factorization(
            reconstructed,
            wall_seconds=_factor_backend.FACTOR_VERIFY_WALL_SECONDS,
        )
    except FactorBackendExhaustedError:
        return
    except FactorBackendInterruptedError as exc:
        raise _validation_error(
            "budget-exceeded outcome could not be re-derived because the "
            "verification replay was itself stopped before completing"
        ) from exc
    from jacobian.math.polynomials._conversions import (
        rational_polynomial_from_sympy,
        rational_polynomial_to_sympy,
    )
    from jacobian.math.polynomials._sympy import _monic_decomposition

    source = rational_polynomial_to_sympy(reconstructed)
    _content, raw_factors, _reconstructed = _monic_decomposition(
        source,
        decomposition,
        label="multivariate factorization",
    )
    for factor, _multiplicity in raw_factors:
        try:
            rational_polynomial_from_sympy(
                factor,
                reconstructed.variables,
                maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            )
        except ValueError as exc:
            if "term operation budget" in str(exc):
                return
            raise
    raise _validation_error(
        "claimed output-budget exceedance is not reproduced by the exact "
        "factorization of the restated polynomial"
    )


def _check_factor_records(
    factors: tuple[MultivariateIrreducibleFactor, ...],
    variables: tuple[str, ...],
) -> None:
    """Enforce the reconstruction-safe envelope before any SymPy expansion."""

    for record in factors:
        if record.factor.variables != variables:
            raise _validation_error("irreducible factors must use the source ring")
        require_polynomial_budget(
            record.factor,
            maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )
        if _factor_total_degree(record) == 0:
            raise _validation_error("irreducible factor must be non-constant")


def _require_distinct_canonical_order(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    seen: set[_FactorContentKey] = set()
    for key in (_factor_content_key(record) for record in factors):
        if key in seen:
            raise _validation_error("irreducible factors must be distinct")
        seen.add(key)
    ordered = tuple(
        sorted(
            factors,
            key=lambda record: (
                record.multiplicity,
                _factor_total_degree(record),
                _factor_content_key(record),
            ),
        ),
    )
    if factors != ordered:
        raise _validation_error("irreducible factors must use canonical order")


def _require_monic(poly: Any, factor: RationalPolynomial) -> None:
    lc = poly.LC()
    if getattr(lc, "p", None) != 1 or getattr(lc, "q", None) != 1:
        raise _validation_error(f"irreducible factor {factor} is not monic")


def _verify_monic_irreducibles(
    factors: tuple[MultivariateIrreducibleFactor, ...],
) -> None:
    """Enforce CONTENT_AND_MONIC_IRREDUCIBLES on every listed factor."""

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    for record in factors:
        poly = rational_polynomial_to_sympy(record.factor)
        try:
            _require_monic(poly, record.factor)
            if not poly.is_irreducible:
                raise _validation_error(f"factor {record.factor} is not irreducible")
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise _validation_error("invalid factor normalization check") from exc


def _sympy_factor_key(poly: Any) -> _SympyFactorKey:
    """Return the canonical hashable form of one monic QQ ``Poly``."""

    return tuple(
        sorted(
            (tuple(monom), int(coeff.p), int(coeff.q)) for monom, coeff in poly.terms()
        )
    )


def _verify_exact_reconstruction(
    coefficient: CanonicalRational,
    factors: tuple[MultivariateIrreducibleFactor, ...],
    reconstructed: RationalPolynomial,
) -> None:
    """Replay the bounded factorization and compare its unique monic factors."""

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._sympy import _monic_decomposition
    from jacobian.math.polynomials.multivariate import _factor_backend
    from jacobian.math.polynomials.multivariate._factor_backend import (
        FactorBackendExhaustedError,
        FactorBackendInterruptedError,
    )

    try:
        decomposition = _factor_backend.run_bounded_factorization(
            reconstructed,
            wall_seconds=_factor_backend.FACTOR_VERIFY_WALL_SECONDS,
        )
        source = rational_polynomial_to_sympy(reconstructed)
        content, raw_factors, _ = _monic_decomposition(
            source,
            decomposition,
            label="multivariate factorization",
        )
        claimed: dict[_SympyFactorKey, int] = {}
        for record in factors:
            key = _sympy_factor_key(rational_polynomial_to_sympy(record.factor))
            claimed[key] = claimed.get(key, 0) + record.multiplicity
        replayed: dict[_SympyFactorKey, int] = {}
        for factor, multiplicity in raw_factors:
            key = _sympy_factor_key(factor)
            replayed[key] = replayed.get(key, 0) + multiplicity
        if (
            _monic_content_fraction(content) != coefficient.as_fraction()
            or claimed != replayed
        ):
            raise _validation_error(
                "factorization product does not equal reconstructed polynomial"
            )
    except ValueError:
        raise
    except (FactorBackendExhaustedError, FactorBackendInterruptedError) as exc:
        raise _validation_error(
            "factorization verification could not reproduce the exact "
            "factorization within the declared work budget"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise _validation_error("invalid factorization reconstruction") from exc


__all__ = [
    "MultivariateFactorRequest",
    "MultivariateFactorResult",
    "MultivariateIrreducibleFactor",
]
