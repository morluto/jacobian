"""Exact Sturm chain and root counting kernels backed by SymPy."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._models import (
    PolynomialTerm,
    RootCountResult,
    SturmChainResult,
    UnivariatePolynomial,
    _require_bounded_integer_coefficients,
)
from jacobian.math.polynomials.real_algebra._models import (
    _validation_error as _real_algebra_validation_error,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel import (
    compute_strict_sublevel_payload,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel_models import (
    MAX_STRICT_SUBLEVEL_BOUNDARY_HEIGHT_DIGITS,
    MAX_STRICT_SUBLEVEL_DEGREE,
    MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
    MAX_STRICT_SUBLEVEL_ISOLATION_WORK,
    MAX_STRICT_SUBLEVEL_TERMS,
    StrictSublevelMeasureResult,
    _level_polynomial_height_digits,
    _polynomial_degree,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

__all__ = [
    "compute_root_count",
    "compute_strict_sublevel_measure",
    "compute_sturm_chain",
    "root_count",
    "sturm_chain",
]


def _to_sympy_poly(terms: list[tuple[Fraction, int]]) -> Any:
    from sympy import Poly, Rational, Symbol

    x = Symbol("x")
    poly_dict = {exp: Rational(val.numerator, val.denominator) for val, exp in terms}
    return Poly(poly_dict, x, domain="QQ")


def sturm_chain(terms: list[tuple[Fraction, int]]) -> list[list[tuple[Fraction, int]]]:
    """Compute the exact Sturm subresultant chain of a univariate polynomial."""

    from sympy import sturm

    poly = _to_sympy_poly(terms)
    chain = sturm(poly)
    return [_sympy_poly_to_terms(p) for p in chain]


def root_count(
    terms: list[tuple[Fraction, int]],
    lower: Fraction,
    upper: Fraction,
) -> int:
    """Count distinct real roots in the closed interval [lower, upper]."""

    from sympy import Rational

    poly = _to_sympy_poly(terms)
    chain = _build_sturm_chain(poly)

    a = Rational(lower.numerator, lower.denominator)
    b = Rational(upper.numerator, upper.denominator)

    sign_changes_a = _sign_changes(chain, a)
    sign_changes_b = _sign_changes(chain, b)
    root_at_lower = _evaluates_to_zero(poly, a)

    # The zero-skipping variation is right-continuous, so the difference counts
    # the half-open interval (lower, upper]. Add the root exactly at ``lower``
    # back to obtain the advertised closed interval.
    return sign_changes_a - sign_changes_b + (1 if root_at_lower else 0)


def _build_sturm_chain(poly: Any) -> list[Any]:
    from sympy import sturm

    return list(sturm(poly))


def _sign_changes(chain: list[Any], point: Any) -> int:
    if len(chain) == 0:
        return 0
    signs: list[int] = []
    for poly in chain:
        value = poly.as_expr().subs(poly.gen, point)
        if value != 0:
            signs.append(1 if value > 0 else -1)
    count = 0
    for index in range(1, len(signs)):
        if signs[index] != signs[index - 1]:
            count += 1
    return count


def _evaluates_to_zero(poly: Any, point: Any) -> bool:
    return bool(poly.as_expr().subs(poly.gen, point) == 0)


def _sympy_poly_to_terms(poly: Any) -> list[tuple[Fraction, int]]:
    result: list[tuple[Fraction, int]] = []
    for exps, coeff in poly.as_dict().items():
        if coeff == 0:
            continue
        if hasattr(coeff, "p") and hasattr(coeff, "q"):
            fraction = Fraction(int(coeff.p), int(coeff.q))
        else:
            fraction = Fraction(coeff)
        result.append((fraction, int(exps[0])))
    return result


def _admit_strict_sublevel(
    polynomial: RationalPolynomial,
    threshold: CanonicalRational,
    lower: CanonicalRational,
    upper: CanonicalRational,
) -> None:
    if len(polynomial.variables) != 1:
        raise _validation_error(
            "variable_count", "strict sublevel measure requires one polynomial variable"
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_STRICT_SUBLEVEL_TERMS,
        maximum_exponent=MAX_STRICT_SUBLEVEL_DEGREE,
        maximum_coefficient_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
        label="strict sublevel polynomial",
    )
    for value, label in (
        (threshold, "strict sublevel threshold"),
        (lower, "strict sublevel lower scope endpoint"),
        (upper, "strict sublevel upper scope endpoint"),
    ):
        require_bounded_rational(
            value, max_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS, label=label
        )
    if threshold.as_fraction() < 0:
        raise _validation_error(
            "negative_threshold", "strict sublevel threshold must be nonnegative"
        )
    if lower.as_fraction() > upper.as_fraction():
        raise _validation_error(
            "scope_order", "strict sublevel lower endpoint must not exceed upper"
        )
    if (
        threshold.as_fraction() == 0
        or _polynomial_degree(polynomial) == 0
        or lower == upper
    ):
        return
    boundary_heights = []
    for subtract_threshold, label in ((True, "f-threshold"), (False, "f+threshold")):
        height_digits = _level_polynomial_height_digits(
            polynomial,
            threshold,
            subtract_threshold=subtract_threshold,
        )
        if height_digits > MAX_STRICT_SUBLEVEL_BOUNDARY_HEIGHT_DIGITS:
            raise _validation_error(
                "boundary_height",
                f"primitive {label} height exceeds the {MAX_STRICT_SUBLEVEL_BOUNDARY_HEIGHT_DIGITS}-digit root-isolation bound",
            )
        boundary_heights.append(height_digits)
    degree = _polynomial_degree(polynomial)
    isolation_work = degree**5 * sum(boundary_heights)
    if isolation_work > MAX_STRICT_SUBLEVEL_ISOLATION_WORK:
        raise _validation_error(
            "isolation_work",
            "strict sublevel exact-root isolation exceeds the work bound "
            f"(degree^5*level-height-sum={isolation_work} > {MAX_STRICT_SUBLEVEL_ISOLATION_WORK}); reduce degree or coefficient/threshold height",
        )


def _run_admission(
    polynomial: RationalPolynomial,
    threshold: CanonicalRational,
    lower: CanonicalRational,
    upper: CanonicalRational,
) -> None:
    try:
        _admit_strict_sublevel(polynomial, threshold, lower, upper)
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.strict_sublevel_admission", message=str(exc)
        ) from exc


def _poly_to_terms(poly: UnivariatePolynomial) -> list[tuple[Fraction, int]]:
    return [(t.coefficient.as_fraction(), t.exponent) for t in poly.terms]


def _terms_to_poly(terms: list[tuple[Fraction, int]]) -> UnivariatePolynomial:
    return UnivariatePolynomial(
        terms=tuple(
            PolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coeff),
                exponent=exp,
            )
            for coeff, exp in terms
            if coeff != 0
        )
    )


def _admit_integer_polynomial(polynomial: UnivariatePolynomial) -> None:
    try:
        _require_bounded_integer_coefficients(polynomial)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("polynomial",), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.real_algebra_coefficient_bound",
            message=str(exc),
        ) from exc


def compute_sturm_chain(poly: UnivariatePolynomial) -> SturmChainResult:
    _admit_integer_polynomial(poly)
    if max(term.exponent for term in poly.terms) < 1:
        error = _real_algebra_validation_error(
            "constant_input", "Sturm chain requires a non-constant polynomial"
        )
        raise OperationDomainValidationError(
            location=("polynomial",), code=error.type, message=error.message()
        )
    terms = _poly_to_terms(poly)
    chain = sturm_chain(terms)
    degree = max(t.exponent for t in poly.terms)
    return SturmChainResult(
        chain=tuple(_terms_to_poly(c) for c in chain),
        degree=degree,
    )


def compute_root_count(
    polynomial: UnivariatePolynomial,
    lower_bound: CanonicalRational,
    upper_bound: CanonicalRational,
) -> RootCountResult:
    _admit_integer_polynomial(polynomial)
    if lower_bound.as_fraction() > upper_bound.as_fraction():
        error = _real_algebra_validation_error(
            "interval_order", "lower bound must not exceed upper bound"
        )
        raise OperationDomainValidationError(
            location=("lower", "upper"), code=error.type, message=error.message()
        )
    terms = _poly_to_terms(polynomial)
    lower = lower_bound.as_fraction()
    upper = upper_bound.as_fraction()
    count = root_count(terms, lower, upper)
    return RootCountResult(
        source_polynomial=polynomial,
        root_count=count,
        lower=lower_bound,
        upper=upper_bound,
    )


def compute_strict_sublevel_measure(
    polynomial: RationalPolynomial,
    threshold: CanonicalRational,
    lower: CanonicalRational,
    upper: CanonicalRational,
) -> StrictSublevelMeasureResult:
    _run_admission(polynomial, threshold, lower, upper)
    payload = compute_strict_sublevel_payload(polynomial, threshold, lower, upper)
    return StrictSublevelMeasureResult._from_kernel(
        polynomial,
        threshold,
        lower,
        upper,
        components=payload.components,
        measure=payload.measure,
    )
