"""SymPy-backed exact root--critical-point profile computation."""

from __future__ import annotations

from typing import Any, Literal

import sympy

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.root_critical._models import (
    MAX_ROOT_CRITICAL_DEGREE,
    MAX_ROOT_CRITICAL_DISTANCE_DEGREE,
    MAX_ROOT_CRITICAL_PAIRS,
    RootCriticalDistanceProfile,
    RootCriticalDistanceRow,
    RootCriticalRectangle,
    RootCriticalRoot,
)
from jacobian.math.polynomials.values import RationalPolynomial

__all__ = ["root_critical_distance_profile"]


def _rational(value: object) -> CanonicalRational:
    rational = sympy.Rational(value)
    return CanonicalRational(num=int(rational.p), den=int(rational.q))


def _primitive_integer_poly(poly: sympy.Poly) -> sympy.Poly:
    _denominator, integer_poly = poly.clear_denoms(convert=True)
    _content, primitive = integer_poly.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    return sympy.Poly(primitive, poly.gens[0], domain=sympy.ZZ)


def _factor_key(poly: sympy.Poly) -> tuple[tuple[int, ...], ...]:
    primitive = _primitive_integer_poly(poly)
    return tuple((int(value),) for value in primitive.all_coeffs())


def _rectangle(
    poly: sympy.Poly, root_index: int, root: object
) -> RootCriticalRectangle:
    """Project SymPy's exact root-isolation object to rational bounds."""

    if not isinstance(root, sympy.RootOf) and getattr(root, "is_rational", False):
        value = _rational(root)
        zero = CanonicalRational(num=0, den=1)
        return RootCriticalRectangle(
            real_lower=value,
            real_upper=value,
            imaginary_lower=zero,
            imaginary_upper=zero,
        )

    root_of = sympy.CRootOf(poly.as_expr(), root_index)
    interval = root_of._get_interval()
    if getattr(root_of, "is_real", False):
        lower, upper = interval.a, interval.b
        zero = CanonicalRational(num=0, den=1)
        return RootCriticalRectangle(
            real_lower=_rational(lower),
            real_upper=_rational(upper),
            imaginary_lower=zero,
            imaginary_upper=zero,
        )
    (real_lower, imaginary_lower), (real_upper, imaginary_upper) = interval.as_tuple()
    return RootCriticalRectangle(
        real_lower=_rational(real_lower),
        real_upper=_rational(real_upper),
        imaginary_lower=_rational(imaginary_lower),
        imaginary_upper=_rational(imaginary_upper),
    )


def _root_value(root_index: int, roots: tuple[Any, ...]) -> Any:
    # ``all_roots`` gives SymPy's exact radicals for the low-degree slice.  A
    # RootOf object is intentionally not allowed to reach the distance kernel:
    # SymPy cannot reduce products of independently indexed RootOf values
    # reliably, while the radical expressions have a stable minpoly path.
    value = roots[root_index]
    if isinstance(value, sympy.RootOf):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.root_critical.backend_root_form",
            message="the admitted exact profile requires a backend root expression",
        )
    return value


def _family(
    poly: sympy.Poly,
) -> tuple[tuple[RootCriticalRoot, ...], tuple[Any, ...]]:
    primitive_factors: list[tuple[sympy.Poly, int]] = []
    for factor, multiplicity in poly.factor_list()[1]:
        primitive_factors.append((_primitive_integer_poly(factor), int(multiplicity)))
    primitive_factors.sort(key=lambda item: _factor_key(item[0]))
    records: list[RootCriticalRoot] = []
    values: list[Any] = []
    for factor, multiplicity in primitive_factors:
        exact_roots = tuple(factor.all_roots())
        for root_index in range(factor.degree()):
            root = _root_value(root_index, exact_roots)
            records.append(
                RootCriticalRoot(
                    axis_index=len(records),
                    factor=tuple(int(value) for value in factor.all_coeffs()),
                    root_index=root_index,
                    multiplicity=multiplicity,
                    rectangle=_rectangle(factor, root_index, root),
                )
            )
            values.append(root)
    return tuple(records), tuple(values)


def _select_real_algebraic_root(minimal: sympy.Poly, distance: Any) -> int:
    """Select the real minpoly root by certified isolation, not expression identity."""

    if sympy.simplify(distance) == 0:
        roots = minimal.real_roots()
        for index, candidate in enumerate(roots):
            if candidate == 0:
                return index
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_root_selection",
            message="exact real distance root could not be selected",
        )
    precision = 16
    intervals = list(minimal.intervals())
    while precision <= 1024:
        approximation = sympy.re(distance.evalf(precision))
        hits = [
            index
            for index, ((lower, upper), _multiplicity) in enumerate(intervals)
            if lower <= approximation <= upper
        ]
        if len(hits) == 1:
            return hits[0]
        intervals = list(minimal.intervals(eps=sympy.Rational(1, 10**precision)))
        precision *= 2
    raise OperationDomainValidationError(
        location=("pairs",),
        code="polynomial.root_critical.distance_root_selection",
        message="exact real distance root could not be selected",
    )


def _distance_value(
    left: Any,
    right: Any,
) -> tuple[
    RealAlgebraicValue,
    RationalIsolatingInterval,
    Literal["POSITIVE", "ZERO_DISTANCE"],
]:
    variable = sympy.Symbol("distance")
    distance = sympy.simplify(
        sympy.expand((left - right) * sympy.conjugate(left - right))
    )
    try:
        minimal = sympy.Poly(
            sympy.minpoly(distance, variable), variable, domain=sympy.QQ
        )
    except (NotImplementedError, ValueError, sympy.PolynomialError) as exc:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_minpoly",
            message="exact distance minimal-polynomial construction is unsupported",
        ) from exc
    minimal = _primitive_integer_poly(minimal)
    degree = minimal.degree()
    if degree > MAX_ROOT_CRITICAL_DISTANCE_DEGREE:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_degree_bound",
            message="exact distance degree exceeds the bounded real-algebraic carrier",
        )
    coefficients = tuple(int(value) for value in minimal.all_coeffs())
    selected_index = _select_real_algebraic_root(minimal, distance)
    intervals = minimal.intervals()
    lower, upper = intervals[selected_index][0]
    value = RealAlgebraicValue._from_admitted_polynomial(
        polynomial=coefficients,
        real_root_index=selected_index,
    )
    interval = RationalIsolatingInterval(
        lower=_rational(lower),
        upper=_rational(upper),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )
    kind: Literal["POSITIVE", "ZERO_DISTANCE"] = (
        "ZERO_DISTANCE" if sympy.simplify(distance) == 0 else "POSITIVE"
    )
    if lower < 0:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_sign",
            message="exact distance isolation did not establish nonnegativity",
        )
    return value, interval, kind


def _admit(
    polynomial: RationalPolynomial,
    *,
    max_pair_rows: int,
) -> tuple[sympy.Poly, int, int]:
    if len(polynomial.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.root_critical.univariate_required",
            message="root-critical profiles require one polynomial variable",
        )
    source = rational_polynomial_to_sympy(polynomial)
    degree = source.degree()
    if degree <= 0:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.root_critical.nonconstant_required",
            message="root-critical profiles require a nonconstant polynomial",
        )
    if degree > MAX_ROOT_CRITICAL_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.degree_bound",
            message="source degree exceeds the exact root-critical profile envelope",
        )
    source_digits = max(
        (
            max(
                len(format_canonical_integer(abs(term.coefficient.num))),
                len(format_canonical_integer(term.coefficient.den)),
            )
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )
    if source_digits > 128:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.coefficient_bound",
            message="source coefficient height exceeds the exact root-critical profile envelope",
        )
    critical_degree = max(0, degree - 1)
    source_factors = source.factor_list()[1]
    derivative_backend = source.diff()
    derivative_factors = derivative_backend.factor_list()[1] if critical_degree else ()
    if any(factor.degree() > 4 for factor, _ in (*source_factors, *derivative_factors)):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.root_carrier_bound",
            message="the exact-root carrier admits irreducible factors through degree four",
        )
    if any(
        isinstance(root, sympy.RootOf)
        for factor, _ in (*source_factors, *derivative_factors)
        for root in factor.all_roots()
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.root_carrier_backend_form",
            message="the exact-root carrier requires explicit maintained algebraic expressions",
        )
    root_count = sum(factor.degree() for factor, _ in source_factors)
    critical_count = (
        sum(factor.degree() for factor, _ in derivative_factors)
        if critical_degree
        else 0
    )
    max_distance_degree = max(
        (
            source_factor.degree() * critical_factor.degree()
            for source_factor, _ in source_factors
            for critical_factor, _ in derivative_factors
        ),
        default=0,
    )
    if max_distance_degree > MAX_ROOT_CRITICAL_DISTANCE_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.distance_degree_bound",
            message="the source can produce a distance algebraic degree beyond the admitted carrier",
        )
    pair_count = root_count * critical_count
    if pair_count > MAX_ROOT_CRITICAL_PAIRS or pair_count > max_pair_rows:
        raise OperationResourceAdmissionError(
            location=("max_pair_rows",),
            code="polynomial.root_critical.pair_output_bound",
            message="complete root-critical pair expansion exceeds the admitted row budget",
        )
    return source, root_count, critical_count


def root_critical_distance_profile(
    polynomial: RationalPolynomial,
    *,
    max_pair_rows: int = MAX_ROOT_CRITICAL_PAIRS,
) -> RootCriticalDistanceProfile:
    """Return every distinct root/critical pair and its exact squared distance."""

    source, _root_count, _critical_count = _admit(
        polynomial,
        max_pair_rows=max_pair_rows,
    )
    derivative_backend = source.diff()
    variables = polynomial.variables
    derivative = rational_polynomial_from_sympy(derivative_backend, variables)
    roots, root_values = _family(source)
    critical_points, critical_values = (
        _family(derivative_backend) if derivative_backend.degree() > 0 else ((), ())
    )
    rows: list[RootCriticalDistanceRow] = []
    for root_record, root in zip(roots, root_values, strict=True):
        for critical_record, critical in zip(
            critical_points, critical_values, strict=True
        ):
            value, interval, kind = _distance_value(root, critical)
            rows.append(
                RootCriticalDistanceRow(
                    root_axis_index=root_record.axis_index,
                    critical_axis_index=critical_record.axis_index,
                    distance_squared=value,
                    isolating_interval=interval,
                    kind=kind,
                )
            )
    result = RootCriticalDistanceProfile._from_kernel(
        source_polynomial=polynomial,
        derivative=derivative,
        roots=roots,
        critical_points=critical_points,
        pairs=tuple(rows),
    )
    return result
