"""SymPy-backed exact root--critical-point profile computation."""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any, Literal, NoReturn

import sympy

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    RequestCancellationSignal,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
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
    MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS,
    RootCriticalDistanceProfile,
    RootCriticalDistanceRow,
    RootCriticalRectangle,
    RootCriticalRoot,
)
from jacobian.math.polynomials.root_critical._profile_process import (
    PROFILE_STDOUT_BYTES,
    run_profile_worker_process,
)
from jacobian.math.polynomials.values import RationalPolynomial

_PROFILE_FINALIZATION_SECONDS = 1.0

_NTH_ROOT_RELATIVE_BITS = 96

__all__ = ["root_critical_distance_profile"]

ROOT_CRITICAL_WALL_SECONDS = 60.0


def _rational(value: object) -> CanonicalRational:
    rational = sympy.Rational(value)
    return CanonicalRational(num=int(rational.p), den=int(rational.q))


def _rational_fraction(value: object) -> Fraction:
    rational = sympy.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def _primitive_integer_poly(poly: sympy.Poly) -> sympy.Poly:
    _denominator, integer_poly = poly.clear_denoms(convert=True)
    _content, primitive = integer_poly.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    return sympy.Poly(primitive, poly.gens[0], domain=sympy.ZZ)


def _factor_key(poly: sympy.Poly) -> tuple[tuple[int, ...], ...]:
    primitive = _primitive_integer_poly(poly)
    return tuple((int(value),) for value in primitive.all_coeffs())


def _rectangle(root: object) -> RootCriticalRectangle:
    """Build a request-local isolating rectangle from the exact root expression."""

    if not isinstance(root, sympy.RootOf) and getattr(root, "is_rational", False):
        value = _rational(root)
        zero = CanonicalRational(num=0, den=1)
        return RootCriticalRectangle(
            real_lower=value,
            real_upper=value,
            imaginary_lower=zero,
            imaginary_upper=zero,
        )

    try:
        real_lo, real_hi, imag_lo, imag_hi = _enclose_sympy(root)
    except (ValueError, AttributeError, TypeError):
        real_lo, real_hi, imag_lo, imag_hi = _evalf_containing_box(root)
    return RootCriticalRectangle(
        real_lower=_fit_rectangle_component(real_lo, round_up=False),
        real_upper=_fit_rectangle_component(real_hi, round_up=True),
        imaginary_lower=_fit_rectangle_component(imag_lo, round_up=False),
        imaginary_upper=_fit_rectangle_component(imag_hi, round_up=True),
    )


def _component_digit_count(value: int) -> int:
    magnitude = abs(value)
    if magnitude < 10:
        return 1
    return len(str(magnitude))


def _fit_rectangle_component(value: Fraction, *, round_up: bool) -> CanonicalRational:
    """Keep isolating endpoints inside the published rectangle digit envelope."""

    if (
        _component_digit_count(value.numerator)
        <= MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
        and _component_digit_count(value.denominator)
        <= MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
    ):
        return CanonicalRational.from_fraction(value)
    scale_power = MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS - 1
    if value != 0:
        # Place the grid relative to the component's own magnitude. A fixed
        # absolute grid rounds a root below ``10**-31`` to zero, which would
        # publish an origin-containing rectangle for a root away from the axis
        # origin and make the selected root ambiguous.
        decimal_exponent = _component_digit_count(
            value.numerator
        ) - _component_digit_count(value.denominator)
        if decimal_exponent < 0:
            scale_power -= decimal_exponent
    scale = 10**scale_power
    scaled = value * scale
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if remainder:
        if round_up:
            # The denominator is positive, so ``quotient`` is the floor and the
            # ceiling is one higher for every sign. Rounding a negative upper
            # endpoint down would narrow the certified enclosure below its root.
            quotient += 1
        elif scaled.numerator < 0:
            quotient -= 1
    return CanonicalRational.from_fraction(Fraction(quotient, scale))


def _simplest_rational_between(low: Fraction, high: Fraction) -> Fraction:
    """Smallest-denominator rational strictly inside ``(low, high)``."""

    if not low < high:
        raise ValueError("separating interval must be nonempty")
    floor_low = low.numerator // low.denominator
    if Fraction(floor_low + 1) < high:
        return Fraction(floor_low + 1, 1)
    frac_low = low - floor_low
    frac_high = high - floor_low
    if frac_low == 0:
        # ``(0, frac_high)`` shifted by ``floor_low``: ``1/k`` for the least
        # ``k`` above ``1/frac_high`` is the unique simplest choice.
        inverse = Fraction(frac_high.denominator, frac_high.numerator)
        step = inverse.numerator // inverse.denominator + 1
        return Fraction(floor_low * step + 1, step)
    inner = _simplest_rational_between(
        Fraction(frac_high.denominator, frac_high.numerator),
        Fraction(frac_low.denominator, frac_low.numerator),
    )
    return Fraction(
        floor_low * inner.numerator + inner.denominator, inner.numerator
    )


def _representable_separating_bound(low: Fraction, high: Fraction) -> CanonicalRational:
    """A carrier-representable rational strictly inside ``(low, high)``."""

    candidate = _simplest_rational_between(low, high)
    if (
        _component_digit_count(candidate.numerator)
        > MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
        or _component_digit_count(candidate.denominator)
        > MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.sibling_separation",
            message=(
                "a sibling root separation needs more exact digits than the "
                "admitted root-component envelope"
            ),
        )
    return CanonicalRational.from_fraction(candidate)


def _evalf_containing_box(
    root: object,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Contain a radical in a request-local box without reading CRootOf's cache.

    The expression is outside the certified enclosure grammar, so evaluate it at
    two precisions and pad by their agreement scaled to the component magnitude.
    A fixed absolute pad is unsound for roots with large components, where a
    low-precision approximation can be wrong by far more than the pad.
    """

    coarse = root.evalf(30)  # type: ignore[attr-defined]
    fine = root.evalf(60)  # type: ignore[attr-defined]
    coarse_real, coarse_imag = coarse.as_real_imag()
    fine_real, fine_imag = fine.as_real_imag()
    real = _rational_fraction(coarse_real)
    imag = _rational_fraction(coarse_imag)
    real_error = abs(real - _rational_fraction(fine_real))
    imag_error = abs(imag - _rational_fraction(fine_imag))
    # Add a magnitude-scaled guard so a 30-digit approximation of a large
    # component is contained even when the two precisions happen to agree.
    real_guard = real_error + _magnitude_guard(real)
    imag_guard = imag_error + _magnitude_guard(imag)
    return (
        real - real_guard,
        real + real_guard,
        imag - imag_guard,
        imag + imag_guard,
    )


def _magnitude_guard(value: Fraction) -> Fraction:
    """Absolute error guard for a 30-significant-digit approximation."""

    if not value:
        return Fraction(1, 10**30)
    magnitude = abs(value.numerator) // value.denominator
    digits = len(str(magnitude)) if magnitude else 1
    # 30 significant digits at this magnitude, with a factor-of-ten safety.
    guard: Fraction = Fraction(10, 10**30) * Fraction(10 ** (digits - 1), 1)
    return guard


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


def _algebraic_root_value(
    factor: sympy.Poly, root_index: int, rectangle: RootCriticalRectangle
) -> RealAlgebraicValue | ComplexAlgebraicValue:
    coefficients = tuple(int(value) for value in factor.all_coeffs())
    imag_zero = (
        rectangle.imaginary_lower.as_fraction() == 0
        and rectangle.imaginary_upper.as_fraction() == 0
    )
    if imag_zero:
        selected = None
        target_lower = rectangle.real_lower.as_fraction()
        target_upper = rectangle.real_upper.as_fraction()
        for index, candidate in enumerate(factor.intervals()):
            lower, upper = candidate[0]
            if Fraction(lower) <= target_lower and target_upper <= Fraction(upper):
                selected = index
                break
        if selected is None:
            raise OperationDomainValidationError(
                location=("polynomial",),
                code="polynomial.root_critical.real_root_index",
                message="real algebraic root index could not be selected",
            )
        return RealAlgebraicValue._from_admitted_polynomial(
            polynomial=coefficients,
            real_root_index=selected,
        )
    return ComplexAlgebraicValue._from_admitted_polynomial(
        polynomial=coefficients,
        root_index=root_index,
    )


def _family(
    poly: sympy.Poly,
) -> tuple[tuple[RootCriticalRoot, ...], tuple[Any, ...]]:
    primitive_factors: list[tuple[sympy.Poly, int]] = []
    request_checkpoint("during root-critical factor_list")
    for factor, multiplicity in poly.factor_list()[1]:
        primitive_factors.append((_primitive_integer_poly(factor), int(multiplicity)))
    primitive_factors.sort(key=lambda item: _factor_key(item[0]))
    # Collect every root first, so each rectangle can be refined to exclude the
    # other roots it might otherwise contain across factors.
    collected: list[tuple[sympy.Poly, int, int, Any]] = []
    for factor, multiplicity in primitive_factors:
        request_checkpoint("during root-critical all_roots")
        exact_roots = tuple(factor.all_roots())
        for root_index in range(factor.degree()):
            collected.append(
                (factor, multiplicity, root_index, _root_value(root_index, exact_roots))
            )
    real_values = tuple(
        value
        for _, _, _, root in collected
        if _root_is_real(root) and (value := _root_rational_value(root)) is not None
    )
    records: list[RootCriticalRoot] = []
    values: list[Any] = []
    for factor, multiplicity, root_index, root in collected:
        rectangle = _isolating_rectangle(root, _root_rational_value(root), real_values)
        records.append(
            RootCriticalRoot(
                axis_index=len(records),
                value=_algebraic_root_value(factor, root_index, rectangle),
                multiplicity=multiplicity,
                rectangle=rectangle,
            )
        )
        values.append(root)
    return tuple(records), tuple(values)


def _root_is_real(root: Any) -> bool:
    """Whether a backend root is real."""

    return bool(getattr(root, "is_real", False))


def _root_rational_value(root: Any) -> CanonicalRational | None:
    """Exact value of a rational root, or None when it is irrational."""

    if not getattr(root, "is_rational", False):
        return None
    return _rational(root)


def _isolating_rectangle(
    root: Any,
    own_value: CanonicalRational | None,
    real_values: tuple[CanonicalRational, ...],
) -> RootCriticalRectangle:
    """Return a rectangle refined to exclude every other real root value.

    A rational root can lie inside an irrational sibling's rectangle (for
    example a good rational approximation of ``sqrt(2)``). Refine the interval
    around this root until all colliding rational values fall outside it.
    """

    rectangle = _rectangle(root)
    if own_value is not None:
        # A rational root is its own singleton; nothing to refine.
        return rectangle
    colliding = sorted(value.as_fraction() for value in real_values)
    lower = rectangle.real_lower.as_fraction()
    upper = rectangle.real_upper.as_fraction()
    if not colliding or not any(lower <= value <= upper for value in colliding):
        return rectangle
    # Refine toward the root at increasing precision until no colliding rational
    # lies inside the interval; the isolating interval is eventually disjoint
    # from each finite set of distinct rationals.
    for digits in (60, 120, 240, 480, 960, 1920):
        refined_lower, refined_upper = _refined_real_interval(root, digits)
        candidate_lower = max(lower, refined_lower)
        candidate_upper = min(upper, refined_upper)
        if candidate_lower > candidate_upper:
            break
        lower, upper = candidate_lower, candidate_upper
        if not any(lower <= value <= upper for value in colliding):
            break
    fitted_lower = _fit_rectangle_component(lower, round_up=False)
    fitted_upper = _fit_rectangle_component(upper, round_up=True)
    if not any(
        fitted_lower.as_fraction() <= value <= fitted_upper.as_fraction()
        for value in colliding
    ):
        return RootCriticalRectangle(
            real_lower=fitted_lower,
            real_upper=fitted_upper,
            imaginary_lower=rectangle.imaginary_lower,
            imaginary_upper=rectangle.imaginary_upper,
        )
    # Fitting snapped the interval onto a carrier grid cell shared with a
    # sibling (for example a 256-digit Pell gap inside one 10**-255 cell), so
    # reintroducing the grid bounds would publish a non-isolating rectangle.
    # Separate with carrier-representable bounds instead; each tightened side
    # only shrinks toward the excluding interval, so no sibling is re-included.
    if any(lower <= value <= upper for value in colliding):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.sibling_separation",
            message=(
                "a sibling root separation needs more exact digits than the "
                "admitted root-component envelope"
            ),
        )
    below = [value for value in colliding if value < lower]
    above = [value for value in colliding if value > upper]
    real_lower = (
        _representable_separating_bound(max(below), lower) if below else fitted_lower
    )
    real_upper = (
        _representable_separating_bound(upper, min(above)) if above else fitted_upper
    )
    if (
        real_lower.as_fraction() > lower
        or real_upper.as_fraction() < upper
        or any(
            real_lower.as_fraction() <= value <= real_upper.as_fraction()
            for value in colliding
        )
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.sibling_separation",
            message=(
                "a sibling root separation needs more exact digits than the "
                "admitted root-component envelope"
            ),
        )
    return RootCriticalRectangle(
        real_lower=real_lower,
        real_upper=real_upper,
        imaginary_lower=rectangle.imaginary_lower,
        imaginary_upper=rectangle.imaginary_upper,
    )


def _refined_real_interval(root: Any, digits: int) -> tuple[Fraction, Fraction]:
    """A narrow certified real interval for a real algebraic root."""

    try:
        real_lo, real_hi, _imag_lo, _imag_hi = _enclose_sympy(root)
    except (ValueError, AttributeError, TypeError):
        real_lo, real_hi, _imag_lo, _imag_hi = _evalf_containing_box(root)
    value = root.evalf(digits)
    real = value.as_real_imag()[0]
    centre = Fraction(int(sympy.Rational(real).p), int(sympy.Rational(real).q))
    # A ``digits``-significant-digit approximation has absolute error at most
    # ``|value| * 10**(1 - digits)``; combine it with the certified interval.
    magnitude = abs(centre)
    guard = magnitude / 10 ** (digits - 1) + Fraction(1, 10**digits)
    return max(real_lo, centre - guard), min(real_hi, centre + guard)


def _enclose_add(expr: Any) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    real_lo = real_hi = imag_lo = imag_hi = Fraction()
    first = True
    for argument in expr.args:
        r0, r1, i0, i1 = _enclose_sympy(argument)
        if first:
            real_lo, real_hi, imag_lo, imag_hi = r0, r1, i0, i1
            first = False
        else:
            real_lo += r0
            real_hi += r1
            imag_lo += i0
            imag_hi += i1
    return real_lo, real_hi, imag_lo, imag_hi


def _enclose_mul(expr: Any) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    result = (Fraction(1), Fraction(1), Fraction(), Fraction())
    for argument in expr.args:
        result = _multiply_boxes(result, _enclose_sympy(argument))
    return result


def _integer_nth_root_bounds(value: int, n: int) -> tuple[int, int]:
    if value < 0:
        raise ValueError("certified real radical requires a nonnegative radicand")
    if value <= 1:
        return value, value
    high = 1 << ((value.bit_length() + n - 1) // n)
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        power = mid**n
        if power == value:
            return mid, mid
        if power < value:
            low = mid
        else:
            high = mid - 1
    return low, low + 1


def _nth_root_bounds(value: Fraction, n: int) -> tuple[Fraction, Fraction]:
    if value < 0:
        if n % 2 == 0:
            raise ValueError("certified even root requires a nonnegative radicand")
        lower, upper = _nth_root_bounds(-value, n)
        return -upper, -lower
    if value == 0:
        return Fraction(), Fraction()
    scaled = value.numerator * value.denominator ** (n - 1)
    lower_int, upper_int = _integer_nth_root_bounds(scaled, n)
    denominator = value.denominator
    coarse_lower = Fraction(lower_int, denominator)
    coarse_upper = Fraction(upper_int, denominator)
    if coarse_lower == coarse_upper:
        return coarse_lower, coarse_upper
    # Refine on a dyadic grid sized to the radicand's own magnitude. A fixed
    # absolute grid is not usable here: ``sqrt(A**2 - 4)`` with ``A = 10**40``
    # would be published with width ``2**-32``, which is wider than the gap
    # between the two neighbouring roots, so the corresponding isolating
    # rectangle would also contain the root at the origin.
    magnitude_bits = (value.numerator.bit_length() + n - 1) // n
    bits = max(32, magnitude_bits + _NTH_ROOT_RELATIVE_BITS)
    scale = 1 << bits
    scaled_radicand = value.numerator << (bits * n)
    low = 0
    high = int(coarse_upper * scale) + 1
    while low < high:
        mid = (low + high + 1) // 2
        if mid**n * value.denominator <= scaled_radicand:
            low = mid
        else:
            high = mid - 1
    lower = Fraction(low, scale)
    if lower**n == value:
        return lower, lower
    return lower, Fraction(low + 1, scale)


def _integer_power_box(
    factor: tuple[Fraction, Fraction, Fraction, Fraction], power: int
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if power < 0:
        raise ValueError("certified enclosure does not invert")
    result = (Fraction(1), Fraction(1), Fraction(), Fraction())
    remaining = power
    while remaining:
        if remaining & 1:
            result = _multiply_boxes(result, factor)
        remaining >>= 1
        if remaining:
            factor = _multiply_boxes(factor, factor)
    return result


def _enclose_rational_power(
    base: Any, exponent: Any
) -> tuple[Fraction, Fraction, Fraction, Fraction] | None:
    if not (getattr(exponent, "is_Rational", False) or isinstance(exponent, Fraction)):
        return None
    numerator = int(exponent.p if hasattr(exponent, "p") else exponent.numerator)
    denominator = int(exponent.q if hasattr(exponent, "q") else exponent.denominator)
    if denominator <= 0 or numerator < 0:
        return None
    r0, r1, i0, i1 = _enclose_sympy(base)
    if i0 != 0 or i1 != 0:
        if denominator == 2 and numerator == 1:
            return _enclose_complex_square_root(r0, r1, i0, i1)
        raise ValueError("certified radical requires a real box")
    if denominator > 1:
        if r0 < 0 and denominator % 2 == 0:
            raise ValueError("certified even root requires a nonnegative real box")
        r0, r1 = (
            _nth_root_bounds(r0, denominator)[0],
            _nth_root_bounds(r1, denominator)[1],
        )
    if numerator == 0:
        return Fraction(1), Fraction(1), Fraction(), Fraction()
    return _integer_power_box((r0, r1, Fraction(), Fraction()), numerator)


def _enclose_pow(expr: Any) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    base, exponent = expr.args
    if exponent == 2:
        r0, r1, i0, i1 = _enclose_sympy(base)
        return _square_box(r0, r1, i0, i1)
    rational_box = _enclose_rational_power(base, exponent)
    if rational_box is not None:
        return rational_box
    if getattr(exponent, "is_Integer", False):
        return _integer_power_box(_enclose_sympy(base), int(exponent))
    raise ValueError("expression is outside the certified enclosure grammar")


def _enclose_complex_square_root(
    r0: Fraction, r1: Fraction, i0: Fraction, i1: Fraction
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Certify the principal square root of a complex box.

    For ``z = r + i s`` the principal root has real part
    ``sqrt((|z| + r) / 2)`` and imaginary part ``sign(s) sqrt((|z| - r) / 2)``.
    The box must stay off the branch cut (not straddle the negative real axis),
    which the caller checks. Each inner term is bounded over the box corners.
    """

    if i0 <= 0 <= i1:
        raise ValueError("complex square root requires a definite imaginary sign")
    negative_real_axis = r1 < 0 and i0 <= 0 <= i1
    if negative_real_axis:
        raise ValueError("complex square root crosses the branch cut")
    sign = Fraction(1) if i0 > 0 else Fraction(-1)
    # |z|^2 = r^2 + s^2 over the box corners.
    squared_corners = tuple(
        real * real + imaginary * imaginary
        for real in (r0, r1)
        for imaginary in (i0, i1)
    )
    squared_lo = min(squared_corners)
    squared_hi = max(squared_corners)
    modulus_lo = _nth_root_bounds(max(squared_lo, Fraction()), 2)[0]
    modulus_hi = _nth_root_bounds(squared_hi, 2)[1]
    # Real part: sqrt((|z| + r) / 2).
    real_inner_lo = (modulus_lo + r0) / 2
    real_inner_hi = (modulus_hi + r1) / 2
    real_lo = _nth_root_bounds(max(real_inner_lo, Fraction()), 2)[0]
    real_hi = _nth_root_bounds(max(real_inner_hi, Fraction()), 2)[1]
    # Imaginary magnitude: sqrt((|z| - r) / 2) >= 0.
    imaginary_inner_lo = max(modulus_lo - r1, Fraction()) / 2
    imaginary_inner_hi = max(modulus_hi - r0, Fraction()) / 2
    imaginary_lo = _nth_root_bounds(imaginary_inner_lo, 2)[0]
    imaginary_hi = _nth_root_bounds(imaginary_inner_hi, 2)[1]
    if sign < 0:
        return real_lo, real_hi, -imaginary_hi, -imaginary_lo
    return real_lo, real_hi, imaginary_lo, imaginary_hi


def _enclose_sympy(expr: Any) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return a certified complex box (real_lo, real_hi, imag_lo, imag_hi)."""

    if isinstance(expr, int):
        value = Fraction(expr)
        return value, value, Fraction(), Fraction()
    if isinstance(expr, Fraction):
        return expr, expr, Fraction(), Fraction()
    if getattr(expr, "is_Rational", False) and expr.is_real:
        value = Fraction(int(expr.p), int(expr.q))
        return value, value, Fraction(), Fraction()
    if expr == sympy.I:
        return Fraction(), Fraction(), Fraction(1), Fraction(1)
    if expr.is_Add:
        return _enclose_add(expr)
    if expr.is_Mul:
        return _enclose_mul(expr)
    if expr.is_Pow:
        return _enclose_pow(expr)
    if expr.func is sympy.conjugate:
        r0, r1, i0, i1 = _enclose_sympy(expr.args[0])
        return r0, r1, -i1, -i0
    if expr.func is sympy.re:
        r0, r1, _i0, _i1 = _enclose_sympy(expr.args[0])
        return r0, r1, Fraction(), Fraction()
    if expr.func is sympy.im:
        _r0, _r1, i0, i1 = _enclose_sympy(expr.args[0])
        return i0, i1, Fraction(), Fraction()
    raise ValueError("expression is outside the certified enclosure grammar")


def _square_box(
    r0: Fraction, r1: Fraction, i0: Fraction, i1: Fraction
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return _multiply_boxes((r0, r1, i0, i1), (r0, r1, i0, i1))


def _multiply_boxes(
    left: tuple[Fraction, Fraction, Fraction, Fraction],
    right: tuple[Fraction, Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    real_lo, real_hi, imag_lo, imag_hi = left
    r0, r1, i0, i1 = right
    corners_real = []
    corners_imag = []
    for left_r in (real_lo, real_hi):
        for left_i in (imag_lo, imag_hi):
            for right_r in (r0, r1):
                for right_i in (i0, i1):
                    corners_real.append(left_r * right_r - left_i * right_i)
                    corners_imag.append(left_r * right_i + left_i * right_r)
    return min(corners_real), max(corners_real), min(corners_imag), max(corners_imag)


def _select_real_algebraic_root(minimal: sympy.Poly, distance: Any) -> int:
    """Select the real minpoly root by certified interval comparison."""

    simplified = sympy.simplify(distance)
    if simplified == 0:
        roots = minimal.real_roots()
        for index, candidate in enumerate(roots):
            if candidate == 0:
                return index
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_root_selection",
            message="exact real distance root could not be selected",
        )
    try:
        real_lo, real_hi, imag_lo, imag_hi = _enclose_sympy(simplified)
    except (ValueError, AttributeError, TypeError) as exc:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_root_selection",
            message="exact real distance root could not be selected",
        ) from exc
    if imag_lo > 0 or imag_hi < 0:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_root_selection",
            message="exact real distance root could not be selected",
        )
    hits = _unique_interval_hit(list(minimal.intervals()), real_lo, real_hi)
    if hits is not None:
        return hits
    eps = Fraction(1, 1 << 20)
    for _ in range(8):
        hits = _unique_interval_hit(list(minimal.intervals(eps=eps)), real_lo, real_hi)
        if hits is not None:
            return hits
        eps /= 4
    raise OperationDomainValidationError(
        location=("pairs",),
        code="polynomial.root_critical.distance_root_selection",
        message="exact real distance root could not be selected",
    )


def _unique_interval_hit(
    intervals: list[Any], real_lo: Fraction, real_hi: Fraction
) -> int | None:
    """Return the unique isolating interval that meets a certified enclosure."""

    hits = [
        index
        for index, ((lower, upper), _multiplicity) in enumerate(intervals)
        if not (real_hi < Fraction(lower) or real_lo > Fraction(upper))
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _distance_value(
    left: Any,
    right: Any,
) -> tuple[
    RealAlgebraicValue,
    RationalIsolatingInterval,
    Literal["POSITIVE", "ZERO_DISTANCE"],
]:
    variable = sympy.Symbol("distance")
    request_checkpoint("during root-critical distance minpoly")
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
    if any(
        len(format_canonical_integer(abs(coefficient)))
        > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
        for coefficient in coefficients
    ):
        raise OperationResourceAdmissionError(
            location=("pairs",),
            code="polynomial.root_critical.distance_coefficient_bound",
            message=(
                "squared-distance minimal polynomial coefficients exceed the "
                f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit real-algebraic carrier"
            ),
        )
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


def _conjugate_field_multiplier(factor: sympy.Poly) -> int:
    degree = int(factor.degree())
    if degree <= 1:
        return 1
    real_count = int(factor.count_roots())
    if real_count == degree:
        return 1
    return 2


def _admit_cleared_coefficient_heights(candidates: tuple[Any, ...]) -> None:
    """Reject over-height cleared primitive polynomials before factorization."""

    for candidate in candidates:
        if candidate.degree() <= 0:
            continue
        primitive = _primitive_integer_poly(candidate)
        digits = max(
            (
                len(format_canonical_integer(abs(int(coefficient))))
                for coefficient in primitive.all_coeffs()
            ),
            default=1,
        )
        if digits > MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("polynomial",),
                code="polynomial.root_critical.factor_coefficient_bound",
                message=(
                    "the cleared primitive source or derivative exceeds the "
                    "admitted exact root-component digit envelope"
                ),
            )


def _admit(
    polynomial: object,
    *,
    max_pair_rows: object,
) -> tuple[sympy.Poly, int, int]:
    if type(max_pair_rows) is not int or isinstance(max_pair_rows, bool):
        raise OperationDomainValidationError(
            location=("max_pair_rows",),
            code="polynomial.root_critical.pair_budget_type",
            message="max_pair_rows must be a non-boolean integer",
        )
    if max_pair_rows < 0 or max_pair_rows > MAX_ROOT_CRITICAL_PAIRS:
        raise OperationDomainValidationError(
            location=("max_pair_rows",),
            code="polynomial.root_critical.pair_budget_range",
            message="max_pair_rows must lie in the admitted 0..64 row budget",
        )
    if not isinstance(polynomial, RationalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.root_critical.polynomial_type",
            message="root-critical profiles require a canonical rational polynomial",
        )
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
    # Bound the cleared primitive source and derivative before factorization:
    # clearing denominators can grow the coefficients far past the input
    # component height, and factor_list is the expensive phase.
    _admit_cleared_coefficient_heights((source, source.diff()))
    request_checkpoint("during root-critical admission factor_list")
    source_factors = source.factor_list()[1]
    derivative_backend = source.diff()
    derivative_factors = derivative_backend.factor_list()[1] if critical_degree else ()
    for factor, _ in (*source_factors, *derivative_factors):
        primitive = _primitive_integer_poly(factor)
        digits = max(
            (
                len(format_canonical_integer(abs(int(coefficient))))
                for coefficient in primitive.all_coeffs()
            ),
            default=1,
        )
        if digits > MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("polynomial",),
                code="polynomial.root_critical.factor_coefficient_bound",
                message=(
                    "a primitive source or derivative factor exceeds the "
                    "admitted exact root-component digit envelope"
                ),
            )
    if any(factor.degree() > 4 for factor, _ in (*source_factors, *derivative_factors)):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.root_carrier_bound",
            message="the exact-root carrier admits irreducible factors through degree four",
        )
    root_count = sum(int(factor.degree()) for factor, _ in source_factors)
    critical_count = (
        sum(int(factor.degree()) for factor, _ in derivative_factors)
        if critical_degree
        else 0
    )
    pair_count = root_count * critical_count
    if pair_count > MAX_ROOT_CRITICAL_PAIRS or pair_count > max_pair_rows:
        raise OperationResourceAdmissionError(
            location=("max_pair_rows",),
            code="polynomial.root_critical.pair_output_bound",
            message="complete root-critical pair expansion exceeds the admitted row budget",
        )
    max_distance_degree = max(
        (
            _conjugate_field_multiplier(source_factor)
            * source_factor.degree()
            * _conjugate_field_multiplier(critical_factor)
            * critical_factor.degree()
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
    request_checkpoint("during root-critical admission all_roots")
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
    return source, root_count, critical_count


def _compute_profile(
    polynomial: RationalPolynomial,
    *,
    max_pair_rows: object,
) -> RootCriticalDistanceProfile:
    """Compute the exact profile in-process after admission."""

    request_checkpoint("before root-critical admission")
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
    request_checkpoint("after root-critical result construction")
    return result


def _run_profile_worker(
    polynomial: RationalPolynomial,
    *,
    max_pair_rows: object,
    deadline: float,
    cancellation_signal: RequestCancellationSignal | None,
) -> RootCriticalDistanceProfile:
    """Run the blocking kernel in a killable child and revalidate its profile."""

    remaining = deadline - time.monotonic() - _PROFILE_FINALIZATION_SECONDS
    if remaining <= 0:
        from jacobian._execution import OperationExecutionTimeoutError

        raise OperationExecutionTimeoutError(
            "root-critical profile deadline expired before the kernel worker"
        )
    worker_stdout = run_profile_worker_process(
        polynomial,
        max_pair_rows=max_pair_rows,
        remaining_seconds=remaining,
        cancellation_signal=cancellation_signal,
    )
    response = loads_strict_json(
        worker_stdout,
        limits=CanonicalLimits(
            max_input_bytes=PROFILE_STDOUT_BYTES,
            max_output_bytes=PROFILE_STDOUT_BYTES,
        ),
    )
    if not isinstance(response, dict):
        raise RuntimeError(
            "bounded root-critical kernel worker returned malformed output"
        )
    if response.get("ok") is False:
        return _raise_worker_error(response)
    if response.get("ok") is not True or not isinstance(response.get("profile"), str):
        raise RuntimeError(
            "bounded root-critical kernel worker returned malformed output"
        )
    request_checkpoint("after root-critical kernel worker")
    return RootCriticalDistanceProfile.model_validate_json(response["profile"])


def _raise_worker_error(response: dict[str, object]) -> NoReturn:
    """Re-raise the typed admission or domain error the kernel worker reported."""

    location = response.get("location")
    code = response.get("code")
    message = response.get("message")
    kind = response.get("kind")
    if (
        not isinstance(location, list)
        or not isinstance(code, str)
        or not isinstance(message, str)
        or kind not in {"domain", "resource"}
    ):
        raise RuntimeError(
            "bounded root-critical kernel worker returned malformed diagnostics"
        )
    error_type = (
        OperationResourceAdmissionError
        if kind == "resource"
        else OperationDomainValidationError
    )
    raise error_type(location=tuple(location), code=code, message=message)


def root_critical_distance_profile(
    polynomial: RationalPolynomial,
    *,
    max_pair_rows: int = MAX_ROOT_CRITICAL_PAIRS,
) -> RootCriticalDistanceProfile:
    """Return every distinct root/critical pair and its exact squared distance.

    The blocking SymPy phases (``factor_list``, ``all_roots``, and ``minpoly``)
    run inside a killable child process so the request deadline and cancellation
    can stop them; the returned profile is re-validated before it is trusted.
    """

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return root_critical_distance_profile(
                polynomial, max_pair_rows=max_pair_rows
            )
    if not isinstance(polynomial, RationalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.root_critical.polynomial_type",
            message="root-critical profile requires a rational polynomial",
        )
    if type(max_pair_rows) is not int or isinstance(max_pair_rows, bool):
        raise OperationDomainValidationError(
            location=("max_pair_rows",),
            code="polynomial.root_critical.pair_row_type",
            message="max_pair_rows must be a non-boolean integer",
        )
    if max_pair_rows < 0 or max_pair_rows > MAX_ROOT_CRITICAL_PAIRS:
        raise OperationDomainValidationError(
            location=("max_pair_rows",),
            code="polynomial.root_critical.pair_row_range",
            message="max_pair_rows must lie in the admitted 0..64 row budget",
        )
    deadline = execution.started_at + ROOT_CRITICAL_WALL_SECONDS
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    return _run_profile_worker(
        polynomial,
        max_pair_rows=max_pair_rows,
        deadline=deadline,
        cancellation_signal=execution.cancellation_signal,
    )
