"""SymPy-backed exact root--critical-point profile computation."""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any, Literal, NoReturn

import sympy

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    RequestCancellationSignal,
    RequestExecutionEnvelope,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RationalIsolatingInterval,
    RealAlgebraicValue,
    require_primitive_real_algebraic_value,
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
    ExactSplittingField,
    RootCriticalDistanceProfile,
    RootCriticalDistanceRow,
    RootCriticalRectangle,
    RootCriticalRoot,
    SplittingFieldDistanceProfile,
    SplittingFieldKernelPayload,
)
from jacobian.math.polynomials.root_critical._profile_process import (
    PROFILE_STDOUT_BYTES,
    run_profile_worker_process,
)
from jacobian.math.polynomials.root_critical._splitting_process import (
    SPLITTING_STDOUT_BYTES,
    run_splitting_worker_process,
)
from jacobian.math.polynomials.values import RationalPolynomial

_PROFILE_FINALIZATION_SECONDS = 1.0

_NTH_ROOT_RELATIVE_BITS = 96

__all__ = [
    "exact_splitting_field",
    "root_critical_distance_profile",
    "splitting_field_distance_profile",
]

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
    return Fraction(floor_low * inner.numerator + inner.denominator, inner.numerator)


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
    # CRootOf is retained as a source identity.  It must not be handed to
    # ``minpoly`` together with another independently indexed CRootOf (SymPy
    # cannot reliably reduce that product).  The bounded distance kernel below
    # works from the defining factor and a rational critical point instead.
    return roots[root_index]


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


def _squarefree_support(
    primitive_factors: list[tuple[sympy.Poly, int]],
) -> sympy.Poly:
    """The squarefree product of every distinct source or derivative factor."""

    variable = primitive_factors[0][0].gens[0]
    expression = sympy.Integer(1)
    for factor, _multiplicity in primitive_factors:
        expression *= factor.as_expr()
    return sympy.Poly(expression, variable, domain=sympy.ZZ)


def _family(
    poly: sympy.Poly,
) -> tuple[tuple[RootCriticalRoot, ...], tuple[Any, ...]]:
    primitive_factors: list[tuple[sympy.Poly, int]] = []
    request_checkpoint("during root-critical factor_list")
    for factor, multiplicity in poly.factor_list()[1]:
        primitive_factors.append((_primitive_integer_poly(factor), int(multiplicity)))
    primitive_factors.sort(key=lambda item: _factor_key(item[0]))
    support = _squarefree_support(primitive_factors)
    # Collect every root first, so each rectangle can be checked against the
    # complete isolating family of the squarefree support.
    collected: list[tuple[sympy.Poly, int, int, Any]] = []
    for factor, multiplicity in primitive_factors:
        request_checkpoint("during root-critical all_roots")
        exact_roots = tuple(factor.all_roots())
        for root_index in range(factor.degree()):
            collected.append(
                (factor, multiplicity, root_index, _root_value(root_index, exact_roots))
            )
    for _, _, _, root in collected:
        if getattr(root, "is_real", None) is None:
            raise OperationDomainValidationError(
                location=("polynomial",),
                code="polynomial.root_critical.root_reality_undecidable",
                message=(
                    "an admitted source or derivative root has no decidable "
                    "real/complex classification"
                ),
            )
    all_roots = tuple(root for _, _, _, root in collected)
    real_roots = tuple(root for root in all_roots if _root_is_real(root))
    records: list[RootCriticalRoot] = []
    values: list[Any] = []
    for factor, multiplicity, root_index, root in collected:
        rectangle = _isolating_rectangle(
            root, _root_rational_value(root), real_roots, all_roots, support
        )
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


def _intervals_meet(
    left_lower: Fraction,
    left_upper: Fraction,
    right_lower: Fraction,
    right_upper: Fraction,
) -> bool:
    """Whether two closed rational intervals intersect."""

    return left_lower <= right_upper and right_lower <= left_upper


def _sibling_real_intervals(
    real_roots: tuple[Any, ...], root: Any, digits: int
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Certified real intervals for every real sibling of ``root``.

    A rational sibling is its exact singleton; an irrational sibling is bounded
    by a certified interval narrow enough to compare against ``root`` at the
    requested precision.
    """

    intervals: list[tuple[Fraction, Fraction]] = []
    for sibling in real_roots:
        if sibling is root:
            continue
        value = _root_rational_value(sibling)
        if value is not None:
            rational = value.as_fraction()
            intervals.append((rational, rational))
        else:
            intervals.append(_refined_real_interval(sibling, digits))
    return tuple(intervals)


def _boxes_meet(
    left: tuple[Fraction, Fraction, Fraction, Fraction],
    right: tuple[Fraction, Fraction, Fraction, Fraction],
) -> bool:
    """Whether two closed real/imaginary boxes intersect."""

    return (
        left[0] <= right[1]
        and right[0] <= left[1]
        and left[2] <= right[3]
        and right[2] <= left[3]
    )


def _sibling_boxes(
    all_roots: tuple[Any, ...], root: Any, digits: int
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]:
    """Certified boxes for every sibling root of ``root`` at ``digits``."""

    boxes: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
    for sibling in all_roots:
        if sibling is root:
            continue
        value = _root_rational_value(sibling)
        if value is not None:
            rational = value.as_fraction()
            boxes.append((rational, rational, Fraction(0), Fraction(0)))
        elif _root_is_real(sibling):
            lower, upper = _refined_real_interval(sibling, digits)
            boxes.append((lower, upper, Fraction(0), Fraction(0)))
        else:
            boxes.append(_refined_complex_box(sibling, digits))
    return tuple(boxes)


def _isolating_rectangle(
    root: Any,
    own_value: CanonicalRational | None,
    real_roots: tuple[Any, ...],
    all_roots: tuple[Any, ...],
    support: sympy.Poly,
) -> RootCriticalRectangle:
    """Return a rectangle certified to contain exactly one source-axis root.

    A rational root is its own exact singleton.  An irrational real root is
    refined against the complete real sibling family, and a non-real root is
    refined against every sibling box.  The result is checked to establish
    exactly one support root, so a non-isolating rectangle is refused rather
    than published.
    """

    rectangle = _rectangle(root)
    if own_value is not None:
        # A rational root is its own singleton; nothing to refine.
        return rectangle
    if not _root_is_real(root):
        return _complex_isolating_rectangle(root, all_roots)
    rectangle = _real_isolating_rectangle(root, real_roots)
    if (
        strict_root_count(
            support,
            rectangle.real_lower.as_fraction(),
            rectangle.real_upper.as_fraction(),
        )
        != 1
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.sibling_separation",
            message=(
                "the refined real rectangle does not isolate one support root "
                "within the admitted root-component envelope"
            ),
        )
    _require_box_separated_from_nonreal(
        (
            rectangle.real_lower.as_fraction(),
            rectangle.real_upper.as_fraction(),
            rectangle.imaginary_lower.as_fraction(),
            rectangle.imaginary_upper.as_fraction(),
        ),
        root,
        all_roots,
    )
    return rectangle


def _real_isolating_rectangle(
    root: Any,
    real_roots: tuple[Any, ...],
) -> RootCriticalRectangle:
    """Refine a real root's interval against every real sibling's enclosure."""

    rectangle = _rectangle(root)
    lower = rectangle.real_lower.as_fraction()
    upper = rectangle.real_upper.as_fraction()
    colliding = _sibling_real_intervals(real_roots, root, 60)
    if not any(
        _intervals_meet(sibling_lower, sibling_upper, lower, upper)
        for sibling_lower, sibling_upper in colliding
    ):
        return rectangle
    # Refine toward the root at increasing precision until no real sibling
    # interval meets this interval; distinct real roots are eventually
    # separated from each finite set of certified enclosures.
    for digits in (60, 120, 240, 480, 960, 1920):
        refined_lower, refined_upper = _refined_real_interval(root, digits)
        candidate_lower = max(lower, refined_lower)
        candidate_upper = min(upper, refined_upper)
        if candidate_lower > candidate_upper:
            break
        lower, upper = candidate_lower, candidate_upper
        colliding = _sibling_real_intervals(real_roots, root, digits)
        if not any(
            _intervals_meet(sibling_lower, sibling_upper, lower, upper)
            for sibling_lower, sibling_upper in colliding
        ):
            break
    fitted_lower = _fit_rectangle_component(lower, round_up=False)
    fitted_upper = _fit_rectangle_component(upper, round_up=True)
    if not any(
        _intervals_meet(
            sibling_lower,
            sibling_upper,
            fitted_lower.as_fraction(),
            fitted_upper.as_fraction(),
        )
        for sibling_lower, sibling_upper in colliding
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
    if any(
        _intervals_meet(sibling_lower, sibling_upper, lower, upper)
        for sibling_lower, sibling_upper in colliding
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.sibling_separation",
            message=(
                "a sibling root separation needs more exact digits than the "
                "admitted root-component envelope"
            ),
        )
    # Only the siblings the fitted grid re-included need a separating bound;
    # far-away siblings (such as the reflected root ``-sqrt(2)``) would pull the
    # simplest rational bound far away from the root.
    separating = [
        (sibling_lower, sibling_upper)
        for sibling_lower, sibling_upper in colliding
        if _intervals_meet(
            sibling_lower,
            sibling_upper,
            fitted_lower.as_fraction(),
            fitted_upper.as_fraction(),
        )
    ]
    below = [
        sibling_upper
        for sibling_lower, sibling_upper in separating
        if sibling_upper < lower
    ]
    above = [
        sibling_lower
        for sibling_lower, sibling_upper in separating
        if sibling_lower > upper
    ]
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
            _intervals_meet(
                sibling_lower,
                sibling_upper,
                real_lower.as_fraction(),
                real_upper.as_fraction(),
            )
            for sibling_lower, sibling_upper in colliding
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


def _box_within_envelope(box: tuple[Fraction, Fraction, Fraction, Fraction]) -> bool:
    """Whether every published component stays inside the carrier digit bound."""

    return all(
        _component_digit_count(value.numerator)
        <= MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
        and _component_digit_count(value.denominator)
        <= MAX_ROOT_CRITICAL_ROOT_COMPONENT_DIGITS
        for value in box
    )


def _rectangle_from_box(
    box: tuple[Fraction, Fraction, Fraction, Fraction],
) -> RootCriticalRectangle:
    return RootCriticalRectangle(
        real_lower=CanonicalRational.from_fraction(box[0]),
        real_upper=CanonicalRational.from_fraction(box[1]),
        imaginary_lower=CanonicalRational.from_fraction(box[2]),
        imaginary_upper=CanonicalRational.from_fraction(box[3]),
    )


def _refined_complex_box(
    root: Any, digits: int
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """A narrow certified complex box for a non-real algebraic root."""

    try:
        real_lo, real_hi, imag_lo, imag_hi = _enclose_sympy(root)
    except (ValueError, AttributeError, TypeError):
        real_lo, real_hi, imag_lo, imag_hi = _evalf_containing_box(root)
    value = root.evalf(digits)
    real, imag = value.as_real_imag()
    centre_real = Fraction(int(sympy.Rational(real).p), int(sympy.Rational(real).q))
    centre_imag = Fraction(int(sympy.Rational(imag).p), int(sympy.Rational(imag).q))
    magnitude = max(abs(centre_real), abs(centre_imag))
    guard = magnitude / 10 ** (digits - 1) + Fraction(1, 10**digits)
    return (
        max(real_lo, centre_real - guard),
        min(real_hi, centre_real + guard),
        max(imag_lo, centre_imag - guard),
        min(imag_hi, centre_imag + guard),
    )


def _require_box_separated_from_nonreal(
    box: tuple[Fraction, Fraction, Fraction, Fraction],
    root: Any,
    all_roots: tuple[Any, ...],
) -> None:
    """Refuse a real rectangle that meets any non-real sibling enclosure."""

    for sibling in all_roots:
        if sibling is root or _root_is_real(sibling):
            continue
        separated = False
        for digits in (60, 120, 240, 480, 960, 1920):
            if not _boxes_meet(box, _refined_complex_box(sibling, digits)):
                separated = True
                break
        if not separated:
            raise OperationResourceAdmissionError(
                location=("polynomial",),
                code="polynomial.root_critical.sibling_separation",
                message=(
                    "a real rectangle could not be separated from a non-real "
                    "sibling root within the admitted root-component envelope"
                ),
            )


def _complex_isolating_rectangle(
    root: Any,
    all_roots: tuple[Any, ...],
) -> RootCriticalRectangle:
    """Refine a non-real root's box until it excludes every sibling root."""

    for digits in (60, 120, 240, 480, 960, 1920):
        box = _refined_complex_box(root, digits)
        siblings = _sibling_boxes(all_roots, root, digits)
        if any(_boxes_meet(box, sibling) for sibling in siblings):
            continue
        if _box_within_envelope(box):
            return _rectangle_from_box(box)
        fitted = (
            _fit_rectangle_component(box[0], round_up=False),
            _fit_rectangle_component(box[1], round_up=True),
            _fit_rectangle_component(box[2], round_up=False),
            _fit_rectangle_component(box[3], round_up=True),
        )
        fitted_box = (
            fitted[0].as_fraction(),
            fitted[1].as_fraction(),
            fitted[2].as_fraction(),
            fitted[3].as_fraction(),
        )
        if not any(_boxes_meet(fitted_box, sibling) for sibling in siblings):
            return _rectangle_from_box(fitted_box)
    raise OperationResourceAdmissionError(
        location=("polynomial",),
        code="polynomial.root_critical.sibling_separation",
        message=(
            "a non-real root separation needs more exact digits than the "
            "admitted root-component envelope"
        ),
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


def _distance_box_for_rational_critical(
    rectangle: RootCriticalRectangle,
    critical: CanonicalRational,
) -> tuple[Fraction, Fraction]:
    """Bound ``|(x+iy)-critical|²`` over a certified root rectangle."""

    real_lower = rectangle.real_lower.as_fraction()
    real_upper = rectangle.real_upper.as_fraction()
    imag_lower = rectangle.imaginary_lower.as_fraction()
    imag_upper = rectangle.imaginary_upper.as_fraction()
    c = critical.as_fraction()
    if real_lower <= c <= real_upper:
        real_min = Fraction()
    else:
        real_min = min((real_lower - c) ** 2, (real_upper - c) ** 2)
    real_max = max((real_lower - c) ** 2, (real_upper - c) ** 2)
    if imag_lower <= 0 <= imag_upper:
        imag_min = Fraction()
    else:
        imag_min = min(imag_lower**2, imag_upper**2)
    imag_max = max(imag_lower**2, imag_upper**2)
    return real_min + imag_min, real_max + imag_max


def _distance_box_for_rectangles(
    root: RootCriticalRectangle,
    critical: RootCriticalRectangle,
) -> tuple[Fraction, Fraction]:
    """Bound squared distance using two certified complex rectangles."""

    real_lower = root.real_lower.as_fraction() - critical.real_upper.as_fraction()
    real_upper = root.real_upper.as_fraction() - critical.real_lower.as_fraction()
    imag_lower = (
        root.imaginary_lower.as_fraction() - critical.imaginary_upper.as_fraction()
    )
    imag_upper = (
        root.imaginary_upper.as_fraction() - critical.imaginary_lower.as_fraction()
    )
    real_min = (
        Fraction()
        if real_lower <= 0 <= real_upper
        else min(real_lower**2, real_upper**2)
    )
    imag_min = (
        Fraction()
        if imag_lower <= 0 <= imag_upper
        else min(imag_lower**2, imag_upper**2)
    )
    return (
        real_min + imag_min,
        max(real_lower**2, real_upper**2) + max(imag_lower**2, imag_upper**2),
    )


def _distance_interval_relation(
    lower: Fraction,
    upper: Fraction,
    enclosure_lower: Fraction,
    enclosure_upper: Fraction,
) -> Literal["CONTAINED", "DISJOINT", "AMBIGUOUS"]:
    """Classify an isolating interval against a closed exact enclosure.

    Touching at an endpoint is deliberately ambiguous until refinement proves
    whether the algebraic root lies on that endpoint.  A singleton endpoint
    is exact and can therefore be classified immediately.
    """

    if lower == upper:
        return (
            "CONTAINED" if enclosure_lower <= lower <= enclosure_upper else "DISJOINT"
        )
    if lower >= enclosure_lower and upper <= enclosure_upper:
        return "CONTAINED"
    if upper < enclosure_lower or lower > enclosure_upper:
        return "DISJOINT"
    return "AMBIGUOUS"


def _select_eliminated_distance_root(
    factors: list[sympy.Poly],
    enclosure_lower: Fraction,
    enclosure_upper: Fraction,
) -> tuple[sympy.Poly, int, Fraction, Fraction]:
    """Refine factor root intervals until exactly one is in the enclosure."""

    # The unrefined intervals are useful for cheap disjointness, then the
    # progressively tighter rational isolators settle endpoint contacts.
    epsilons: tuple[Fraction | None, ...] = (
        None,
        Fraction(1, 1 << 20),
        Fraction(1, 1 << 40),
        Fraction(1, 1 << 80),
        Fraction(1, 1 << 160),
        Fraction(1, 1 << 320),
        Fraction(1, 1 << 640),
        Fraction(1, 1 << 1280),
    )
    for epsilon in epsilons:
        contained: list[tuple[sympy.Poly, int, Fraction, Fraction]] = []
        ambiguous = False
        for factor in factors:
            intervals = (
                factor.intervals() if epsilon is None else factor.intervals(eps=epsilon)
            )
            for index, ((lower, upper), _multiplicity) in enumerate(intervals):
                relation = _distance_interval_relation(
                    Fraction(lower),
                    Fraction(upper),
                    enclosure_lower,
                    enclosure_upper,
                )
                if relation == "CONTAINED":
                    contained.append((factor, index, Fraction(lower), Fraction(upper)))
                elif relation == "AMBIGUOUS":
                    ambiguous = True
        if len(contained) > 1:
            raise OperationDomainValidationError(
                location=("pairs",),
                code="polynomial.root_critical.distance_root_selection",
                message="certified distance enclosure contains multiple algebraic roots",
            )
        if len(contained) == 1 and not ambiguous:
            return contained[0]
        if not ambiguous:
            break
    raise OperationDomainValidationError(
        location=("pairs",),
        code="polynomial.root_critical.distance_root_selection",
        message="certified distance enclosure did not isolate one algebraic root",
    )


def _distance_value_from_elimination(
    elimination: sympy.Poly,
    rectangle: RootCriticalRectangle,
    critical: CanonicalRational,
) -> tuple[
    RealAlgebraicValue,
    RationalIsolatingInterval,
    Literal["POSITIVE", "ZERO_DISTANCE"],
]:
    """Factor an exact elimination polynomial and select its geometric root.

    The source root remains identified by its certified rectangle.  Thus this
    routine never treats an opaque backend root index as a distance witness:
    the witness is a primitive irreducible factor and one of its isolated real
    roots selected by the rectangle's exact distance enclosure.
    """

    request_checkpoint("during root-critical distance factorization")
    factors = sympy.factor_list(elimination)[1]
    real_lower, real_upper = _distance_box_for_rational_critical(rectangle, critical)
    factors_for_selection: list[sympy.Poly] = []
    for factor, _multiplicity in factors:
        primitive = _primitive_integer_poly(sympy.Poly(factor, elimination.gens[0]))
        if primitive.degree() > MAX_ROOT_CRITICAL_DISTANCE_DEGREE:
            continue
        coefficients = tuple(int(value) for value in primitive.all_coeffs())
        if any(
            len(format_canonical_integer(abs(coefficient)))
            > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
            for coefficient in coefficients
        ):
            raise OperationResourceAdmissionError(
                location=("pairs",),
                code="polynomial.root_critical.distance_coefficient_bound",
                message="eliminated distance coefficients exceed the real-algebraic carrier",
            )
        factors_for_selection.append(primitive)
    minimal, selected_index, lower, upper = _select_eliminated_distance_root(
        factors_for_selection,
        real_lower,
        real_upper,
    )
    if lower < 0:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_sign",
            message="exact distance isolation did not establish nonnegativity",
        )
    coefficients = tuple(int(value) for value in minimal.all_coeffs())
    value = RealAlgebraicValue._from_admitted_polynomial(
        polynomial=coefficients,
        real_root_index=selected_index,
    )
    interval = RationalIsolatingInterval(
        lower=_rational(lower),
        upper=_rational(upper),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )
    return (
        value,
        interval,
        "ZERO_DISTANCE" if lower == upper == 0 else "POSITIVE",
    )


def _root_factor(
    polynomial: sympy.Poly,
    root: Any,
) -> sympy.Poly | None:
    """Recover the rational source factor carrying a CRootOf root."""

    backend_poly = getattr(root, "poly", None)
    if backend_poly is None:
        return None
    try:
        backend_variable = backend_poly.gens[0]
        candidate = _primitive_integer_poly(
            sympy.Poly(
                backend_poly.as_expr().subs(backend_variable, polynomial.gens[0]),
                polynomial.gens[0],
                domain=sympy.QQ,
            )
        )
    except (AttributeError, TypeError, ValueError, sympy.PolynomialError):
        return None
    for factor, _multiplicity in polynomial.factor_list()[1]:
        primitive = _primitive_integer_poly(factor)
        if primitive == candidate:
            return primitive
    return None


def _eliminated_distance_polynomial(
    factor: sympy.Poly,
    root: Any,
    critical: CanonicalRational,
) -> sympy.Poly:
    """Eliminate a source root against a rational critical point.

    Real roots use a univariate resultant.  Non-real roots use real and
    imaginary coordinates, which enforces conjugacy geometrically and avoids
    the unsound independent-``CRootOf`` product that motivated this regime.
    """

    z = factor.gens[0]
    c = sympy.Rational(critical.num, critical.den)
    distance = sympy.Symbol("distance")
    request_checkpoint("during root-critical distance elimination")
    if _root_is_real(root):
        result = sympy.resultant(factor.as_expr(), distance - (z - c) ** 2, z)
        return _primitive_integer_poly(sympy.Poly(result, distance, domain=sympy.QQ))
    real_part = sympy.Symbol("root_real", real=True)
    imag_part = sympy.Symbol("root_imag", real=True)
    expression = sympy.expand(factor.as_expr().subs(z, real_part + sympy.I * imag_part))
    real_equation, imaginary_equation = expression.as_real_imag()
    ideal = sympy.groebner(
        (
            real_equation,
            imaginary_equation,
            distance - ((real_part - c) ** 2 + imag_part**2),
        ),
        real_part,
        imag_part,
        distance,
        order="lex",
    )
    distance_polynomials = [
        sympy.Poly(generator, distance, domain=sympy.QQ)
        for generator in ideal.polys
        if generator.as_expr().free_symbols <= {distance}
    ]
    if not distance_polynomials:
        raise OperationDomainValidationError(
            location=("pairs",),
            code="polynomial.root_critical.distance_elimination",
            message="real/imaginary distance elimination produced no univariate polynomial",
        )
    result = min(distance_polynomials, key=lambda polynomial: polynomial.degree())
    return _primitive_integer_poly(result)


def _distance_value_from_root_factor(
    factor: sympy.Poly,
    root: Any,
    rectangle: RootCriticalRectangle,
    critical: CanonicalRational,
) -> tuple[
    RealAlgebraicValue,
    RationalIsolatingInterval,
    Literal["POSITIVE", "ZERO_DISTANCE"],
]:
    elimination = _eliminated_distance_polynomial(factor, root, critical)
    return _distance_value_from_elimination(elimination, rectangle, critical)


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
    source_has_backend_roots = any(
        isinstance(root, sympy.RootOf)
        for factor, _ in source_factors
        for root in factor.all_roots()
    )
    derivative_has_backend_roots = any(
        isinstance(root, sympy.RootOf)
        for factor, _ in derivative_factors
        for root in factor.all_roots()
    )
    # The bounded common-embedding regime is deliberately narrow: a CRootOf
    # source root may be paired only with an exact rational critical point.  A
    # second algebraic carrier would require a common number field and is not
    # silently approximated here.
    critical_is_nonrational = any(
        _root_rational_value(root) is None
        for factor, _ in derivative_factors
        for root in factor.all_roots()
    )
    if derivative_has_backend_roots or (
        source_has_backend_roots and critical_is_nonrational
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.root_carrier_backend_form",
            message=(
                "CRootOf distances are admitted only when every critical point "
                "is rational; common algebraic embeddings remain unsupported"
            ),
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
            critical_rational = _root_rational_value(critical)
            source_factor = _root_factor(source, root)
            if (
                isinstance(root, sympy.RootOf)
                and critical_rational is not None
                and source_factor is not None
            ):
                value, interval, kind = _distance_value_from_root_factor(
                    source_factor,
                    root,
                    root_record.rectangle,
                    critical_rational,
                )
            else:
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


def _validate_worker_profile_source_binding(
    profile: RootCriticalDistanceProfile,
    polynomial: RationalPolynomial,
) -> None:
    """Validate worker rows against retained geometric source rectangles.

    This is deliberately a cheap consumer check: it does not rerun
    factorization or distance elimination.  Primitive irreducible row
    polynomials are instead required to have exactly one isolated root inside
    the certified geometric distance enclosure for their indexed pair.  It is
    a consistency check, not adversarial authentication of the worker's
    elimination relation.
    """

    if profile.source_polynomial != polynomial:
        raise RuntimeError(
            "root-critical worker returned a different source polynomial"
        )
    distance_symbol = sympy.Symbol("distance")
    for row in profile.pairs:
        root = profile.roots[row.root_axis_index]
        critical = profile.critical_points[row.critical_axis_index]
        enclosure_lower, enclosure_upper = _distance_box_for_rectangles(
            root.rectangle,
            critical.rectangle,
        )
        value = row.distance_squared
        require_primitive_real_algebraic_value(value, location=("pairs",))
        value_poly = sympy.Poly.from_list(
            list(value.polynomial),
            gens=distance_symbol,
            domain=sympy.ZZ,
        )
        factor_list = sympy.factor_list(value_poly)[1]
        if len(factor_list) != 1 or factor_list[0][1] != 1:
            raise RuntimeError(
                "root-critical worker returned a non-minimal distance polynomial"
            )
        selected_poly, selected_index, selected_lower, selected_upper = (
            _select_eliminated_distance_root(
                [value_poly],
                enclosure_lower,
                enclosure_upper,
            )
        )
        if selected_poly != value_poly or selected_index != value.real_root_index:
            raise RuntimeError(
                "root-critical worker distance root is not bound to its source pair"
            )
        row_lower = row.isolating_interval.lower.as_fraction()
        row_upper = row.isolating_interval.upper.as_fraction()
        if row_lower > selected_lower or row_upper < selected_upper:
            raise RuntimeError(
                "root-critical worker interval does not contain its selected root"
            )
        selected_zero = selected_lower == selected_upper == 0
        if row.kind == "ZERO_DISTANCE" and not selected_zero:
            raise RuntimeError(
                "root-critical worker zero-distance kind is not source-bound"
            )
        if row.kind == "POSITIVE" and selected_zero:
            raise RuntimeError("root-critical worker positive kind is not source-bound")


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
        worker_stdout
        if isinstance(worker_stdout, bytes)
        else encode_strict_json(worker_stdout),
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
    profile = RootCriticalDistanceProfile.model_validate_json(response["profile"])
    _validate_worker_profile_source_binding(profile, polynomial)
    return profile


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


def _run_splitting_worker(
    polynomial: RationalPolynomial,
    *,
    mode: str,
    embedding_index: int,
    max_pair_rows: object,
    deadline: float,
    cancellation_signal: RequestCancellationSignal | None,
) -> tuple[SplittingFieldKernelPayload, RootCriticalDistanceProfile | None]:
    """Run the splitting-field kernel in a killable child and decode its output."""

    remaining = deadline - time.monotonic() - _PROFILE_FINALIZATION_SECONDS
    if remaining <= 0:
        from jacobian._execution import OperationExecutionTimeoutError

        raise OperationExecutionTimeoutError(
            "splitting-field deadline expired before the kernel worker"
        )
    worker_stdout = run_splitting_worker_process(
        polynomial,
        mode=mode,
        embedding_index=embedding_index,
        max_pair_rows=max_pair_rows,
        remaining_seconds=remaining,
        cancellation_signal=cancellation_signal,
    )
    response = loads_strict_json(
        worker_stdout
        if isinstance(worker_stdout, bytes)
        else encode_strict_json(worker_stdout),
        limits=CanonicalLimits(
            max_input_bytes=SPLITTING_STDOUT_BYTES,
            max_output_bytes=SPLITTING_STDOUT_BYTES,
        ),
    )
    if not isinstance(response, dict):
        raise RuntimeError(
            "bounded splitting-field kernel worker returned malformed output"
        )
    if response.get("ok") is False:
        _raise_worker_error(response)
    field = response.get("field")
    if response.get("ok") is not True or not isinstance(field, dict):
        raise RuntimeError(
            "bounded splitting-field kernel worker returned malformed output"
        )
    payload = SplittingFieldKernelPayload.model_validate_json(
        encode_strict_json(field), strict=True
    )
    profile: RootCriticalDistanceProfile | None = None
    raw_profile = response.get("profile")
    if raw_profile is not None:
        if not isinstance(raw_profile, (str, dict)):
            raise RuntimeError(
                "bounded splitting-field kernel worker returned malformed profile"
            )
        profile = RootCriticalDistanceProfile.model_validate_json(
            raw_profile
            if isinstance(raw_profile, str)
            else encode_strict_json(raw_profile),
            strict=True,
        )
    request_checkpoint("after splitting-field kernel worker")
    return payload, profile


def _splitting_deadline(
    execution: RequestExecutionEnvelope,
) -> tuple[float, RequestCancellationSignal | None]:
    deadline = execution.started_at + ROOT_CRITICAL_WALL_SECONDS
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    return deadline, execution.cancellation_signal


def _require_splitting_polynomial(polynomial: object, embedding_index: object) -> None:
    if not isinstance(polynomial, RationalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.root_critical.splitting_field_polynomial_type",
            message="a splitting field requires a rational polynomial",
        )
    if type(embedding_index) is not int or isinstance(embedding_index, bool):
        raise OperationDomainValidationError(
            location=("embedding_index",),
            code="polynomial.root_critical.embedding_index_type",
            message="embedding_index must be a non-boolean integer",
        )
    if embedding_index < 0:
        raise OperationDomainValidationError(
            location=("embedding_index",),
            code="polynomial.root_critical.embedding_index_range",
            message="embedding_index must be nonnegative",
        )


def _field_from_payload(
    payload: SplittingFieldKernelPayload, polynomial: RationalPolynomial
) -> ExactSplittingField:
    return ExactSplittingField._from_kernel(
        source_polynomial=polynomial,
        squarefree_support=payload.squarefree_support,
        defining_polynomial=payload.defining_polynomial,
        conjugation_coefficients=payload.conjugation_coefficients,
        embedding_index=payload.embedding_index,
        embedding_rectangle=payload.embedding_rectangle,
        roots=payload.roots,
    )


def exact_splitting_field(
    polynomial: RationalPolynomial,
    *,
    embedding_index: int = 0,
) -> ExactSplittingField:
    """Return the exact splitting field of the square-free ``p * p'`` support.

    The blocking SymPy phases (``all_roots``, ``to_number_field``, and
    ``minimal_polynomial``) run inside a killable child process so the request
    deadline and cancellation can stop them; the returned field is re-validated
    before it is trusted.
    """

    _require_splitting_polynomial(polynomial, embedding_index)
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()) as execution:
            return _exact_splitting_field_with_execution(
                polynomial, embedding_index, execution
            )
    return _exact_splitting_field_with_execution(polynomial, embedding_index, execution)


def _exact_splitting_field_with_execution(
    polynomial: RationalPolynomial,
    embedding_index: int,
    execution: RequestExecutionEnvelope,
) -> ExactSplittingField:
    deadline, cancellation_signal = _splitting_deadline(execution)
    payload, _profile = _run_splitting_worker(
        polynomial,
        mode="field",
        embedding_index=embedding_index,
        max_pair_rows=0,
        deadline=deadline,
        cancellation_signal=cancellation_signal,
    )
    return _field_from_payload(payload, polynomial)


def splitting_field_distance_profile(
    polynomial: RationalPolynomial,
    splitting_field: ExactSplittingField,
    *,
    max_pair_rows: int = MAX_ROOT_CRITICAL_PAIRS,
) -> SplittingFieldDistanceProfile:
    """Bind the root-critical distance profile to a supplied splitting field.

    The supplied field must be bound to this exact polynomial: its source and
    square-free support must match the computed ``p * p'`` support, and its
    defining polynomial, conjugation element, and root family must agree with
    the field the kernel rebuilds. Distances are returned as exact field
    elements together with the certified nonnegative interval of the in-process
    profile.
    """

    _require_splitting_polynomial(polynomial, 0)
    if not isinstance(splitting_field, ExactSplittingField):
        raise OperationDomainValidationError(
            location=("splitting_field",),
            code="polynomial.root_critical.splitting_field_type",
            message="a bound profile requires an exact splitting field value",
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
    if splitting_field.source_polynomial != polynomial:
        raise OperationDomainValidationError(
            location=("splitting_field", "source_polynomial"),
            code="polynomial.root_critical.binding_source_mismatch",
            message="the splitting field must be bound to the exact source polynomial",
        )
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()) as execution:
            return _splitting_field_profile_with_execution(
                polynomial, splitting_field, max_pair_rows, execution
            )
    return _splitting_field_profile_with_execution(
        polynomial, splitting_field, max_pair_rows, execution
    )


def _splitting_field_profile_with_execution(
    polynomial: RationalPolynomial,
    splitting_field: ExactSplittingField,
    max_pair_rows: int,
    execution: RequestExecutionEnvelope,
) -> SplittingFieldDistanceProfile:
    deadline, cancellation_signal = _splitting_deadline(execution)
    payload, profile = _run_splitting_worker(
        polynomial,
        mode="bind",
        embedding_index=splitting_field.embedding_index,
        max_pair_rows=max_pair_rows,
        deadline=deadline,
        cancellation_signal=cancellation_signal,
    )
    if profile is None:
        raise RuntimeError("bound splitting-field worker did not return a profile")
    rebuilt = _field_from_payload(payload, polynomial)
    rebuilt_root_coefficients = tuple(
        root.coefficients_ascending for root in rebuilt.roots
    )
    supplied_root_coefficients = tuple(
        root.coefficients_ascending for root in splitting_field.roots
    )
    if (
        rebuilt.squarefree_support != splitting_field.squarefree_support
        or rebuilt.defining_polynomial != splitting_field.defining_polynomial
        or rebuilt.conjugation_coefficients != splitting_field.conjugation_coefficients
        or rebuilt_root_coefficients != supplied_root_coefficients
    ):
        raise OperationDomainValidationError(
            location=("splitting_field",),
            code="polynomial.root_critical.binding_field_mismatch",
            message="the supplied splitting field is not the computed exact field",
        )
    return SplittingFieldDistanceProfile._from_kernel(
        splitting_field=splitting_field,
        profile=profile,
        root_field_coefficients=payload.root_field_coefficients,
        critical_field_coefficients=payload.critical_field_coefficients,
        distance_field_coefficients=payload.distance_field_coefficients,
    )
