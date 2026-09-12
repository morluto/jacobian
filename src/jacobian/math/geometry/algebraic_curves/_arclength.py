"""Bounded axis-aligned quadratic arclength kernel.

The kernel keeps the public claim intentionally narrow.  A regular diagonal
quadratic is parameterised by the rational half-angle chart; every retained
parameter cell is integrated by the existing exact-interval Arb operation.
The projective point at infinity is handled by the reciprocal chart, so a
closed ellipse does not require an inexact ``pi`` endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from time import monotonic

import sympy

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError, bind_request_deadline
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis._definite_integral_enclosure import (
    DefiniteIntegralEnclosureRequest,
    DefiniteIntegralEnclosureResult,
    DefiniteIntegralTargetMet,
    _compute_definite_integral_enclosure,
)
from jacobian.math.analysis._models import (
    ExactDyadic,
    IntervalExpressionNode,
    RationalIntervalBox,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.geometry.algebraic_curves._arclength_models import (
    ArclengthEmpty,
    ArclengthEnclosed,
    ArclengthSegment,
    ArclengthSingularUnsupported,
    ArclengthUnknown,
    PlaneCurveArclengthRequest,
    PlaneCurveArclengthResult,
)
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy


@dataclass(frozen=True, slots=True)
class _Ellipse:
    h: Fraction
    k: Fraction
    a2: Fraction
    b2: Fraction


@dataclass(frozen=True, slots=True)
class _ParameterCell:
    lower: Fraction | None
    upper: Fraction | None


def _domain_error(code: str, message: str, *location: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _square_root_rational(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        return None
    return Fraction(numerator, denominator)


def _source_ellipse(request: PlaneCurveArclengthRequest) -> _Ellipse | None:
    source = rational_polynomial_to_sympy(request.polynomial)
    if source.is_zero:
        return None
    if int(source.total_degree()) == 0:
        if source.as_expr() != 0:
            return None
        return None
    if int(source.total_degree()) != 2:
        if any(multiplicity > 1 for _, multiplicity in source.factor_list()[1]):
            raise ValueError("SINGULAR_LEVEL_SET")
        _domain_error(
            "unsupported_degree",
            "the first arclength envelope admits only quadratic plane curves",
            "polynomial",
        )
    x, y = source.gens
    poly = sympy.Poly(source, x, y, domain=sympy.QQ)
    coefficients = {
        monomial: Fraction(int(coefficient.p), int(coefficient.q))
        for monomial, coefficient in poly.terms()
    }
    a = coefficients.get((2, 0), Fraction())
    cross = coefficients.get((1, 1), Fraction())
    c = coefficients.get((0, 2), Fraction())
    d = coefficients.get((1, 0), Fraction())
    e = coefficients.get((0, 1), Fraction())
    f = coefficients.get((0, 0), Fraction())
    if cross != 0 or a <= 0 or c <= 0:
        _domain_error(
            "unsupported_quadratic",
            "the first arclength envelope admits only positive diagonal quadratic parts",
            "polynomial",
        )
    h = -d / (2 * a)
    k = -e / (2 * c)
    radius = a * h * h + c * k * k - f
    if radius < 0:
        return None
    if radius == 0:
        raise ValueError("DEGENERATE_SOURCE")
    return _Ellipse(h=h, k=k, a2=radius / a, b2=radius / c)


def _coordinate_numerator(
    ellipse: _Ellipse, *, axis: str, boundary: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    """Return numerator coefficients for coordinate(t)-boundary."""

    if axis == "x":
        root_squared = ellipse.a2
        # ``a`` is needed only for clipping.  The caller checks it is rational.
        a = _square_root_rational(root_squared)
        assert a is not None
        return (ellipse.h + a - boundary, Fraction(), ellipse.h - a - boundary)
    root_squared = ellipse.b2
    b = _square_root_rational(root_squared)
    assert b is not None
    return (ellipse.k - boundary, 2 * b, ellipse.k - boundary)


def _quadratic_roots(
    coefficients: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, ...] | None:
    constant, linear, quadratic = coefficients
    if quadratic == 0:
        if linear == 0:
            return ()
        return (-constant / linear,)
    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < 0:
        return ()
    root = _square_root_rational(discriminant)
    if root is None:
        return None
    if root == 0:
        raise ValueError("BOUNDARY_NONTRANSVERSE")
    return tuple(
        sorted({(-linear - root) / (2 * quadratic), (-linear + root) / (2 * quadratic)})
    )


def _parameter_coordinates(ellipse: _Ellipse, t: Fraction) -> tuple[Fraction, Fraction]:
    denominator = 1 + t * t
    a = _square_root_rational(ellipse.a2)
    b = _square_root_rational(ellipse.b2)
    if a is None or b is None:
        raise AssertionError("rational coordinates are required for clipping")
    return (
        ellipse.h + a * (1 - t * t) / denominator,
        ellipse.k + b * 2 * t / denominator,
    )


def _cells_for_box(
    ellipse: _Ellipse, request: PlaneCurveArclengthRequest
) -> tuple[_ParameterCell, ...] | str:
    """Return exact cells on the half-angle projective line.

    Irrational edge intersections are a deliberate UNKNOWN boundary in this
    first slice; accepting a floating endpoint would make the lower bound
    unsound.  Tangencies and corner hits are separately typed as unsupported.
    """

    a = _square_root_rational(ellipse.a2)
    b = _square_root_rational(ellipse.b2)
    box_x, box_y = (interval for interval in request.box.intervals)
    if a is None or b is None:
        # A full containment check does not require representing an algebraic
        # boundary endpoint.  A disjoint coordinate range proves emptiness.
        if box_x.upper.as_fraction() < ellipse.h - _sqrt_lower_bound(
            ellipse.a2
        ) or box_x.lower.as_fraction() > ellipse.h + _sqrt_upper_bound(ellipse.a2):
            return ()
        if box_y.upper.as_fraction() < ellipse.k - _sqrt_lower_bound(
            ellipse.b2
        ) or box_y.lower.as_fraction() > ellipse.k + _sqrt_upper_bound(ellipse.b2):
            return ()
        return "IRRATIONAL_BOUNDARY"
    if (
        box_x.upper.as_fraction() < ellipse.h - a
        or box_x.lower.as_fraction() > ellipse.h + a
    ):
        return ()
    if (
        box_y.upper.as_fraction() < ellipse.k - b
        or box_y.lower.as_fraction() > ellipse.k + b
    ):
        return ()

    boundaries: dict[Fraction, set[str]] = {}
    for axis, interval in (("x", box_x), ("y", box_y)):
        for label, boundary in (
            (f"{axis}-lower", interval.lower.as_fraction()),
            (f"{axis}-upper", interval.upper.as_fraction()),
        ):
            boundary_roots = _quadratic_roots(
                _coordinate_numerator(ellipse, axis=axis, boundary=boundary)
            )
            if boundary_roots is None:
                return "IRRATIONAL_BOUNDARY"
            for root in boundary_roots:
                boundaries.setdefault(root, set()).add(label)
    if any(len(labels) > 1 for labels in boundaries.values()):
        return "BOUNDARY_CORNER"
    parameter_roots: tuple[Fraction, ...] = tuple(
        sorted({Fraction(-1), Fraction(0), Fraction(1), *boundaries})
    )
    cells: list[_ParameterCell] = []
    left_edges: tuple[Fraction | None, ...] = (None, *parameter_roots)
    right_edges: tuple[Fraction | None, ...] = (*parameter_roots, None)
    for lower, upper in zip(
        left_edges, right_edges, strict=True
    ):
        if lower is None:
            assert upper is not None
            sample = upper - Fraction(1)
        elif upper is None:
            assert lower is not None
            sample = lower + Fraction(1)
        else:
            sample = (lower + upper) / 2
        x, y = _parameter_coordinates(ellipse, sample)
        if (
            box_x.lower.as_fraction() <= x <= box_x.upper.as_fraction()
            and box_y.lower.as_fraction() <= y <= box_y.upper.as_fraction()
        ):
            cells.append(_ParameterCell(lower=lower, upper=upper))
    return tuple(cells)


def _sqrt_lower_bound(value: Fraction) -> Fraction:
    # A rational lower bound used only to prove a coordinate-range disjointness.
    if value <= 0:
        return Fraction()
    numerator = isqrt(value.numerator * 10**12)
    denominator = isqrt(value.denominator * 10**12)
    return Fraction(numerator, denominator * 10**6)


def _sqrt_upper_bound(value: Fraction) -> Fraction:
    lower = _sqrt_lower_bound(value)
    while lower * lower < value:
        lower += Fraction(1, 10**6)
    return lower


def _const(value: Fraction) -> IntervalExpressionNode:
    return IntervalExpressionNode(
        op="const", value=CanonicalRational.from_fraction(value)
    )


def _var(variable: str) -> IntervalExpressionNode:
    return IntervalExpressionNode(op="var", variable=variable)


def _integrand(
    variable: str, first: Fraction, second: Fraction
) -> IntervalExpressionNode:
    t = _var(variable)
    t2 = IntervalExpressionNode(op="pow", children=(t,), exponent=2)
    if first == second:
        radius = _square_root_rational(first)
        if radius is not None:
            return IntervalExpressionNode(
                op="div",
                children=(
                    _const(2 * radius),
                    IntervalExpressionNode(
                        op="add", children=(_const(Fraction(1)), t2)
                    ),
                ),
            )
    one_minus_t2 = IntervalExpressionNode(op="sub", children=(_const(Fraction(1)), t2))
    numerator_inside = IntervalExpressionNode(
        op="add",
        children=(
            IntervalExpressionNode(op="mul", children=(_const(4 * first), t2)),
            IntervalExpressionNode(
                op="mul",
                children=(
                    _const(second),
                    IntervalExpressionNode(
                        op="pow", children=(one_minus_t2,), exponent=2
                    ),
                ),
            ),
        ),
    )
    numerator = IntervalExpressionNode(
        op="mul",
        children=(
            _const(Fraction(2)),
            IntervalExpressionNode(op="sqrt", children=(numerator_inside,)),
        ),
    )
    denominator = IntervalExpressionNode(
        op="pow",
        children=(
            IntervalExpressionNode(op="add", children=(_const(Fraction(1)), t2)),
        ),
        exponent=2,
    )
    return IntervalExpressionNode(op="div", children=(numerator, denominator))


def _dyadic_target(width: Fraction) -> ExactDyadic:
    bits = 0
    while Fraction(1, 2**bits) > width:
        bits += 1
    return ExactDyadic(mantissa=1, exponent=-bits)


def _integrate_cell(
    ellipse: _Ellipse,
    cell: _ParameterCell,
    request: PlaneCurveArclengthRequest,
    *,
    deadline: float,
    target: Fraction,
) -> tuple[Fraction, Fraction] | None:
    if cell.lower is None and cell.upper is None:
        raise AssertionError("projective parameter cell cannot have two infinite ends")
    if cell.lower is not None and cell.upper is not None:
        intervals = ((cell.lower, cell.upper, ellipse.a2, ellipse.b2),)
    elif cell.lower is None:
        assert cell.upper is not None and cell.upper < 0
        endpoint = -Fraction(1, 1) / cell.upper
        intervals = ((Fraction(0), endpoint, ellipse.a2, ellipse.b2),)
    else:
        assert cell.lower is not None and cell.lower > 0
        endpoint = Fraction(1, 1) / cell.lower
        intervals = ((Fraction(0), endpoint, ellipse.a2, ellipse.b2),)
    lower_total = Fraction()
    upper_total = Fraction()
    for left, right, first, second in intervals:
        if right <= left:
            continue
        midpoint = (left + right) / 2
        for piece_left, piece_right in ((left, midpoint), (midpoint, right)):
            if monotonic() >= deadline:
                raise OperationExecutionTimeoutError(
                    "arclength deadline expired before Arb integration"
                )
            remaining = max(1, int(deadline - monotonic()))
            integral_request = DefiniteIntegralEnclosureRequest(
                expression=_integrand("t", first, second),
                box=RationalIntervalBox(
                    variables=("t",),
                    intervals=(
                        ClosedRationalInterval(
                            lower=CanonicalRational.from_fraction(piece_left),
                            upper=CanonicalRational.from_fraction(piece_right),
                        ),
                    ),
                ),
                precision_bits=request.resource_budget.precision_bits,
                target_width=_dyadic_target(target / 2),
                # The outer segment budget bounds topology cells.  The validated
                # integral owner separately admits its fixed 1,024-leaf ceiling;
                # using that ceiling here is what makes a requested 1/100 circle
                # enclosure a useful accepted regression rather than a merely
                # sound but permanently coarse range.
                max_leaves=1024,
                # The request execution context owns the single absolute deadline;
                # this per-phase value only keeps standalone native calls bounded.
                wall_seconds=(
                    request.resource_budget.wall_seconds
                    if request.resource_budget.wall_seconds > remaining
                    else remaining
                ),
            )
            result: DefiniteIntegralEnclosureResult = (
                _compute_definite_integral_enclosure(integral_request)
            )
            if not isinstance(result.outcome, DefiniteIntegralTargetMet):
                return None
            lower_total += result.outcome.enclosure.lower.as_fraction()
            upper_total += result.outcome.enclosure.upper.as_fraction()
    return lower_total, upper_total


def enclose_arclength(  # noqa: C901
    request: PlaneCurveArclengthRequest,
) -> PlaneCurveArclengthResult:
    started = monotonic()
    deadline = started + request.resource_budget.wall_seconds
    bind_request_deadline(deadline)
    try:
        source = rational_polynomial_to_sympy(request.polynomial)
        if int(source.total_degree()) == 0:
            if source.as_expr() == 0:
                return PlaneCurveArclengthResult._from_kernel(
                    request,
                    outcome=ArclengthSingularUnsupported(reason="DEGENERATE_SOURCE"),
                )
            return PlaneCurveArclengthResult._from_kernel(
                request, outcome=ArclengthEmpty()
            )
        try:
            ellipse = _source_ellipse(request)
        except ValueError as exc:
            if str(exc) == "DEGENERATE_SOURCE":
                return PlaneCurveArclengthResult._from_kernel(
                    request,
                    outcome=ArclengthSingularUnsupported(reason="DEGENERATE_SOURCE"),
                )
            if str(exc) == "BOUNDARY_NONTRANSVERSE":
                return PlaneCurveArclengthResult._from_kernel(
                    request,
                    outcome=ArclengthSingularUnsupported(
                        reason="BOUNDARY_NONTRANSVERSE"
                    ),
                )
            if str(exc) == "SINGULAR_LEVEL_SET":
                return PlaneCurveArclengthResult._from_kernel(
                    request,
                    outcome=ArclengthSingularUnsupported(reason="SINGULAR_LEVEL_SET"),
                )
            raise
        if ellipse is None:
            return PlaneCurveArclengthResult._from_kernel(
                request, outcome=ArclengthEmpty()
            )
        cells = _cells_for_box(ellipse, request)
        if isinstance(cells, str):
            if cells == "BOUNDARY_CORNER":
                return PlaneCurveArclengthResult._from_kernel(
                    request,
                    outcome=ArclengthSingularUnsupported(
                        reason="BOUNDARY_NONTRANSVERSE"
                    ),
                )
            return PlaneCurveArclengthResult._from_kernel(
                request, outcome=ArclengthUnknown(reason="TOPOLOGY_UNRESOLVED")
            )
        if not cells:
            return PlaneCurveArclengthResult._from_kernel(
                request, outcome=ArclengthEmpty()
            )
        if len(cells) > request.resource_budget.max_segments:
            return PlaneCurveArclengthResult._from_kernel(
                request, outcome=ArclengthUnknown(reason="REFINEMENT_INCOMPLETE")
            )
        piece_target = request.target_width.as_fraction() / len(cells)
        segments: list[ArclengthSegment] = []
        lower_total = Fraction()
        upper_total = Fraction()
        for cell in cells:
            contribution = _integrate_cell(
                ellipse, cell, request, deadline=deadline, target=piece_target
            )
            if contribution is None:
                return PlaneCurveArclengthResult._from_kernel(
                    request, outcome=ArclengthUnknown(reason="REFINEMENT_INCOMPLETE")
                )
            lower, upper = contribution
            lower_total += lower
            upper_total += upper
            segments.append(
                ArclengthSegment(
                    lower=(
                        CanonicalRational.from_fraction(cell.lower)
                        if cell.lower is not None
                        else None
                    ),
                    upper=(
                        CanonicalRational.from_fraction(cell.upper)
                        if cell.upper is not None
                        else None
                    ),
                    contribution_lower=CanonicalRational.from_fraction(lower),
                    contribution_upper=CanonicalRational.from_fraction(upper),
                )
            )
        if upper_total - lower_total > request.target_width.as_fraction():
            return PlaneCurveArclengthResult._from_kernel(
                request, outcome=ArclengthUnknown(reason="REFINEMENT_INCOMPLETE")
            )
        return PlaneCurveArclengthResult._from_kernel(
            request,
            outcome=ArclengthEnclosed(
                lower=CanonicalRational.from_fraction(lower_total),
                upper=CanonicalRational.from_fraction(upper_total),
                segments=tuple(segments),
            ),
        )
    except OperationExecutionTimeoutError:
        return PlaneCurveArclengthResult._from_kernel(
            request, outcome=ArclengthUnknown(reason="DEADLINE_EXPIRED")
        )
    except (ImportError, ModuleNotFoundError):
        return PlaneCurveArclengthResult._from_kernel(
            request, outcome=ArclengthUnknown(reason="BACKEND_UNAVAILABLE")
        )


__all__ = ["enclose_arclength"]
