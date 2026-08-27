"""Exact strict polynomial sublevel decomposition backed by SymPy roots."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Any, Literal, cast

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.real_algebra._strict_sublevel_models import (
    AlgebraicMeasureRootTerm,
    LevelEquation,
    LevelRootEndpoint,
    LevelRootReference,
    ScopeEndpoint,
    SourceBoundAlgebraicMeasure,
    StrictSublevelComponent,
    StrictSublevelEndpoint,
    StrictSublevelMeasureRequest,
    StrictSublevelMeasureResult,
)


@dataclass(frozen=True)
class StrictSublevelPayload:
    components: tuple[StrictSublevelComponent, ...]
    measure: SourceBoundAlgebraicMeasure


@dataclass(frozen=True)
class _RealLevelRoot:
    value: Any
    reference: LevelRootReference


def _sympy_rational(value: CanonicalRational) -> Any:
    from sympy import Rational

    return Rational(*value.as_integer_ratio())


def _compare_level_roots(left: _RealLevelRoot, right: _RealLevelRoot) -> int:
    if bool(left.value < right.value):
        return -1
    if bool(left.value > right.value):
        return 1
    raise RuntimeError("distinct positive-threshold level equations shared a root")


def _real_level_roots(polynomial: Any, equation: LevelEquation) -> list[_RealLevelRoot]:
    return [
        _RealLevelRoot(
            value=root,
            reference=LevelRootReference(
                equation=equation,
                root_index=index,
                multiplicity=int(multiplicity),
            ),
        )
        for index, (root, multiplicity) in enumerate(
            polynomial.real_roots(multiple=False, radicals=False)
        )
    ]


def _root_endpoint(root: _RealLevelRoot) -> LevelRootEndpoint:
    return LevelRootEndpoint(root=root.reference)


def _strict_at(polynomial: Any, threshold: Any, point: Any) -> bool:
    return bool(abs(polynomial.eval(point)) < threshold)


def _whole_scope_payload(
    request: StrictSublevelMeasureRequest,
) -> StrictSublevelPayload:
    lower = ScopeEndpoint(value=request.lower)
    upper = ScopeEndpoint(value=request.upper)
    component = StrictSublevelComponent(
        left=lower,
        right=upper,
        left_included=True,
        right_included=True,
    )
    measure = SourceBoundAlgebraicMeasure(
        rational_part=CanonicalRational.from_fraction(
            request.upper.as_fraction() - request.lower.as_fraction()
        )
    )
    return StrictSublevelPayload(components=(component,), measure=measure)


def _empty_payload() -> StrictSublevelPayload:
    return StrictSublevelPayload(
        components=(),
        measure=SourceBoundAlgebraicMeasure(
            rational_part=CanonicalRational(num="0", den="1")
        ),
    )


def _measure_from_components(
    components: tuple[StrictSublevelComponent, ...],
    roots: tuple[_RealLevelRoot, ...],
) -> SourceBoundAlgebraicMeasure:
    from sympy import Rational

    roots_by_key = {
        (root.reference.equation, root.reference.root_index): root for root in roots
    }
    rational_part = Fraction(0)
    coefficients: dict[tuple[LevelEquation, int], int] = {}

    def add_endpoint(endpoint: StrictSublevelEndpoint, sign: int) -> None:
        nonlocal rational_part
        if isinstance(endpoint, ScopeEndpoint):
            rational_part += sign * endpoint.value.as_fraction()
            return
        key = (endpoint.root.equation, endpoint.root.root_index)
        root = roots_by_key[key]
        if isinstance(root.value, Rational):
            rational_part += sign * Fraction(int(root.value.p), int(root.value.q))
            return
        coefficients[key] = coefficients.get(key, 0) + sign

    for component in components:
        add_endpoint(component.left, -1)
        add_endpoint(component.right, 1)

    terms = []
    for key in sorted(coefficients):
        coefficient = coefficients[key]
        if coefficient == 0:
            continue
        if coefficient not in (-1, 1):
            raise RuntimeError("a level boundary had non-canonical measure incidence")
        terms.append(
            AlgebraicMeasureRootTerm(
                root=roots_by_key[key].reference,
                coefficient=cast(Literal[-1, 1], coefficient),
            )
        )
    return SourceBoundAlgebraicMeasure(
        rational_part=CanonicalRational.from_fraction(rational_part),
        root_terms=tuple(terms),
    )


def compute_strict_sublevel_payload(
    request: StrictSublevelMeasureRequest,
) -> StrictSublevelPayload:
    """Return every contributing cell and its exact endpoint-difference sum.

    SymPy supplies the sorted exact real roots of ``f-t`` and ``f+t``.  Since
    their product has positive leading coefficient and even degree, its sign is
    positive at negative infinity and flips exactly at odd-multiplicity level
    roots.  This yields the complete sign chart without numerical samples or a
    custom isolation kernel.
    """

    if request.threshold.as_fraction() == 0:
        return _empty_payload()

    polynomial = rational_polynomial_to_sympy(request.polynomial)
    threshold = _sympy_rational(request.threshold)
    lower_value = _sympy_rational(request.lower)
    upper_value = _sympy_rational(request.upper)

    if request.lower == request.upper:
        if _strict_at(polynomial, threshold, lower_value):
            return _whole_scope_payload(request)
        return _empty_payload()

    if polynomial.degree() <= 0:
        if _strict_at(polynomial, threshold, lower_value):
            return _whole_scope_payload(request)
        return _empty_payload()

    roots = tuple(
        sorted(
            (
                *_real_level_roots(
                    polynomial - threshold,
                    "F_MINUS_THRESHOLD",
                ),
                *_real_level_roots(
                    polynomial + threshold,
                    "F_PLUS_THRESHOLD",
                ),
            ),
            key=cmp_to_key(_compare_level_roots),
        )
    )

    # The level-product sign is positive on the leftmost unbounded cell.
    negative = False
    for root in roots:
        if bool(root.value <= lower_value):
            if root.reference.multiplicity % 2:
                negative = not negative
        else:
            break

    components: list[StrictSublevelComponent] = []
    left_endpoint: StrictSublevelEndpoint = ScopeEndpoint(value=request.lower)
    left_included = _strict_at(polynomial, threshold, lower_value)
    for root in roots:
        if bool(root.value <= lower_value):
            continue
        if bool(root.value >= upper_value):
            break
        endpoint = _root_endpoint(root)
        if negative:
            components.append(
                StrictSublevelComponent(
                    left=left_endpoint,
                    right=endpoint,
                    left_included=left_included,
                    right_included=False,
                )
            )
        if root.reference.multiplicity % 2:
            negative = not negative
        left_endpoint = endpoint
        left_included = False

    if negative:
        components.append(
            StrictSublevelComponent(
                left=left_endpoint,
                right=ScopeEndpoint(value=request.upper),
                left_included=left_included,
                right_included=_strict_at(polynomial, threshold, upper_value),
            )
        )

    component_tuple = tuple(components)
    return StrictSublevelPayload(
        components=component_tuple,
        measure=_measure_from_components(component_tuple, roots),
    )


def verify_strict_sublevel_measure_result(result: StrictSublevelMeasureResult) -> bool:
    """Verify one independently supplied strict-sublevel claim, boundedly."""

    try:
        request = StrictSublevelMeasureRequest(
            polynomial=result.source_polynomial,
            threshold=result.threshold,
            lower=result.lower,
            upper=result.upper,
        )
    except ValidationError:
        return False

    expected = compute_strict_sublevel_payload(request)
    return (
        result.components == expected.components and result.measure == expected.measure
    )


__all__ = ["StrictSublevelPayload", "compute_strict_sublevel_payload"]
