"""Supported native operations on canonical integral binary quadratic forms."""

from __future__ import annotations

from jacobian.math.number_theory.quadratic_forms.binary._kernel import (
    gcd as _gcd,
)
from jacobian.math.number_theory.quadratic_forms.binary._kernel import (
    is_reduced as _check_reduced,
)
from jacobian.math.number_theory.quadratic_forms.binary._kernel import (
    reduce as _reduce,
)
from jacobian.math.number_theory.quadratic_forms.binary._kernel import (
    representations as _representations,
)
from jacobian.math.number_theory.quadratic_forms.binary._models import (
    MAX_COEFFICIENT,
    BinaryQuadraticFormCheckResult,
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    ProperEquivalenceResult,
    _require_evaluated_value_bound,
    _require_reduced_class_search_budget,
    _require_representation_coordinate,
)


def evaluate(form: PrimitivePositiveDefiniteBinaryQuadraticForm, x: int, y: int) -> int:
    """Return the exact value ``Q(x,y)`` within the public coordinate envelope."""
    _require_representation_coordinate(x)
    _require_representation_coordinate(y)
    return _require_evaluated_value_bound(form, x, y)


def reduced_form(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm,
) -> PrimitivePositiveDefiniteBinaryQuadraticForm:
    """Return the canonical Gauss-reduced representative of ``form``."""
    a, b, c, _p, _q, _r, _s = _reduce(form.a, form.b, form.c)
    return PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c)


def reduction(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm,
) -> tuple[
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    tuple[tuple[int, int], tuple[int, int]],
]:
    """Return the reduced form and its certifying unimodular matrix."""
    a, b, c, p, q, r, s = _reduce(form.a, form.b, form.c)
    return PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c), (
        (p, q),
        (r, s),
    )


def check(a: int, b: int, c: int) -> BinaryQuadraticFormCheckResult:
    """Classify integer coefficients as a primitive positive-definite form."""
    if any(abs(value) > MAX_COEFFICIENT for value in (a, b, c)):
        raise ValueError("form coefficients exceed the supported bound")
    discriminant = b * b - 4 * a * c
    if a <= 0:
        status = "a<=0: form is not positive definite (a must be positive)"
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a, b=b, c=c, status="NOT_IN_INITIAL_DOMAIN", obstruction=status
        )
    if discriminant >= 0:
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"discriminant D={discriminant}>=0: only negative discriminants are supported",
        )
    gcd_value = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if gcd_value > 1:
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"gcd(a,b,c)={gcd_value}>1: form is not primitive",
        )
    if discriminant % 4 not in (0, 1):
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"discriminant D={discriminant} mod 4 = {discriminant % 4}: must be 0 or 1",
        )
    return BinaryQuadraticFormCheckResult._from_kernel(
        a=a,
        b=b,
        c=c,
        status="PRIMITIVE_POSITIVE_DEFINITE",
        form=PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c),
    )


def proper_equivalence(
    first: PrimitivePositiveDefiniteBinaryQuadraticForm,
    second: PrimitivePositiveDefiniteBinaryQuadraticForm,
) -> ProperEquivalenceResult:
    """Decide proper equivalence and return the SL₂(Z) witness when equivalent."""
    if first.discriminant != second.discriminant:
        return ProperEquivalenceResult._from_kernel(
            first=first, second=second, status="NOT_PROPERLY_EQUIVALENT"
        )
    ra1, rb1, rc1, p1, q1, r1, s1 = _reduce(first.a, first.b, first.c)
    ra2, rb2, rc2, p2, q2, r2, s2 = _reduce(second.a, second.b, second.c)
    if (ra1, rb1, rc1) != (ra2, rb2, rc2):
        return ProperEquivalenceResult._from_kernel(
            first=first, second=second, status="NOT_PROPERLY_EQUIVALENT"
        )
    return ProperEquivalenceResult._from_kernel(
        first=first,
        second=second,
        status="PROPERLY_EQUIVALENT",
        matrix=(
            (p1 * s2 - q1 * r2, -p1 * q2 + q1 * p2),
            (r1 * s2 - s1 * r2, -r1 * q2 + s1 * p2),
        ),
    )


def reduced_classes(
    discriminant: int,
) -> tuple[PrimitivePositiveDefiniteBinaryQuadraticForm, ...]:
    """Enumerate all reduced primitive classes of a discriminant."""
    from math import isqrt

    _require_reduced_class_search_budget(discriminant)
    a_bound = isqrt(abs(discriminant) // 3) + 1
    classes: list[PrimitivePositiveDefiniteBinaryQuadraticForm] = []
    for a in range(1, a_bound + 1):
        for b in range(-a, a + 1):
            numerator = b * b - discriminant
            if numerator % (4 * a):
                continue
            c = numerator // (4 * a)
            if c < a or c == 0 or _gcd(_gcd(a, abs(b)), c) > 1:
                continue
            if _check_reduced(a, b, c):
                classes.append(
                    PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c)
                )
    return tuple(sorted(classes, key=lambda form: (form.a, form.b, form.c)))


def representations(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> tuple[BinaryQuadraticFormRepresentation, ...]:
    """Return every ordered signed representation of ``target`` by ``form``."""
    return _representations(form, target)


__all__ = [
    "check",
    "evaluate",
    "proper_equivalence",
    "reduced_classes",
    "reduced_form",
    "representations",
]
