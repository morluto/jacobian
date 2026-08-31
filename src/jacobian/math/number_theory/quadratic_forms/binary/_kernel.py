"""Reusable exact kernels for integral binary quadratic forms."""

from dataclasses import dataclass
from math import isqrt

from jacobian.math.number_theory.quadratic_forms.binary._models import (
    MAX_REPRESENTATION_TARGET,
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    _has_sum_of_two_squares_mod_four_obstruction,
    _is_reduced_coefficients,
    _representation_y_bound,
    _require_representation_budget,
)


def gcd(a: int, b: int) -> int:
    """Return the positive gcd, including the conventional zero cases."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    if a == 0 and b == 0:
        return 0
    return a or 1


def evaluate(a: int, b: int, c: int, x: int, y: int) -> int:
    """Evaluate ``a*x² + b*x*y + c*y²`` exactly."""
    return a * x * x + b * x * y + c * y * y


def is_reduced(a: int, b: int, c: int) -> bool:
    """Check Gauss reduction: ``|b| <= a <= c``, with tie-breaking ``b >= 0``."""
    return a > 0 and c > 0 and _is_reduced_coefficients(a, b, c)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return nonnegative gcd and deterministic Bezout coefficients."""
    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s if a >= 0 else -old_s, old_t if b >= 0 else -old_t


def _extended_gcd_three(a: int, b: int, c: int) -> tuple[int, int, int, int]:
    gcd_ab, coefficient_a, coefficient_b = _extended_gcd(a, b)
    common_divisor, coefficient_ab, coefficient_c = _extended_gcd(gcd_ab, c)
    return (
        common_divisor,
        coefficient_ab * coefficient_a,
        coefficient_ab * coefficient_b,
        coefficient_c,
    )


def _exact_quotient(numerator: int, denominator: int, label: str) -> int:
    quotient, remainder = divmod(numerator, denominator)
    if remainder:
        raise ArithmeticError(f"{label} is not integral")
    return quotient


@dataclass(frozen=True, slots=True)
class DirectComposition:
    """One direct Gauss composite and its bilinear substitution."""

    a: int
    b: int
    c: int
    x_coefficients: tuple[int, int, int, int]
    y_coefficients: tuple[int, int, int, int]


def compose(
    first: PrimitivePositiveDefiniteBinaryQuadraticForm,
    second: PrimitivePositiveDefiniteBinaryQuadraticForm,
) -> DirectComposition:
    """Return a direct Gauss composite using Buell's ternary-Bezout formula."""
    if first.discriminant != second.discriminant:
        raise ValueError("forms must have the same discriminant")

    a1, b1 = first.a, first.b
    a2, b2 = second.a, second.b
    discriminant = first.discriminant
    beta = _exact_quotient(b1 + b2, 2, "composition parity term")
    common_divisor, t, u, v = _extended_gcd_three(a1, a2, beta)
    if common_divisor <= 0:
        raise ArithmeticError("composition common divisor must be positive")

    composed_a = _exact_quotient(
        a1 * a2,
        common_divisor * common_divisor,
        "composition leading coefficient",
    )
    discriminant_term = _exact_quotient(
        b1 * b2 + discriminant,
        2,
        "composition discriminant term",
    )
    middle_numerator = a1 * b2 * t + a2 * b1 * u + v * discriminant_term
    initial_b = _exact_quotient(
        middle_numerator, common_divisor, "composition middle coefficient"
    )

    x_coefficients = (
        common_divisor,
        _exact_quotient(
            (b2 - initial_b) * common_divisor,
            2 * a2,
            "first mixed composition coefficient",
        ),
        _exact_quotient(
            (b1 - initial_b) * common_divisor,
            2 * a1,
            "second mixed composition coefficient",
        ),
        _exact_quotient(
            (b1 * b2 + discriminant - initial_b * (b1 + b2)) * common_divisor,
            4 * a1 * a2,
            "trailing mixed composition coefficient",
        ),
    )
    y_coefficients = (
        0,
        _exact_quotient(a1, common_divisor, "first Y coefficient"),
        _exact_quotient(a2, common_divisor, "second Y coefficient"),
        _exact_quotient(beta, common_divisor, "trailing Y coefficient"),
    )

    # Replace the incidental middle coefficient by its deterministic residue
    # in [-A, A). This is the SL_2(Z) substitution x -> x+k*y. Adjust the
    # direct-composition map by the inverse substitution so its identity stays
    # attached to the normalized form.
    composed_b = (initial_b + composed_a) % (2 * composed_a) - composed_a
    shift = _exact_quotient(
        composed_b - initial_b,
        2 * composed_a,
        "composition normalization shift",
    )
    normalized_x_coefficients = tuple(
        x_value - shift * y_value
        for x_value, y_value in zip(x_coefficients, y_coefficients, strict=True)
    )
    x_coefficients = (
        normalized_x_coefficients[0],
        normalized_x_coefficients[1],
        normalized_x_coefficients[2],
        normalized_x_coefficients[3],
    )
    c_numerator = composed_b * composed_b - discriminant
    composed_c = _exact_quotient(
        c_numerator,
        4 * composed_a,
        "composition trailing coefficient",
    )
    return DirectComposition(
        a=composed_a,
        b=composed_b,
        c=composed_c,
        x_coefficients=x_coefficients,
        y_coefficients=y_coefficients,
    )


def _reduce_step(
    a: int, b: int, c: int
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """Perform one Gauss reduction step.

    Returns ``(a, b, c, p, q, r, s, new_a, new_b, new_c)``.  An already
    reduced form returns the identity transformation.
    """
    if c < a:
        return a, b, c, 0, -1, 1, 0, c, -b, a
    if abs(b) > a:
        quotient, remainder = divmod(abs(b), 2 * a)
        if remainder * 2 >= 2 * a:
            quotient += 1
        n = -quotient if b > 0 else quotient
        new_b = b + 2 * n * a
        new_c = c + n * b + n * n * a
        return a, b, c, 1, n, 0, 1, a, new_b, new_c
    if abs(b) == a and b < 0:
        new_b = b + 2 * a
        new_c = c + b + a
        return a, b, c, 1, 1, 0, 1, a, new_b, new_c
    if a == c and b < 0:
        return a, b, c, 0, -1, 1, 0, c, -b, a
    return a, b, c, 1, 0, 0, 1, a, b, c


def reduce(a: int, b: int, c: int) -> tuple[int, int, int, int, int, int, int]:
    """Return the reduced coefficients and the certifying SL₂(Z) matrix."""

    def compose(
        m1: tuple[tuple[int, int], tuple[int, int]],
        m2: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        p1, q1 = m1[0]
        r1, s1 = m1[1]
        p2, q2 = m2[0]
        r2, s2 = m2[1]
        return (
            (p1 * p2 + q1 * r2, p1 * q2 + q1 * s2),
            (r1 * p2 + s1 * r2, r1 * q2 + s1 * s2),
        )

    cur_a, cur_b, cur_c = a, b, c
    matrix = ((1, 0), (0, 1))
    max_iter = 100
    for _ in range(max_iter):
        if is_reduced(cur_a, cur_b, cur_c):
            break
        _oa, _ob, _oc, p, q, r, s, na, nb, nc = _reduce_step(cur_a, cur_b, cur_c)
        matrix = compose(matrix, ((p, q), (r, s)))
        cur_a, cur_b, cur_c = na, nb, nc
    else:
        raise RuntimeError("reduction did not converge")

    p, q = matrix[0]
    r, s = matrix[1]
    assert p * s - q * r == 1, "reduction matrix must have det 1"
    return cur_a, cur_b, cur_c, p, q, r, s


def representations(
    form: PrimitivePositiveDefiniteBinaryQuadraticForm, target: int
) -> tuple[BinaryQuadraticFormRepresentation, ...]:
    """Enumerate every ordered signed pair satisfying ``Q(x,y) = target``."""
    if not 0 <= target <= MAX_REPRESENTATION_TARGET:
        raise ValueError(f"target must be between 0 and {MAX_REPRESENTATION_TARGET}")
    _require_representation_budget(form, target)

    a, b, c = form.a, form.b, form.c
    discriminant = form.discriminant
    rows: list[BinaryQuadraticFormRepresentation] = []
    if _has_sum_of_two_squares_mod_four_obstruction(form, target):
        return ()
    y_bound = _representation_y_bound(form, target)
    for y in range(-y_bound, y_bound + 1):
        x_discriminant = 4 * a * target + discriminant * y * y
        if x_discriminant < 0:
            continue
        root = isqrt(x_discriminant)
        if root * root != x_discriminant:
            continue
        numerator = -b * y
        candidates = (
            (numerator + root,) if root == 0 else (numerator - root, numerator + root)
        )
        for value in candidates:
            if value % (2 * a) != 0:
                continue
            x = value // (2 * a)
            if evaluate(a, b, c, x, y) != target:
                raise AssertionError("quadratic discriminant reconstruction failed")
            rows.append(
                BinaryQuadraticFormRepresentation._from_kernel(
                    x=x,
                    y=y,
                    primitive=gcd(x, y) == 1,
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.x, row.y)))


__all__ = [
    "DirectComposition",
    "compose",
    "evaluate",
    "gcd",
    "is_reduced",
    "reduce",
    "representations",
]
