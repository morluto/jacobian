"""Reusable exact kernels for integral binary quadratic forms."""

from math import isqrt

from jacobian.math.number_theory.quadratic_forms.binary._models import (
    MAX_REPRESENTATION_TARGET,
    BinaryQuadraticFormRepresentation,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    _has_sum_of_two_squares_mod_four_obstruction,
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
    if a <= 0 or c <= 0:
        return False
    if abs(b) > a:
        return False
    if a > c:
        return False
    if abs(b) == a and b < 0:
        return False
    return not (a == c and b < 0)


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


__all__ = ["evaluate", "gcd", "is_reduced", "reduce", "representations"]
