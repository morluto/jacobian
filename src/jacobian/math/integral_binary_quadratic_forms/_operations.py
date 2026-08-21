"""Exact bounded integral binary quadratic form operations."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,return-value,type-arg,misc"

from jacobian.math.integral_binary_quadratic_forms._models import (
    BinaryQuadraticFormCheckRequest,
    BinaryQuadraticFormCheckResult,
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormEvaluateResult,
    BinaryQuadraticFormProperEquivRequest,
    BinaryQuadraticFormReducedClassesRequest,
    BinaryQuadraticFormReduceRequest,
    ProperEquivalenceResult,
    ReducedBinaryQuadraticFormResult,
    ReducedClassesResult,
)


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    if a == 0 and b == 0:
        return 0
    return a or 1


def _evaluate(a: int, b: int, c: int, x: int, y: int) -> int:
    return a * x * x + b * x * y + c * y * y


def _transform(
    a: int, b: int, c: int, p: int, q: int, r: int, s: int
) -> tuple[int, int, int]:
    na = a * p * p + b * p * r + c * r * r
    nb = 2 * a * p * q + b * (p * s + q * r) + 2 * c * r * s
    nc = a * q * q + b * q * s + c * s * s
    return na, nb, nc


def _check_reduced(a: int, b: int, c: int) -> bool:
    """Check Gauss reduction: |b| <= a <= c, with tie-breaking b>=0."""
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
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """One reduction step.

    Returns (a, b, c, p, q, r, s, new_a, new_b, new_c).
    If already reduced, returns identity.
    """
    # Step 1: ensure |b| <= a by applying S: [a,b,c] -> [c,-b,a] and T: [a,b,c] -> [a, b+2a, a+b+c]
    # The standard reduction uses:
    # T: [a, b, c] -> [a, b+2a, c+a+b+a] (i.e. b -> b + 2a, c -> c + b + a)
    # S: [a, b, c] -> [c, -b, a]
    # First, if c < a, swap using S
    if c < a:
        # S = [[0,-1],[1,0]], det=1
        return a, b, c, 0, -1, 1, 0, c, -b, a
    # Now c >= a. Reduce |b| <= a using T^n where T^n shifts b by 2n*a
    if abs(b) > a:
        # Find n such that |b + 2n*a| <= a
        # b' = b + 2n*a, we want -a <= b' <= a
        # n = round(-b / (2*a))
        n = round(-b / (2 * a))
        # T^n = [[1,n],[0,1]]
        new_b = b + 2 * n * a
        new_c = c + n * b + n * n * a
        return a, b, c, 1, n, 0, 1, a, new_b, new_c
    # Now |b| <= a <= c. Apply tie-breaking.
    if abs(b) == a and b < 0:
        # T^1: b -> b + 2a, c -> c + b + a
        new_b = b + 2 * a
        new_c = c + b + a
        return a, b, c, 1, 1, 0, 1, a, new_b, new_c
    if a == c and b < 0:
        # S: [a,b,c] -> [c,-b,a]
        return a, b, c, 0, -1, 1, 0, c, -b, a
    # Already reduced
    return a, b, c, 1, 0, 0, 1, a, b, c


def _reduce(a: int, b: int, c: int) -> tuple[int, int, int, int, int, int, int, list]:
    """Full Gauss reduction.

    Returns (ra, rb, rc, p, q, r, s, steps).
    """

    # Compose two SL_2(Z) matrices
    def compose(m1, m2):
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
    steps = []
    max_iter = 100
    for _ in range(max_iter):
        if _check_reduced(cur_a, cur_b, cur_c):
            break
        oa, ob, oc, p, q, r, s, na, nb, nc = _reduce_step(cur_a, cur_b, cur_c)
        step_matrix = ((p, q), (r, s))
        matrix = compose(matrix, step_matrix)
        steps.append((oa, ob, oc, p, q, r, s, na, nb, nc))
        cur_a, cur_b, cur_c = na, nb, nc
    else:
        raise RuntimeError("reduction did not converge")

    # Verify determinant
    p, q = matrix[0]
    r, s = matrix[1]
    assert p * s - q * r == 1, "reduction matrix must have det 1"

    return cur_a, cur_b, cur_c, p, q, r, s, steps


def compute_check(
    request: BinaryQuadraticFormCheckRequest,
) -> BinaryQuadraticFormCheckResult:
    """Check if coefficients form a primitive positive-definite binary quadratic form."""

    a, b, c = request.a, request.b, request.c
    disc = b * b - 4 * a * c

    if a <= 0:
        return BinaryQuadraticFormCheckResult(
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction="a<=0: form is not positive definite (a must be positive)",
        )
    if disc >= 0:
        return BinaryQuadraticFormCheckResult(
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"discriminant D={disc}>=0: only negative discriminants are supported",
        )
    g = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g > 1:
        return BinaryQuadraticFormCheckResult(
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"gcd(a,b,c)={g}>1: form is not primitive",
        )
    if disc % 4 not in (0, 1):
        return BinaryQuadraticFormCheckResult(
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"discriminant D={disc} mod 4 = {disc % 4}: must be 0 or 1",
        )

    return BinaryQuadraticFormCheckResult(
        status="PRIMITIVE_POSITIVE_DEFINITE",
        a=a,
        b=b,
        c=c,
        discriminant=disc,
        gram=((a, b), (b, c)),
    )


def compute_evaluate(
    request: BinaryQuadraticFormEvaluateRequest,
) -> BinaryQuadraticFormEvaluateResult:
    """Evaluate a binary quadratic form at an integer pair."""

    value = _evaluate(request.a, request.b, request.c, request.x, request.y)
    primitive = _gcd(request.x, request.y) == 1
    return BinaryQuadraticFormEvaluateResult(
        a=request.a,
        b=request.b,
        c=request.c,
        x=request.x,
        y=request.y,
        value=value,
        primitive=primitive,
    )


def compute_reduce(
    request: BinaryQuadraticFormReduceRequest,
) -> ReducedBinaryQuadraticFormResult:
    """Gauss-reduce a primitive positive-definite form."""

    ra, rb, rc, p, q, r, s, steps = _reduce(request.a, request.b, request.c)
    return ReducedBinaryQuadraticFormResult(
        a=request.a,
        b=request.b,
        c=request.c,
        reduced_a=ra,
        reduced_b=rb,
        reduced_c=rc,
        matrix=((p, q), (r, s)),
        steps=tuple(steps),
    )


def compute_proper_equivalence(
    request: BinaryQuadraticFormProperEquivRequest,
) -> ProperEquivalenceResult:
    """Decide proper (SL_2(Z)) equivalence of two forms."""

    a1, b1, c1 = request.form1
    a2, b2, c2 = request.form2

    disc1 = b1 * b1 - 4 * a1 * c1
    disc2 = b2 * b2 - 4 * a2 * c2
    if disc1 != disc2:
        return ProperEquivalenceResult(
            form1=request.form1,
            form2=request.form2,
            status="NOT_PROPERLY_EQUIVALENT",
        )

    ra1, rb1, rc1, p1, q1, r1, s1, _ = _reduce(a1, b1, c1)
    ra2, rb2, rc2, p2, q2, r2, s2, _ = _reduce(a2, b2, c2)

    if (ra1, rb1, rc1) != (ra2, rb2, rc2):
        return ProperEquivalenceResult(
            form1=request.form1,
            form2=request.form2,
            status="NOT_PROPERLY_EQUIVALENT",
        )

    # Compute witness: U1 reduces form1 to reduced, U2 reduces form2 to reduced
    # Then form2 = form1^(U1 * U2^{-1})
    # We need to compute U2^{-1}
    # U2 = [[p2, q2], [r2, s2]], det=1, so U2^{-1} = [[s2, -q2], [-r2, p2]]
    # Witness = U1 * U2^{-1}
    inv_p2, inv_q2 = s2, -q2
    inv_r2, inv_s2 = -r2, p2
    # U1 * U2^{-1}
    wp = p1 * inv_p2 + q1 * inv_r2
    wq = p1 * inv_q2 + q1 * inv_s2
    wr = r1 * inv_p2 + s1 * inv_r2
    ws = r1 * inv_q2 + s1 * inv_s2

    return ProperEquivalenceResult(
        form1=request.form1,
        form2=request.form2,
        status="PROPERLY_EQUIVALENT",
        matrix=((wp, wq), (wr, ws)),
    )


def compute_reduced_classes(
    request: BinaryQuadraticFormReducedClassesRequest,
) -> ReducedClassesResult:
    """Enumerate all reduced primitive positive-definite classes of a discriminant."""

    D = request.discriminant  # noqa: N806  # noqa: N806
    if D >= -2:
        return ReducedClassesResult(discriminant=D, classes=(), class_number=0)
    if D % 4 not in (0, 1):
        return ReducedClassesResult(discriminant=D, classes=(), class_number=0)

    classes: list[tuple[int, int, int]] = []
    # For reduced forms: |b| <= a <= c, b^2 - 4ac = D
    # Since D < 0, we have 4ac = b^2 - D > 0, so a,c > 0
    # |b| <= a, and a <= c = (b^2 - D) / (4a)
    # So a <= (b^2 - D) / (4a) => 4a^2 <= b^2 - D
    # Also |b| <= a, so b^2 <= a^2
    # From 4ac = b^2 - D and a <= c: a^2 <= ac = (b^2 - D)/4
    # So a <= sqrt(|D|/3) (standard bound)
    import math

    a_bound = math.isqrt(abs(D) // 3) + 1
    for a in range(1, a_bound + 1):
        # b ranges from -a to a
        for b in range(-a, a + 1):
            num = b * b - D  # = 4ac
            if num % (4 * a) != 0:
                continue
            c_val = num // (4 * a)
            if c_val < a:
                continue
            if c_val == 0:
                continue
            g = _gcd(_gcd(a, abs(b)), c_val)
            if g > 1:
                continue
            if _check_reduced(a, b, c_val):
                classes.append((a, b, c_val))

    classes.sort()
    return ReducedClassesResult(
        discriminant=D,
        classes=tuple(classes),
        class_number=len(classes),
    )
