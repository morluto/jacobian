"""Exact bounded integral binary quadratic form operations."""

from collections.abc import Callable

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
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
    BinaryQuadraticFormCheckRequest,
    BinaryQuadraticFormCheckResult,
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormEvaluateResult,
    BinaryQuadraticFormProperEquivRequest,
    BinaryQuadraticFormReducedClassesRequest,
    BinaryQuadraticFormReduceRequest,
    BinaryQuadraticFormRepresentationsRequest,
    BinaryQuadraticFormRepresentationsResult,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    ProperEquivalenceResult,
    ReducedBinaryQuadraticFormResult,
    ReducedClassesResult,
    _require_evaluated_value_bound,
    _require_reduced_class_search_budget,
)


def _admit[AdmissionResult](
    operation: Callable[[], AdmissionResult],
    *,
    location: tuple[str | int, ...],
) -> AdmissionResult:
    try:
        return operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def compute_check(
    request: BinaryQuadraticFormCheckRequest,
) -> BinaryQuadraticFormCheckResult:
    """Check if coefficients form a primitive positive-definite binary quadratic form."""

    a, b, c = request.a, request.b, request.c
    disc = b * b - 4 * a * c

    if a <= 0:
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction="a<=0: form is not positive definite (a must be positive)",
        )
    if disc >= 0:
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"discriminant D={disc}>=0: only negative discriminants are supported",
        )
    g = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g > 1:
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"gcd(a,b,c)={g}>1: form is not primitive",
        )
    if disc % 4 not in (0, 1):
        return BinaryQuadraticFormCheckResult._from_kernel(
            a=a,
            b=b,
            c=c,
            status="NOT_IN_INITIAL_DOMAIN",
            obstruction=f"discriminant D={disc} mod 4 = {disc % 4}: must be 0 or 1",
        )

    return BinaryQuadraticFormCheckResult._from_kernel(
        a=a,
        b=b,
        c=c,
        status="PRIMITIVE_POSITIVE_DEFINITE",
        form=PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c),
    )


def compute_evaluate(
    request: BinaryQuadraticFormEvaluateRequest,
) -> BinaryQuadraticFormEvaluateResult:
    """Evaluate a binary quadratic form at an integer pair."""

    value = _admit(
        lambda: _require_evaluated_value_bound(request.form, request.x, request.y),
        location=("x", "y"),
    )
    primitive = _gcd(request.x, request.y) == 1
    return BinaryQuadraticFormEvaluateResult._from_kernel(
        form=request.form,
        x=request.x,
        y=request.y,
        value=value,
        primitive=primitive,
    )


def compute_reduce(
    request: BinaryQuadraticFormReduceRequest,
) -> ReducedBinaryQuadraticFormResult:
    """Gauss-reduce a primitive positive-definite form."""

    ra, rb, rc, p, q, r, s = _reduce(request.form.a, request.form.b, request.form.c)
    return ReducedBinaryQuadraticFormResult._from_kernel(
        form=request.form,
        reduced_form=PrimitivePositiveDefiniteBinaryQuadraticForm(a=ra, b=rb, c=rc),
        matrix=((p, q), (r, s)),
    )


def compute_proper_equivalence(
    request: BinaryQuadraticFormProperEquivRequest,
) -> ProperEquivalenceResult:
    """Decide proper (SL_2(Z)) equivalence of two forms."""

    a1, b1, c1 = request.first.a, request.first.b, request.first.c
    a2, b2, c2 = request.second.a, request.second.b, request.second.c

    disc1 = b1 * b1 - 4 * a1 * c1
    disc2 = b2 * b2 - 4 * a2 * c2
    if disc1 != disc2:
        return ProperEquivalenceResult._from_kernel(
            first=request.first,
            second=request.second,
            status="NOT_PROPERLY_EQUIVALENT",
        )

    ra1, rb1, rc1, p1, q1, r1, s1 = _reduce(a1, b1, c1)
    ra2, rb2, rc2, p2, q2, r2, s2 = _reduce(a2, b2, c2)

    if (ra1, rb1, rc1) != (ra2, rb2, rc2):
        return ProperEquivalenceResult._from_kernel(
            first=request.first,
            second=request.second,
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

    return ProperEquivalenceResult._from_kernel(
        first=request.first,
        second=request.second,
        status="PROPERLY_EQUIVALENT",
        matrix=((wp, wq), (wr, ws)),
    )


def compute_reduced_classes(
    request: BinaryQuadraticFormReducedClassesRequest,
) -> ReducedClassesResult:
    """Enumerate all reduced primitive positive-definite classes of a discriminant."""

    _admit(
        lambda: _require_reduced_class_search_budget(request.discriminant),
        location=("discriminant",),
    )
    classes = _enumerate_reduced_classes(request.discriminant)
    return ReducedClassesResult._from_kernel(
        discriminant=request.discriminant, classes=classes
    )


def compute_representations(
    request: BinaryQuadraticFormRepresentationsRequest,
) -> BinaryQuadraticFormRepresentationsResult:
    """Return all ordered signed integer representations of one target exactly."""
    representations = _admit(
        lambda: _representations(request.form, request.target),
        location=("target",),
    )
    return BinaryQuadraticFormRepresentationsResult._from_kernel(
        form=request.form, target=request.target, representations=representations
    )


def _enumerate_reduced_classes(
    discriminant: int,
) -> tuple[PrimitivePositiveDefiniteBinaryQuadraticForm, ...]:
    """Enumerate every reduced primitive class without constructing a result."""
    classes: list[PrimitivePositiveDefiniteBinaryQuadraticForm] = []
    # For reduced forms: |b| <= a <= c, b^2 - 4ac = D
    # Since D < 0, we have 4ac = b^2 - D > 0, so a,c > 0
    # |b| <= a, and a <= c = (b^2 - D) / (4a)
    # So a <= (b^2 - D) / (4a) => 4a^2 <= b^2 - D
    # Also |b| <= a, so b^2 <= a^2
    # From 4ac = b^2 - D and a <= c: a^2 <= ac = (b^2 - D)/4
    # So a <= sqrt(|D|/3) (standard bound)
    import math

    a_bound = math.isqrt(abs(discriminant) // 3) + 1
    for a in range(1, a_bound + 1):
        # b ranges from -a to a
        for b in range(-a, a + 1):
            num = b * b - discriminant  # = 4ac
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
                classes.append(
                    PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c_val)
                )

    classes.sort(key=lambda form: (form.a, form.b, form.c))
    return tuple(classes)
