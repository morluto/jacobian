"""Exact quadratic form operations using SymPy for linear algebra."""

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math.quadratic_forms._models import (
    DiscriminantRequest,
    DiscriminantResult,
    EvaluationRequest,
    EvaluationResult,
    SignatureRequest,
    SignatureResult,
)


def evaluate_form(request: EvaluationRequest) -> EvaluationResult:
    """Evaluate q(x) = x^T A x for an integer vector x."""
    a = tuple(
        tuple(parse_canonical_integer(entry) for entry in row)
        for row in request.form.matrix
    )
    x = tuple(parse_canonical_integer(entry) for entry in request.vector)
    n = len(a)
    value = 0
    for i in range(n):
        for j in range(n):
            value += a[i][j] * x[i] * x[j]
    return EvaluationResult(value=format_canonical_integer(value), dimension=n)


def compute_discriminant(request: DiscriminantRequest) -> DiscriminantResult:
    """Compute det(A) for the symmetric matrix A."""
    from sympy import Matrix

    a = tuple(
        tuple(parse_canonical_integer(entry) for entry in row)
        for row in request.form.matrix
    )
    n = len(a)
    m = Matrix(a)
    det = int(m.det())
    return DiscriminantResult(discriminant=format_canonical_integer(det), dimension=n)


def compute_signature(request: SignatureRequest) -> SignatureResult:
    """Compute inertia by exact characteristic-polynomial root counting."""
    a = tuple(
        tuple(parse_canonical_integer(entry) for entry in row)
        for row in request.form.matrix
    )
    n = len(a)
    n_positive, n_negative, n_zero = symmetric_inertia(a)

    return SignatureResult(
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        is_positive_definite=n_positive == n and n_zero == 0 and n_negative == 0,
        is_negative_definite=n_negative == n and n_zero == 0 and n_positive == 0,
        is_indefinite=n_positive > 0 and n_negative > 0,
    )
