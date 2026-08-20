"""Exact quadratic form operations using SymPy for linear algebra."""

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
    a = request.form.matrix
    x = request.vector
    n = len(a)
    value = 0
    for i in range(n):
        for j in range(n):
            value += a[i][j] * x[i] * x[j]
    return EvaluationResult(value=value, dimension=n)


def compute_discriminant(request: DiscriminantRequest) -> DiscriminantResult:
    """Compute det(A) for the symmetric matrix A."""
    from sympy import Matrix

    a = request.form.matrix
    n = len(a)
    m = Matrix(a)
    det = int(m.det())
    return DiscriminantResult(discriminant=det, dimension=n)


def compute_signature(request: SignatureRequest) -> SignatureResult:
    """Compute inertia by exact characteristic-polynomial root counting."""
    a = request.form.matrix
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
