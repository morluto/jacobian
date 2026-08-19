"""Exact quadratic form operations using SymPy for linear algebra."""

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
    """Compute the signature (inertia) of a quadratic form using SymPy eigenvalues."""
    from sympy import Matrix

    a = request.form.matrix
    n = len(a)
    m = Matrix(a)

    # Compute eigenvalues
    eigenvals = m.eigenvals()

    n_positive = 0
    n_negative = 0
    n_zero = 0

    for eigenval, mult in eigenvals.items():
        # Use exact sign determination, not int() truncation.
        # int() truncates irrational eigenvalues (e.g. (3-sqrt(5))/2 ≈ 0.38
        # becomes 0), misclassifying positive eigenvalues as zero.
        if eigenval.is_positive:
            n_positive += mult
        elif eigenval.is_negative:
            n_negative += mult
        else:
            n_zero += mult

    return SignatureResult(
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        is_positive_definite=n_positive == n and n_zero == 0 and n_negative == 0,
        is_negative_definite=n_negative == n and n_zero == 0 and n_positive == 0,
        is_indefinite=n_positive > 0 and n_negative > 0,
    )
