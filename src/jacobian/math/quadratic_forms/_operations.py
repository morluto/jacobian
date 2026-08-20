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


def _representation_numbers(
    form: tuple[tuple[int, ...], ...], bound: int
) -> tuple[int, ...]:
    """Compute r(n) for n = 0, 1, ..., bound.

    Brute-force enumeration over a bounded integer box.
    """
    n = len(form)
    counts = [0] * (bound + 1)

    # Compute bounding box from the form
    # For q(x) = x^T A x, if A is positive definite, the level sets are ellipsoids
    # Use the diagonal to estimate bounds
    from sympy import Matrix

    m = Matrix(form)
    eigenvals = m.eigenvals()

    # Find the minimum positive eigenvalue for bounding
    min_eig = float("inf")
    for eigenval, _ in eigenvals.items():
        val = float(eigenval.evalf()) if hasattr(eigenval, "evalf") else float(eigenval)
        if val > 0 and val < min_eig:
            min_eig = val

    if min_eig == float("inf"):
        min_eig = 1  # degenerate case

    # Bounding box: for q(x) <= bound, |x_i| <= sqrt(bound / min_eig)
    import math

    box_bound = int(math.sqrt(bound / min_eig)) + 2 if bound > 0 else 0

    # Enumerate all integer vectors in the box
    def enumerate_dim(dim, vec):
        if dim == n:
            # Compute q(vec)
            q = sum(form[i][j] * vec[i] * vec[j] for i in range(n) for j in range(n))
            if 0 <= q <= bound:
                counts[q] += 1
            return
        for v in range(-box_bound, box_bound + 1):
            vec.append(v)
            enumerate_dim(dim + 1, vec)
            vec.pop()

    enumerate_dim(0, [])
    return tuple(counts)


def _scale_form(
    form: tuple[tuple[int, ...], ...], factor: int
) -> tuple[tuple[int, ...], ...]:
    """Scale a form by an integer factor."""
    n = len(form)
    return tuple(tuple(factor * form[i][j] for j in range(n)) for i in range(n))


def _direct_sum(
    form1: tuple[tuple[int, ...], ...], form2: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """Block diagonal direct sum A ⊕ B."""
    n1 = len(form1)
    n2 = len(form2)
    result = []
    for i in range(n1 + n2):
        row = [0] * (n1 + n2)
        if i < n1:
            for j in range(n1):
                row[j] = form1[i][j]
        else:
            for j in range(n2):
                row[n1 + j] = form2[i - n1][j]
        result.append(tuple(row))
    return tuple(result)


def compute_representation_numbers(request):
    """Compute representation numbers r(0), ..., r(bound)."""
    from jacobian.math.quadratic_forms._models import RepresentationNumbersResult

    counts = _representation_numbers(request.form.matrix, request.bound)
    return RepresentationNumbersResult(
        form=request.form, bound=request.bound, counts=counts
    )


def compute_theta_series_prefix(request):
    """Compute the theta series prefix q^0 through q^bound."""
    from jacobian.math.quadratic_forms._models import ThetaSeriesPrefixResult

    coeffs = _representation_numbers(request.form.matrix, request.bound)
    return ThetaSeriesPrefixResult(
        form=request.form, bound=request.bound, coefficients=coeffs
    )


def compute_scaling(request):
    """Scale a quadratic form by an integer factor."""
    from jacobian.math.quadratic_forms._models import ScalingResult, SymmetricMatrix

    scaled = _scale_form(request.form.matrix, request.factor)
    return ScalingResult(
        form=request.form,
        factor=request.factor,
        scaled_form=SymmetricMatrix(matrix=scaled),
    )


def compute_direct_sum(request):
    """Compute the block diagonal direct sum of two quadratic forms."""
    from jacobian.math.quadratic_forms._models import DirectSumResult, SymmetricMatrix

    result = _direct_sum(request.form1.matrix, request.form2.matrix)
    return DirectSumResult(
        form1=request.form1,
        form2=request.form2,
        direct_sum=SymmetricMatrix(matrix=result),
    )
