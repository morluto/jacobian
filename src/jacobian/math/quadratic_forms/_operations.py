"""Exact quadratic form operations using SymPy for linear algebra."""

from jacobian._exact import CanonicalInteger
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math.quadratic_forms._models import (
    DirectSumRequest,
    DirectSumResult,
    DiscriminantRequest,
    DiscriminantResult,
    EvaluationRequest,
    EvaluationResult,
    RepresentationNumbersRequest,
    RepresentationNumbersResult,
    ScalingRequest,
    ScalingResult,
    SignatureRequest,
    SignatureResult,
    ThetaSeriesPrefixRequest,
    ThetaSeriesPrefixResult,
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


def _representation_numbers(
    form: tuple[tuple[CanonicalInteger, ...], ...], bound: int
) -> tuple[int, ...]:
    """Compute r(n) for n = 0, 1, ..., bound.

    Brute-force enumeration over a bounded integer box.
    """
    integer_form = tuple(
        tuple(parse_canonical_integer(entry) for entry in row) for row in form
    )
    n = len(integer_form)
    counts = [0] * (bound + 1)

    # Compute bounding box from the form
    # For q(x) = x^T A x, if A is positive definite, the level sets are ellipsoids
    # Use the diagonal to estimate bounds
    from sympy import Matrix

    m = Matrix(integer_form)
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
    def enumerate_dim(dim: int, vec: list[int]) -> None:
        if dim == n:
            # Compute q(vec)
            q = sum(
                integer_form[i][j] * vec[i] * vec[j] for i in range(n) for j in range(n)
            )
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
    form: tuple[tuple[CanonicalInteger, ...], ...], factor: int
) -> tuple[tuple[CanonicalInteger, ...], ...]:
    """Scale a form by an integer factor."""
    n = len(form)
    return tuple(
        tuple(
            format_canonical_integer(factor * parse_canonical_integer(form[i][j]))
            for j in range(n)
        )
        for i in range(n)
    )


def _direct_sum(
    form1: tuple[tuple[CanonicalInteger, ...], ...],
    form2: tuple[tuple[CanonicalInteger, ...], ...],
) -> tuple[tuple[CanonicalInteger, ...], ...]:
    """Block diagonal direct sum A ⊕ B."""
    integer_form1 = tuple(
        tuple(parse_canonical_integer(x) for x in row) for row in form1
    )
    integer_form2 = tuple(
        tuple(parse_canonical_integer(x) for x in row) for row in form2
    )
    n1 = len(integer_form1)
    n2 = len(integer_form2)
    result: list[tuple[int, ...]] = []
    for i in range(n1 + n2):
        row = [0] * (n1 + n2)
        if i < n1:
            for j in range(n1):
                row[j] = integer_form1[i][j]
        else:
            for j in range(n2):
                row[n1 + j] = integer_form2[i - n1][j]
        result.append(tuple(row))
    return tuple(tuple(format_canonical_integer(x) for x in row) for row in result)


def compute_representation_numbers(
    request: RepresentationNumbersRequest,
) -> RepresentationNumbersResult:
    """Compute representation numbers r(0), ..., r(bound)."""
    from jacobian.math.quadratic_forms._models import RepresentationNumbersResult

    counts = _representation_numbers(request.form.matrix, request.bound)
    return RepresentationNumbersResult(
        form=request.form, bound=request.bound, counts=counts
    )


def compute_theta_series_prefix(
    request: ThetaSeriesPrefixRequest,
) -> ThetaSeriesPrefixResult:
    """Compute the theta series prefix q^0 through q^bound."""
    from jacobian.math.quadratic_forms._models import ThetaSeriesPrefixResult

    coeffs = _representation_numbers(request.form.matrix, request.bound)
    return ThetaSeriesPrefixResult(
        form=request.form, bound=request.bound, coefficients=coeffs
    )


def compute_scaling(request: ScalingRequest) -> ScalingResult:
    """Scale a quadratic form by an integer factor."""
    from jacobian.math.quadratic_forms._models import ScalingResult, SymmetricMatrix

    scaled = _scale_form(request.form.matrix, request.factor)
    return ScalingResult(
        form=request.form,
        factor=request.factor,
        scaled_form=SymmetricMatrix(matrix=scaled),
    )


def compute_direct_sum(request: DirectSumRequest) -> DirectSumResult:
    """Compute the block diagonal direct sum of two quadratic forms."""
    from jacobian.math.quadratic_forms._models import DirectSumResult, SymmetricMatrix

    result = _direct_sum(request.form1.matrix, request.form2.matrix)
    return DirectSumResult(
        form1=request.form1,
        form2=request.form2,
        direct_sum=SymmetricMatrix(matrix=result),
    )
