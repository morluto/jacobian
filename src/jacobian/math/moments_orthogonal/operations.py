"""Exact bounded native kernels for moments and orthogonal polynomials."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_DIMENSION,
    MAX_MOMENTS,
    MAX_POLYNOMIAL_COUNT,
    MAX_QUADRATURE_MAGNITUDE,
    MAX_QUADRATURE_POINTS,
    MAX_RECURRENCE_ORDER,
    MIN_QUADRATURE_SUBDIAGONAL,
    ChristoffelDarbouxKernel,
    GaussianQuadrature,
    HankelMatrix,
    JacobiMatrix,
    RecurrenceCoefficients,
)

type Poly = tuple[Fraction, ...]


def hankel_matrix(moments: Sequence[Fraction]) -> HankelMatrix:
    """Build the Hankel matrix ``H[i][j] = moment[i+j]`` from a moment sequence.

    The number of moments ``m`` determines a Hankel matrix of dimension
    ``(m + 1) // 2``: the largest square Hankel matrix whose entries never
    reference an out-of-range moment.
    """
    if not 1 <= len(moments) <= MAX_MOMENTS:
        raise ValueError("moment sequence must contain between 1 and 64 moments")
    if any(type(value) is not Fraction for value in moments):
        raise TypeError("moments must use exact Fractions")
    n = (len(moments) + 1) // 2
    if n > MAX_HANKEL_DIMENSION:
        raise ValueError("Hankel matrix dimension exceeds the supported bound")
    matrix = tuple(tuple(moments[i + j] for j in range(n)) for i in range(n))
    return HankelMatrix(matrix=matrix, moments=tuple(moments))


def _inner_product(moments: Sequence[Fraction], p: Poly, q: Poly) -> Fraction:
    """Exact ``<p, q> = sum_{i,j} p_i q_j mu_{i+j}`` from the moment sequence."""
    result = Fraction(0)
    for i, p_i in enumerate(p):
        if p_i == 0:
            continue
        for j, q_j in enumerate(q):
            if q_j == 0:
                continue
            index = i + j
            if index >= len(moments):
                raise ValueError("insufficient moments for the requested order")
            result += p_i * q_j * moments[index]
    return result


def _shift_up(p: Poly) -> Poly:
    """Multiply a polynomial by x (left-shift coefficients)."""
    return (Fraction(0), *p)


def _scale(scalar: Fraction, p: Poly) -> Poly:
    return tuple(scalar * coeff for coeff in p)


def _subtract(p: Poly, q: Poly) -> Poly:
    length = max(len(p), len(q))
    result = [Fraction(0)] * length
    for i, coeff in enumerate(p):
        result[i] += coeff
    for i, coeff in enumerate(q):
        result[i] -= coeff
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def _monic_orthogonal_recurrence(
    moments: Sequence[Fraction], max_order: int
) -> tuple[list[Fraction], list[Fraction]]:
    """Compute monic three-term recurrence coefficients via exact Gram-Schmidt.

    Returns (alpha, beta) where the monic polynomials satisfy
    ``p_{k+1}(x) = (x - alpha_k) p_k(x) - beta_k p_{k-1}(x)``
    with ``p_{-1} = 0``, ``p_0 = 1``, ``beta_0 = mu_0``.

    ``max_order`` recurrence coefficients ``alpha`` are produced, requiring
    at least ``2 * max_order + 1`` moments.
    """
    alpha: list[Fraction] = []
    beta: list[Fraction] = [moments[0]]
    h_prev = Fraction(0)
    p_prev: Poly = ()
    p_curr: Poly = (Fraction(1),)
    h_curr = moments[0]
    for k in range(max_order):
        alpha_k = _inner_product(moments, _shift_up(p_curr), p_curr) / h_curr
        alpha.append(alpha_k)
        beta_k = (
            Fraction(0) if k == 0 else (h_curr / h_prev if h_prev != 0 else Fraction(0))
        )
        x_p = _shift_up(p_curr)
        p_next = _subtract(
            _subtract(x_p, _scale(alpha_k, p_curr)), _scale(beta_k, p_prev)
        )
        h_prev = h_curr
        p_prev = p_curr
        p_curr = p_next
        h_curr = _inner_product(moments, p_curr, p_curr)
        if h_curr <= 0:
            if h_curr == 0:
                raise ValueError(
                    "moment sequence does not define a positive-definite measure"
                )
            raise ValueError(
                "moment sequence does not define a positive-definite measure"
            )
        if k < max_order - 1:
            beta.append(h_curr / h_prev)
    return alpha, beta


def recurrence_coefficients(moments: Sequence[Fraction]) -> RecurrenceCoefficients:
    """Compute monic three-term recurrence coefficients from moments.

    Implements an exact Gram-Schmidt orthogonalization of the monomials
    ``1, x, x^2, ...`` against the moment inner product, then extracts the
    monic three-term recurrence

        p_{0}(x) = 1
        p_{k+1}(x) = (x - alpha_k) p_k(x) - beta_k p_{k-1}(x)

    with ``beta_0 = mu_0`` (the zeroth moment, serving as the inner-product
    reference) and ``p_{-1}(x) = 0``. The returned ``alpha`` are the diagonal
    (shift) coefficients and ``beta`` the subdiagonal (norm-squared ratio)
    coefficients of the symmetric Jacobi matrix.
    """
    m = len(moments)
    if not 1 <= m <= MAX_MOMENTS:
        raise ValueError("moment sequence must contain between 1 and 64 moments")
    # The kernel consumes at most 2 * MAX_RECURRENCE_ORDER + 1 moments; a
    # longer sequence would carry trailing entries whose positivity is never
    # verified, so direct callers face the same boundary as the wire request.
    if m > 2 * MAX_RECURRENCE_ORDER + 1:
        raise ValueError(
            f"moment sequence length {m} exceeds the "
            f"{2 * MAX_RECURRENCE_ORDER + 1} moments consumed by the "
            "maximum supported recurrence order"
        )
    if any(type(value) is not Fraction for value in moments):
        raise TypeError("moments must use exact Fractions")
    # beta_0 = mu_0 seeds every squared norm h_k of the recurrence; a
    # nonpositive seed is never a positive-definite functional and the loop's
    # h_curr check only fires after the first division by this initial norm.
    if moments[0] <= 0:
        raise ValueError(
            "the zeroth moment of a positive-definite moment functional "
            "must be positive"
        )
    max_order = min(MAX_RECURRENCE_ORDER, (m - 1) // 2)
    if max_order < 1:
        return RecurrenceCoefficients(alpha=(), beta=(moments[0],))
    alpha, beta = _monic_orthogonal_recurrence(moments, max_order)
    return RecurrenceCoefficients(alpha=tuple(alpha), beta=tuple(beta))


def _coefficients(
    value: RecurrenceCoefficients,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Unpack the domain-owned recurrence value for exact kernel arithmetic."""
    if not isinstance(value, RecurrenceCoefficients):
        raise TypeError("coefficients must be a RecurrenceCoefficients value")
    return value.alpha, value.beta


def jacobi_matrix(coefficients: RecurrenceCoefficients) -> JacobiMatrix:
    """Build the symmetric tridiagonal Jacobi matrix from recurrence coefficients.

    Accepts the canonical ``RecurrenceCoefficients`` value returned by
    ``recurrence_coefficients`` unchanged. The diagonal entries are
    ``alpha_0, ..., alpha_{n-1}`` and the positive subdiagonal entries are
    ``sqrt(beta_1), ..., sqrt(beta_n)``. Because the square roots may be
    irrational, the returned matrix stores the rational diagonal and the
    rational squared subdiagonal ``beta`` separately so that the full
    symmetric matrix can be reconstructed by any consumer.
    """
    alpha, beta = _coefficients(coefficients)
    if not 1 <= len(beta) <= MAX_RECURRENCE_ORDER:
        raise ValueError("beta must contain between 1 and 16 entries")
    if not 0 <= len(alpha) <= MAX_RECURRENCE_ORDER:
        raise ValueError("alpha out of range")
    if len(alpha) != len(beta) and len(alpha) != len(beta) - 1:
        raise ValueError("alpha must have length len(beta)-1 or len(beta)")
    if any(type(value) is not Fraction for value in alpha):
        raise TypeError("alpha must use exact Fractions")
    if any(type(value) is not Fraction for value in beta):
        raise TypeError("beta must use exact Fractions")
    if beta[0] <= 0:
        raise ValueError("beta_0 (the zeroth moment) must be positive")
    # Squared subdiagonal entries must stay positive: a negative value cannot
    # reconstruct a real symmetric Jacobi matrix.
    for value in tuple(beta)[1 : len(alpha)]:
        if value <= 0:
            raise ValueError(
                "subdiagonal beta entries must be positive squared-norm ratios"
            )
    return JacobiMatrix(
        diagonal=tuple(alpha),
        # The squared subdiagonal entries are beta_1, ..., beta_{n-1}; beta_0 is
        # the zeroth moment and never occupies an off-diagonal position.
        off_diagonal=tuple(beta)[1 : len(alpha)],
    )


def _require_cd_admission(
    alpha: Sequence[Fraction],
    beta: Sequence[Fraction],
    x: Fraction,
    y: Fraction,
) -> None:
    """Shape, exactness, and positivity admission for the CD kernel.

    The kernel consumes only beta_1..beta_{n-1} for n = len(alpha): a
    trailing beta entry beyond that window is never read, so it must not
    reject requests.
    """
    if not 1 <= len(beta) <= MAX_POLYNOMIAL_COUNT:
        raise ValueError("beta must contain between 1 and 32 entries")
    if not 0 <= len(alpha) <= MAX_POLYNOMIAL_COUNT:
        raise ValueError("alpha out of range")
    if len(alpha) != len(beta) and len(alpha) != len(beta) - 1:
        raise ValueError("alpha must have length len(beta)-1 or len(beta)")
    if type(x) is not Fraction or type(y) is not Fraction:
        raise TypeError("x and y must use exact Fractions")
    if any(type(value) is not Fraction for value in alpha):
        raise TypeError("alpha must use exact Fractions")
    if any(type(value) is not Fraction for value in beta):
        raise TypeError("beta must use exact Fractions")
    if beta[0] <= 0:
        raise ValueError("beta_0 must be positive")
    # A negative subdiagonal beta cannot arise from a positive functional;
    # the kernel it produces would be mathematically meaningless.
    for index in range(1, min(len(alpha), len(beta))):
        if beta[index] < 0:
            raise ValueError("subdiagonal beta entries must be nonnegative")


def christoffel_darboux(
    coefficients: RecurrenceCoefficients,
    x: Fraction,
    y: Fraction,
) -> ChristoffelDarbouxKernel:
    """Compute the Christoffel-Darboux kernel ``K_n(x, y)``.

    Accepts the canonical ``RecurrenceCoefficients`` value returned by
    ``recurrence_coefficients`` unchanged, positionally or by keyword, plus
    the exact evaluation points. For the monic orthogonal polynomial family
    with three-term recurrence

        p_{k+1}(x) = (x - alpha_k) p_k(x) - beta_k p_{k-1}(x)

    the squared norms are ``h_0 = beta_0 = mu_0`` and
    ``h_k = beta_k * h_{k-1}`` for ``k >= 1``. The Christoffel-Darboux kernel is

        K_n(x, y) = sum_{k=0}^{n-1} p_k(x) p_k(y) / h_k

    evaluated by forward recurrence of the polynomials at ``x`` and ``y``.
    """
    alpha, beta = _coefficients(coefficients)
    if not isinstance(x, Fraction) or not isinstance(y, Fraction):
        raise TypeError("x and y must use exact Fractions")
    _require_cd_admission(alpha, beta, x, y)
    n = len(alpha)
    if n == 0:
        return ChristoffelDarbouxKernel(
            kernel=Fraction(0), polynomials_evaluated=(Fraction(1),)
        )
    px_prev = Fraction(0)
    px_curr = Fraction(1)
    py_prev = Fraction(0)
    py_curr = Fraction(1)
    h = beta[0]
    kernel = px_curr * py_curr / h
    evaluated: list[Fraction] = [Fraction(1)]
    for k in range(n - 1):
        alpha_k = alpha[k]
        # Step k forms p_{k+1} using the recurrence coefficient beta_k = beta[k]
        # (beta_0 is mu_0 and the k = 0 step multiplies p_{-1} = 0).
        rec_beta = Fraction(0) if k == 0 else beta[k]
        px_next = (x - alpha_k) * px_curr - rec_beta * px_prev
        py_next = (y - alpha_k) * py_curr - rec_beta * py_prev
        px_prev, px_curr = px_curr, px_next
        py_prev, py_curr = py_curr, py_next
        # Advancing the squared norm to h_{k+1} uses beta_{k+1} = beta[k + 1].
        if k + 1 >= len(beta) or beta[k + 1] == 0:
            raise ValueError(
                "recurrence coefficients do not define the requested kernel"
            )
        next_beta = beta[k + 1]
        h = next_beta * h
        kernel += px_curr * py_curr / h
        evaluated.append(px_curr)
    return ChristoffelDarbouxKernel(
        kernel=kernel, polynomials_evaluated=tuple(evaluated)
    )


def _require_finite_double_coefficients(
    alpha: Sequence[Fraction], beta: Sequence[Fraction]
) -> None:
    """Enforce the finite IEEE-double domain the Golub-Welsch backend computes in.

    The decomposition converts every admitted coefficient to an IEEE double: a
    magnitude beyond the double range overflows, a tiny positive value
    underflows to zero, and an underflowed ``beta_0`` silently collapses every
    quadrature weight even though the weights must sum to ``beta_0``. These
    bounds are part of the mathematical domain of the computation, so the
    native kernel enforces them identically to the wire request model and a
    direct caller cannot lose the measure's mass to float conversion.
    """
    if beta[0] <= 0:
        raise ValueError("beta_0 (the zeroth moment) must be positive")
    if beta[0] < MIN_QUADRATURE_SUBDIAGONAL:
        raise ValueError(
            "beta_0 falls below the quadrature underflow bound; "
            "beta_0 must be >= 1e-300 to convert to a finite IEEE double"
        )
    # Subdiagonal entries feed math.sqrt after float conversion; they must be
    # positive squared-norm ratios safely inside the finite double range.
    for index in range(1, min(len(alpha), len(beta))):
        sub = beta[index]
        if sub <= 0:
            raise ValueError(
                "subdiagonal beta entries must be positive squared-norm ratios"
            )
        if sub < MIN_QUADRATURE_SUBDIAGONAL:
            raise ValueError(
                "subdiagonal beta entries fall below the quadrature underflow bound"
            )
    for value in (*alpha, *beta):
        if abs(value) > MAX_QUADRATURE_MAGNITUDE:
            raise ValueError(
                "quadrature coefficients exceed the finite-float magnitude bound"
            )
        try:
            converted = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "quadrature coefficient does not fit in IEEE double"
            ) from error
        if converted == 0.0 and value != 0:
            raise ValueError("quadrature coefficient underflows IEEE double to zero")


def gaussian_quadrature(coefficients: RecurrenceCoefficients) -> GaussianQuadrature:
    """Compute *approximate* Gaussian quadrature nodes and weights via Golub-Welsch.

    Accepts the canonical ``RecurrenceCoefficients`` value returned by
    ``recurrence_coefficients`` unchanged. The nodes are eigenvalues of the
    symmetric tridiagonal Jacobi matrix (generally irrational, e.g.
    ``alpha=(0,0), beta=(1,2)`` has nodes ``±sqrt(2)``) and the weights are
    ``mu_0 * v_{0,i}^2``. The decomposition runs in IEEE doubles, so the
    returned values are **approximations** with double precision, not exact
    algebraic numbers. Each entry is carried as the exact dyadic rational
    image of the computed double for canonical JSON transport.
    """
    import math

    import numpy as np

    alpha, beta = _coefficients(coefficients)
    n = len(alpha)
    if not 1 <= n <= MAX_QUADRATURE_POINTS:
        raise ValueError("alpha must contain between 1 and 16 entries")
    if len(beta) != n and len(beta) != n + 1:
        raise ValueError("beta must have length len(alpha) or len(alpha)+1")
    # The kernel runs in IEEE doubles; enforce its finite-double domain here so
    # direct callers cannot underflow beta_0 (or any coefficient) into silent
    # mass loss, identically to the request model.
    _require_finite_double_coefficients(alpha, beta)
    diagonal = np.array([float(a) for a in alpha], dtype=float)
    off: list[float] = []
    for k in range(n - 1):
        if k + 1 >= len(beta):
            break
        off.append(math.sqrt(float(beta[k + 1])))
    jacobi = np.diag(diagonal)
    if off:
        off_arr = np.array(off, dtype=float)
        jacobi += np.diag(off_arr, 1) + np.diag(off_arr, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(jacobi)
    mu0 = float(beta[0])
    weights = mu0 * eigenvectors[0, :] ** 2
    # Each double is carried as its exact dyadic rational image so the result
    # stays canonical and reconstructible without JSON floats, but the values
    # are explicitly approximate (IEEE-double precision).
    return GaussianQuadrature(
        approximate_nodes=tuple(Fraction(float(v)) for v in eigenvalues),
        approximate_weights=tuple(Fraction(float(w)) for w in weights),
    )


__all__ = [
    "ChristoffelDarbouxKernel",
    "GaussianQuadrature",
    "HankelMatrix",
    "JacobiMatrix",
    "RecurrenceCoefficients",
    "christoffel_darboux",
    "gaussian_quadrature",
    "hankel_matrix",
    "jacobi_matrix",
    "recurrence_coefficients",
]
