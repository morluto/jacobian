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

    ``max_order`` recurrence coefficients ``alpha`` are produced, consuming
    at most moments up to index ``2 * max_order - 1``: the final coefficient
    pair never requires the norm of the last generated polynomial. An
    odd-length prefix determines that final norm through its last moment, so
    it is still checked for positivity and the ratio it fixes is appended as
    one further ``beta`` entry: every coefficient determined by the retained
    moments is returned.
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
        beta_k = Fraction(0) if k == 0 else h_curr / h_prev
        x_p = _shift_up(p_curr)
        p_next = _subtract(
            _subtract(x_p, _scale(alpha_k, p_curr)), _scale(beta_k, p_prev)
        )
        h_prev = h_curr
        p_prev = p_curr
        p_curr = p_next
        if k == max_order - 1:
            # The coefficients never need the final norm, but an odd-length
            # prefix determines the norm of the last generated polynomial
            # through its last moment; skipping it would admit sequences whose
            # leading Hankel minor is already negative and would drop the one
            # further beta ratio the retained moments determine.
            if len(moments) % 2 == 1:
                h_final = _inner_product(moments, p_curr, p_curr)
                if h_final <= 0:
                    raise ValueError(
                        "moment sequence does not define a positive-definite measure"
                    )
                beta.append(h_final / h_prev)
            break
        h_curr = _inner_product(moments, p_curr, p_curr)
        if h_curr <= 0:
            raise ValueError(
                "moment sequence does not define a positive-definite measure"
            )
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
    if not 1 <= m <= 2 * MAX_RECURRENCE_ORDER:
        raise ValueError(
            "moment sequence must contain between 1 and 32 moments; "
            "larger sequences exceed the domain value bound and would be "
            "rejected by downstream Jacobi consumers"
        )
    if any(type(value) is not Fraction for value in moments):
        raise TypeError("moments must use exact Fractions")
    if moments[0] <= 0:
        raise ValueError("the zeroth moment must be positive")
    max_order = min(MAX_RECURRENCE_ORDER, m // 2)
    if max_order < 1:
        return RecurrenceCoefficients(alpha=(), beta=(moments[0],))
    alpha, beta = _monic_orthogonal_recurrence(moments, max_order)
    return RecurrenceCoefficients(alpha=tuple(alpha), beta=tuple(beta))


def jacobi_matrix(alpha: Sequence[Fraction], beta: Sequence[Fraction]) -> JacobiMatrix:
    """Build the symmetric tridiagonal Jacobi matrix from recurrence coefficients.

    The diagonal entries are ``alpha_0, ..., alpha_{n-1}`` and the positive
    subdiagonal entries are ``sqrt(beta_1), ..., sqrt(beta_n)``. Because the
    square roots may be irrational, the returned matrix stores the rational
    diagonal and the rational squared subdiagonal ``beta`` separately so that the
    full symmetric matrix can be reconstructed by any consumer.
    """
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
    return JacobiMatrix(
        diagonal=tuple(alpha),
        # The squared subdiagonal entries are beta_1, ..., beta_{n-1}; beta_0 is
        # the zeroth moment and never occupies an off-diagonal position.
        off_diagonal=tuple(beta)[1 : len(alpha)],
    )


def christoffel_darboux(
    alpha: Sequence[Fraction],
    beta: Sequence[Fraction],
    x: Fraction,
    y: Fraction,
) -> ChristoffelDarbouxKernel:
    """Compute the Christoffel-Darboux kernel ``K_n(x, y)``.

    For the monic orthogonal polynomial family with three-term recurrence

        p_{k+1}(x) = (x - alpha_k) p_k(x) - beta_k p_{k-1}(x)

    the squared norms are ``h_0 = beta_0 = mu_0`` and
    ``h_k = beta_k * h_{k-1}`` for ``k >= 1``. The Christoffel-Darboux kernel is

        K_n(x, y) = sum_{k=0}^{n-1} p_k(x) p_k(y) / h_k

    evaluated by forward recurrence of the polynomials at ``x`` and ``y``.
    """
    if not 1 <= len(beta) <= MAX_POLYNOMIAL_COUNT:
        raise ValueError("beta must contain between 1 and 32 entries")
    if not 0 <= len(alpha) <= MAX_POLYNOMIAL_COUNT:
        raise ValueError("alpha out of range")
    if len(alpha) != len(beta) and len(alpha) != len(beta) - 1:
        raise ValueError("alpha must have length len(beta)-1 or len(beta)")
    if type(x) is not Fraction or type(y) is not Fraction:
        raise TypeError("x and y must use exact Fractions")
    if beta[0] <= 0:
        raise ValueError("beta_0 must be positive")
    n = len(alpha)
    if n == 0:
        return ChristoffelDarbouxKernel(
            kernel=Fraction(0), polynomials_evaluated=(Fraction(1),)
        )
    # Every subdiagonal ratio the recurrence consumes is a squared-norm ratio
    # of a positive-definite family; admitting nonpositive values would let
    # the documented sum of squared polynomial values turn negative.
    for index in range(1, min(n, len(beta))):
        if beta[index] <= 0:
            raise ValueError(
                "subdiagonal beta entries must be positive squared-norm ratios"
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
        # Advancing the squared norm to h_{k+1} uses beta_{k+1} = beta[k + 1];
        # positivity of every consumed ratio was admitted above.
        next_beta = beta[k + 1]
        h = next_beta * h
        kernel += px_curr * py_curr / h
        evaluated.append(px_curr)
    return ChristoffelDarbouxKernel(
        kernel=kernel, polynomials_evaluated=tuple(evaluated)
    )


def _require_quadrature_double_domain(
    alpha: Sequence[Fraction], beta: Sequence[Fraction]
) -> None:
    """Admit exactly the coefficients Golub-Welsch can convert safely.

    The kernel runs in IEEE doubles, so every coefficient must convert to a
    finite double and every subdiagonal entry must stay far from underflow
    before any conversion happens.
    """

    n = len(alpha)
    if not 1 <= n <= MAX_QUADRATURE_POINTS:
        raise ValueError("alpha must contain between 1 and 16 entries")
    if len(beta) != n and len(beta) != n + 1:
        raise ValueError("beta must have length len(alpha) or len(alpha)+1")
    if beta[0] <= 0:
        raise ValueError("beta_0 (the zeroth moment) must be positive")
    if beta[0] < MIN_QUADRATURE_SUBDIAGONAL:
        raise ValueError(
            "beta_0 falls below the quadrature underflow bound and would give zero weight"
        )
    for index in range(1, min(n, len(beta))):
        if beta[index] <= 0:
            raise ValueError(
                "subdiagonal beta entries must be positive squared-norm ratios"
            )
        if beta[index] < MIN_QUADRATURE_SUBDIAGONAL:
            raise ValueError(
                "subdiagonal beta entries fall below the quadrature underflow bound"
            )
    for value in (*alpha, *beta):
        if abs(value) > MAX_QUADRATURE_MAGNITUDE:
            raise ValueError(
                "quadrature coefficients exceed the finite-float magnitude bound"
            )


def gaussian_quadrature(
    alpha: Sequence[Fraction], beta: Sequence[Fraction]
) -> GaussianQuadrature:
    """Compute Gaussian quadrature nodes and weights via the Golub-Welsch algorithm.

    The nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix and
    the weights are ``mu_0 * v_{0,i}^2`` where ``v_{0,i}`` is the first component of
    the normalized eigenvector for node ``i``. Because the nodes are roots of the
    orthogonal polynomial (generically irrational), the result is a floating-point
    approximation: each returned value is the exact dyadic rational image of one
    IEEE-754 double eigenvalue/weight, not the exact algebraic quadrature node.
    Consumers must treat the result as an approximation with no guaranteed error
    bound beyond the double's rounding.
    """
    import math

    import numpy as np

    _require_quadrature_double_domain(alpha, beta)
    n = len(alpha)
    diagonal = np.array([float(a) for a in alpha], dtype=float)
    off: list[float] = []
    for k in range(n - 1):
        off.append(math.sqrt(float(beta[k + 1])))
    jacobi = np.diag(diagonal)
    if off:
        off_arr = np.array(off, dtype=float)
        jacobi += np.diag(off_arr, 1) + np.diag(off_arr, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(jacobi)
    mu0 = float(beta[0])
    weights = mu0 * eigenvectors[0, :] ** 2
    # Each double is carried as its exact dyadic rational image so the result
    # stays canonical and reconstructible without JSON floating points.
    return GaussianQuadrature(
        nodes=tuple(Fraction(float(v)) for v in eigenvalues),
        weights=tuple(Fraction(float(w)) for w in weights),
    )


__all__ = [
    "christoffel_darboux",
    "gaussian_quadrature",
    "hankel_matrix",
    "jacobi_matrix",
    "recurrence_coefficients",
]
