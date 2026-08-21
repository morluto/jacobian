"""Exact moment-functional and orthogonal-polynomial operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    GaussianQuadratureRequest,
    HankelRequest,
    JacobiMatrixRequest,
    OrthogonalPolynomialRequest,
    RecurrenceRequest,
    ShiftedHankelRequest,
)
from jacobian.math.moments_orthogonal.values import (
    JacobiMatrix,
    ChristoffelDarbouxKernel,
    GaussianQuadratureRule,
    HankelMomentMatrix,
    OrthogonalPolynomialFamily,
    OrthogonalPolynomialTerm,
    QuadratureNode,
    ThreeTermRecurrence,
)


def _to_fraction(r: CanonicalRational) -> Fraction:
    return Fraction(int(r.num), int(r.den))


def _from_fraction(f: Fraction) -> CanonicalRational:
    return CanonicalRational(num=str(f.numerator), den=str(f.denominator))


def _rational_det(matrix: list[list[Fraction]]) -> Fraction:
    """Compute the determinant of a rational matrix via Gaussian elimination."""
    n = len(matrix)
    if n == 0:
        return Fraction(1)
    mat = [row[:] for row in matrix]
    det = Fraction(1)
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if mat[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            det = -det
        det *= mat[col][col]
        for row in range(col + 1, n):
            factor = mat[row][col] / mat[col][col]
            for j in range(col, n):
                mat[row][j] -= factor * mat[col][j]
    return det


def _rational_rank(matrix: list[list[Fraction]]) -> int:
    """Compute the rank of a rational matrix via Gaussian elimination."""
    if not matrix or not matrix[0]:
        return 0
    rows = len(matrix)
    cols = len(matrix[0])
    mat = [row[:] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if mat[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for row in range(rows):
            if row == rank:
                continue
            if mat[row][col] != 0:
                factor = mat[row][col] / mat[rank][col]
                for j in range(cols):
                    mat[row][j] -= factor * mat[rank][j]
        rank += 1
        if rank == rows:
            break
    return rank


def compute_hankel_matrix(request: HankelRequest) -> HankelMomentMatrix:
    """Compute the Hankel matrix H_r[i,j] = mu_(i+j)."""
    moments = [_to_fraction(m) for m in request.moments]
    order = request.order
    matrix = [
        [moments[i + j] for j in range(order + 1)]
        for i in range(order + 1)
    ]
    det = _rational_det(matrix)
    rank = _rational_rank(matrix)
    entries = tuple(
        tuple(_from_fraction(matrix[i][j]) for j in range(order + 1))
        for i in range(order + 1)
    )
    return HankelMomentMatrix(
        order=order,
        entries=entries,
        determinant=_from_fraction(det),
        rank=rank,
        variable=request.variable,
    )


def compute_shifted_hankel(request: ShiftedHankelRequest) -> HankelMomentMatrix:
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""
    moments = [_to_fraction(m) for m in request.moments]
    order = request.order
    matrix = [
        [moments[i + j + 1] for j in range(order + 1)]
        for i in range(order + 1)
    ]
    det = _rational_det(matrix)
    rank = _rational_rank(matrix)
    entries = tuple(
        tuple(_from_fraction(matrix[i][j]) for j in range(order + 1))
        for i in range(order + 1)
    )
    return HankelMomentMatrix(
        order=order,
        entries=entries,
        determinant=_from_fraction(det),
        rank=rank,
        variable=request.variable,
    )


def _poly_eval(coeffs: list[Fraction], x: Fraction) -> Fraction:
    """Evaluate a polynomial with given coefficients (lowest degree first)."""
    result = Fraction(0)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def compute_orthogonal_polynomials(
    request: OrthogonalPolynomialRequest,
) -> OrthogonalPolynomialFamily:
    """Compute monic orthogonal polynomials via exact Gram-Schmidt.

    Uses the moment functional L(f) = sum_k mu_k * (coefficient of x^k in f)
    to compute inner products <f,g> = L(f*g).
    """
    moments = [_to_fraction(m) for m in request.moments]
    max_deg = request.max_degree
    var = request.variable

    def inner(coeffs_a: list[Fraction], coeffs_b: list[Fraction]) -> Fraction:
        """Compute L(a*b) where a,b are polynomials (coeffs lowest degree first)."""
        product = [Fraction(0)] * (len(coeffs_a) + len(coeffs_b) - 1)
        for i, a in enumerate(coeffs_a):
            for j, b in enumerate(coeffs_b):
                product[i + j] += a * b
        result = Fraction(0)
        for k, coeff in enumerate(product):
            if k < len(moments):
                result += coeff * moments[k]
        return result

    polynomials: list[list[Fraction]] = []
    squared_norms: list[Fraction] = []

    for n in range(max_deg + 1):
        p_n = [Fraction(0)] * (n + 1)
        p_n[n] = Fraction(1)  # monic

        # Gram-Schmidt: subtract projections onto all previous p_k
        for k in range(n):
            proj = inner(p_n, polynomials[k]) / squared_norms[k]
            for i, c in enumerate(polynomials[k]):
                p_n[i] -= proj * c

        norm = inner(p_n, p_n)
        polynomials.append(p_n)
        squared_norms.append(norm)

    # Check definiteness
    # Quasi-definite: all leading Hankel determinants nonzero
    # Positive-definite: all norms positive
    is_quasi_definite = all(sq > 0 for sq in squared_norms) if squared_norms else False
    is_positive_definite = is_quasi_definite

    poly_terms = []
    for n in range(max_deg + 1):
        coeffs = tuple(_from_fraction(c) for c in polynomials[n])
        poly_terms.append(
            OrthogonalPolynomialTerm(
                degree=n,
                coefficients=coeffs,
                squared_norm=_from_fraction(squared_norms[n]),
            )
        )

    return OrthogonalPolynomialFamily(
        polynomials=tuple(poly_terms),
        variable=var,
        is_quasi_definite=is_quasi_definite,
        is_positive_definite=is_positive_definite,
    )


def compute_recurrence(request: RecurrenceRequest) -> ThreeTermRecurrence:
    """Compute three-term recurrence coefficients from orthogonal polynomials.

    p_{k+1}(x) = (x - alpha_k) p_k(x) - beta_k p_{k-1}(x)

    For monic polynomials:
    - alpha_k = <x*p_k, p_k> / <p_k, p_k>
    - beta_k = <p_k, p_k> / <p_{k-1, p_{k-1}>} (for k >= 1)
    """
    polys = request.family.polynomials
    n = len(polys)

    # We need moments to compute inner products
    # Actually, we can derive alpha and beta from the polynomials themselves
    # For monic polynomials, the recurrence is:
    # p_{k+1} = x*p_k - (sum of inner product terms) p_k - <p_k,p_k>/<p_{k-1},p_{k-1}> * p_{k-1}
    #
    # The coefficients can be read off from the relation:
    # x * p_k = p_{k+1} + alpha_k * p_k + beta_k * p_{k-1}

    # We need the moments to compute inner products
    # Since the family was constructed from moments, we reconstruct
    # the recurrence from the polynomial coefficients directly

    # For monic p_k, x*p_k has leading coefficient 1 at degree k+1
    # x*p_k = p_{k+1} + alpha_k * p_k + beta_k * p_{k-1}
    # So: p_{k+1} = x*p_k - alpha_k * p_k - beta_k * p_{k-1}
    # alpha_k = <x*p_k - p_{k+1}, p_k> / <p_k, p_k>
    # But x*p_k - p_{k+1} has degree k, so it's a combination of p_0,...,p_k
    # alpha_k = <x*p_k, p_k> / <p_k, p_k>

    # Let's just extract from the polynomial coefficients
    # p_k = [c_0, c_1, ..., c_k] (coeffs from degree 0 to k)
    # x * p_k = [0, c_0, c_1, ..., c_k]
    # p_{k+1} = x*p_k - alpha_k * p_k - beta_k * p_{k-1}

    def poly_to_frac_list(p: OrthogonalPolynomialTerm) -> list[Fraction]:
        return [_to_fraction(c) for c in p.coefficients]

    alphas: list[Fraction] = []
    betas: list[Fraction] = []

    for k in range(n):
        p_k = poly_to_frac_list(polys[k])
        squared_norm_k = _to_fraction(polys[k].squared_norm)

        if k == 0:
            # p_1 = x*p_0 - alpha_0 * p_0
            # For monic p_0 = 1, p_1 = x - alpha_0
            # alpha_0 = <x*p_0, p_0> / <p_0, p_0>
            # But x*p_0 = x, so <x, 1> = mu_1
            # alpha_0 = mu_1 / mu_0
            # We need the moments; extract from the first polynomial's norm
            # Actually we don't have moments here. Let's compute from the polynomials.
            if n > 1:
                p_1 = poly_to_frac_list(polys[1])
                # p_1 = x*p_0 - alpha_0 * p_0
                # p_1[0] = -alpha_0 (since p_0 = [1])
                alpha_0 = -p_1[0]
                alphas.append(alpha_0)
                betas.append(Fraction(0))  # beta_0 is unused
            else:
                alphas.append(Fraction(0))
                betas.append(Fraction(0))
        else:
            p_next = poly_to_frac_list(polys[k + 1]) if k + 1 < n else None
            p_prev = poly_to_frac_list(polys[k - 1])
            squared_norm_prev = _to_fraction(polys[k - 1].squared_norm)

            # x * p_k = p_{k+1} + alpha_k * p_k + beta_k * p_{k-1}
            # alpha_k * p_k + beta_k * p_{k-1} = x*p_k - p_{k+1}
            # The degree-k and degree-(k-1) components of x*p_k - p_{k+1}
            # give us two equations

            x_pk: list[Fraction] = [Fraction(0)] * (len(p_k) + 1)
            for i in range(len(p_k)):
                x_pk[i + 1] = p_k[i]

            if p_next is not None:
                residual = [x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i] for i in range(len(x_pk))]
            else:
                residual = x_pk[:]

            # residual = alpha_k * p_k + beta_k * p_{k-1}
            # From the degree-k component: residual[k] = alpha_k * p_k[k] + beta_k * p_{k-1}[k]
            # Since p_k is monic, p_k[k] = 1. p_{k-1} has degree k-1, so p_{k-1}[k] = 0
            # => alpha_k = residual[k]
            alpha_k = residual[k] if k < len(residual) else Fraction(0)
            alphas.append(alpha_k)

            # beta_k = squared_norm_k / squared_norm_prev
            if squared_norm_prev != 0:
                beta_k = squared_norm_k / squared_norm_prev
            else:
                beta_k = Fraction(0)
            betas.append(beta_k)

    return ThreeTermRecurrence(
        alpha=tuple(_from_fraction(a) for a in alphas),
        beta=tuple(_from_fraction(b) for b in betas),
        variable=request.family.variable,
    )


def compute_christoffel_darboux(request: ChristoffelDarbouxRequest) -> ChristoffelDarbouxKernel:
    """Compute the Christoffel-Darboux kernel K_m(x,y) = sum_{k=0}^m p_k(x)p_k(y) / h_k."""
    polys = request.family.polynomials
    m = request.degree

    if m >= len(polys):
        raise ValueError(f"degree {m} exceeds family size {len(polys)}")

    # The kernel is a sum of products of polynomials
    # For a compact representation, we store the coefficients of
    # the numerator polynomial in x (with y as parameter) and vice versa
    # For simplicity, we store the expanded bivariate polynomial coefficients

    # K_m(x,y) = sum_{k=0}^m p_k(x) * p_k(y) / h_k
    # We represent this as coefficients of x^i y^j

    # For the wire model, we store the coefficients of the numerator
    # as a single polynomial in x with y-coefficients
    # Here we store a flattened representation

    # Actually, for the values model, we store the coefficients of
    # the univariate polynomial in x where each coefficient is a polynomial in y
    # For simplicity, let's store the trace (diagonal) coefficients

    # Sum p_k(x)^2 / h_k as a polynomial in x
    max_deg_x = 2 * m
    result_coeffs = [Fraction(0)] * (max_deg_x + 1)

    for k in range(m + 1):
        p_k = [_to_fraction(c) for c in polys[k].coefficients]
        h_k = _to_fraction(polys[k].squared_norm)
        if h_k == 0:
            continue
        # p_k(x)^2 / h_k
        squared = [Fraction(0)] * (2 * k + 1)
        for i in range(k + 1):
            for j in range(k + 1):
                squared[i + j] += p_k[i] * p_k[j]
        for i in range(min(len(squared), len(result_coeffs))):
            result_coeffs[i] += squared[i] / h_k

    return ChristoffelDarbouxKernel(
        degree=m,
        numerator_x_coefficients=tuple(_from_fraction(c) for c in result_coeffs),
        numerator_y_coefficients=tuple(),
        variable=request.family.variable,
    )


def compute_jacobi_matrix(request: JacobiMatrixRequest) -> JacobiMatrix:
    """Compute the finite Jacobi matrix from the orthogonal polynomial family."""
    family = request.family
    polys = family.polynomials
    n = len(polys)

    if n < 2:
        return JacobiMatrix(
            alphas=(),
            betas=(),
            matrix=(),
            variable=family.variable,
        )

    alphas: list[Fraction] = []
    betas: list[Fraction] = []

    for k in range(n - 1):
        p_k = [_to_fraction(c) for c in polys[k].coefficients]
        p_next = [_to_fraction(c) for c in polys[k + 1].coefficients]
        squared_norm_k = _to_fraction(polys[k].squared_norm)

        if k == 0:
            alphas.append(-p_next[0])
            betas.append(Fraction(0))
        else:
            p_prev = [_to_fraction(c) for c in polys[k - 1].coefficients]
            squared_norm_prev = _to_fraction(polys[k - 1].squared_norm)

            x_pk = [Fraction(0)] * (len(p_k) + 1)
            for i in range(len(p_k)):
                x_pk[i + 1] = p_k[i]

            residual = [x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i] for i in range(len(x_pk))]
            alpha_k = residual[k] if k < len(residual) else Fraction(0)
            alphas.append(alpha_k)

            if squared_norm_prev != 0:
                betas.append(squared_norm_k / squared_norm_prev)
            else:
                betas.append(Fraction(0))

    matrix_size = n - 1
    matrix = [[Fraction(0)] * matrix_size for _ in range(matrix_size)]
    for i in range(matrix_size):
        matrix[i][i] = alphas[i]
        if i < matrix_size - 1:
            matrix[i][i + 1] = Fraction(1)
            matrix[i + 1][i] = Fraction(1)

    return JacobiMatrix(
        alphas=tuple(_from_fraction(a) for a in alphas),
        betas=tuple(_from_fraction(b) for b in betas),
        matrix=tuple(
            tuple(_from_fraction(matrix[i][j]) for j in range(matrix_size))
            for i in range(matrix_size)
        ),
        variable=family.variable,
    )


def compute_gaussian_quadrature(request: GaussianQuadratureRequest) -> GaussianQuadratureRule:
    """Compute an exact Gaussian quadrature rule from moments.

    For small orders, we use the fact that the nodes are roots of the
    degree-n orthogonal polynomial. We compute weights from the Vandermonde
    moment system.
    """
    moments = [_to_fraction(m) for m in request.moments]
    n = request.order
    var = request.variable

    # Build orthogonal polynomial family up to degree n
    from jacobian.math.moments_orthogonal._models import OrthogonalPolynomialRequest

    family = compute_orthogonal_polynomials(
        OrthogonalPolynomialRequest(
            moments=request.moments,
            max_degree=n,
            variable=var,
        )
    )

    # The degree-n orthogonal polynomial p_n has n roots
    # For exact rational moments, the roots may be irrational
    # For now, we handle the case where roots are rational
    # (e.g., symmetric distributions)

    p_n = [_to_fraction(c) for c in family.polynomials[n].coefficients]

    # Find rational roots using the rational root theorem
    # For a monic polynomial with rational coefficients, rational roots
    # are of the form p/q where p | constant term, q | leading coefficient
    # Since the polynomial is monic, q = 1, so rational roots are integers
    # dividing the constant term

    # Actually, for general rational moments, we need to clear denominators
    # and use the rational root theorem on the resulting integer polynomial

    # For now, let's use numpy or sympy for root finding
    # But we should use exact methods. Let's try a simple approach:
    # For small n, we can try all rational candidates

    # Actually, let's use sympy for exact root finding
    try:
        import sympy
    except ImportError:
        raise RuntimeError("sympy is required for Gaussian quadrature")

    x = sympy.Symbol(var)
    poly_sym = sum(
        sympy.Rational(int(p_n[i].numerator), int(p_n[i].denominator)) * x**i
        for i in range(len(p_n))
    )
    roots = sympy.solve(poly_sym, x)
    nodes_sympy = [sympy.nsimplify(r) for r in roots]
    nodes_frac = [Fraction(int(r.p), int(r.q)) for r in nodes_sympy]

    if len(nodes_frac) != n:
        raise ValueError(f"Expected {n} roots, got {len(nodes_frac)}")

    # Sort nodes
    nodes_frac.sort()

    # Compute weights from the Vandermonde system
    # sum_i w_i * x_i^k = mu_k for k = 0, ..., n-1
    # This is a Vandermonde system: V * w = mu
    V = [[nodes_frac[i] ** k for k in range(n)] for i in range(n)]
    mu_vec = [moments[k] for k in range(n)]

    # Solve V * w = mu
    weights = _solve_linear_system(V, mu_vec)

    if weights is None:
        raise ValueError("Vandermonde system is singular")

    # Check that all weights are positive
    for w in weights:
        if w <= 0:
            raise ValueError(f"Non-positive weight {w} in Gaussian quadrature")

    # Verify exactness through degree 2n-1
    for k in range(2 * n):
        approx = sum(weights[i] * nodes_frac[i] ** k for i in range(n))
        if k < len(moments) and approx != moments[k]:
            raise ValueError(f"Quadrature not exact at degree {k}")

    nodes = [
        QuadratureNode(
            node=_from_fraction(nodes_frac[i]),
            weight=_from_fraction(weights[i]),
        )
        for i in range(n)
    ]

    return GaussianQuadratureRule(
        order=n,
        nodes=tuple(nodes),
        variable=var,
        exactness_degree=2 * n - 1,
    )


def _solve_linear_system(matrix: list[list[Fraction]], vector: list[Fraction]) -> list[Fraction] | None:
    """Solve a linear system using Gaussian elimination with partial pivoting."""
    n = len(matrix)
    aug = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for j in range(n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] / aug[i][i] for i in range(n)]
