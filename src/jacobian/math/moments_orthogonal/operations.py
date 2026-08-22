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
    ChristoffelDarbouxKernel,
    GaussianQuadratureRule,
    HankelMomentMatrix,
    JacobiMatrix,
    OrthogonalPolynomialFamily,
    OrthogonalPolynomialTerm,
    QuadratureNode,
    ThreeTermRecurrence,
)


def _to_fraction(r: CanonicalRational) -> Fraction:
    # Canonical chunked parsing: int() on the decimal strings would trip
    # CPython's 4300-digit conversion limit inside the admitted range.
    return r.as_fraction()


def _from_fraction(f: Fraction) -> CanonicalRational:
    # Canonical chunked formatting mirrors _to_fraction.
    return CanonicalRational.from_fraction(f)


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
        pivot = next((row for row in range(rank, rows) if mat[row][col] != 0), None)
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        _eliminate_below(mat, rank, col)
        rank += 1
        if rank == rows:
            break
    return rank


def _eliminate_below(mat: list[list[Fraction]], rank: int, col: int) -> None:
    """Clear ``col`` in every row except the current pivot row."""
    cols = len(mat[0])
    for row in range(len(mat)):
        if row == rank or mat[row][col] == 0:
            continue
        factor = mat[row][col] / mat[rank][col]
        for j in range(cols):
            mat[row][j] -= factor * mat[rank][j]


def compute_hankel_matrix(request: HankelRequest) -> HankelMomentMatrix:
    """Compute the Hankel matrix H_r[i,j] = mu_(i+j)."""
    moments = [_to_fraction(m) for m in request.prefix.moments]
    order = request.order
    matrix = [[moments[i + j] for j in range(order + 1)] for i in range(order + 1)]
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
        variable=request.prefix.variable,
    )


def compute_shifted_hankel(request: ShiftedHankelRequest) -> HankelMomentMatrix:
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""
    moments = [_to_fraction(m) for m in request.prefix.moments]
    order = request.order
    matrix = [[moments[i + j + 1] for j in range(order + 1)] for i in range(order + 1)]
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
        variable=request.prefix.variable,
    )


def _poly_eval(coeffs: list[Fraction], x: Fraction) -> Fraction:
    """Evaluate a polynomial with given coefficients (lowest degree first)."""
    result = Fraction(0)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _require_nonzero_norm(norm: Fraction, degree: int) -> None:
    """Quasi-definite prefixes have no vanishing orthogonal-polynomial norm."""
    if norm == 0:
        raise ValueError(
            f"moment functional is not quasi-definite through degree {degree}: "
            f"the squared norm of p_{degree} vanishes, so no orthogonal family "
            "through this degree exists"
        )


def compute_orthogonal_polynomials(
    request: OrthogonalPolynomialRequest,
) -> OrthogonalPolynomialFamily:
    """Compute monic orthogonal polynomials via exact Gram-Schmidt.

    Uses the moment functional L(f) = sum_k mu_k * (coefficient of x^k in f)
    to compute inner products <f,g> = L(f*g).
    """
    moments = [_to_fraction(m) for m in request.prefix.moments]
    max_deg = request.max_degree
    var = request.prefix.variable

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
        _require_nonzero_norm(norm, n)
        polynomials.append(p_n)
        squared_norms.append(norm)

    # Quasi-definiteness requires every computed squared norm to be nonzero;
    # positive-definiteness further requires each norm to be positive.
    # Zero norms are already rejected above.
    is_quasi_definite = all(sq != 0 for sq in squared_norms)
    is_positive_definite = all(sq > 0 for sq in squared_norms)

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

    ``alpha`` carries only coefficients determined by adjacent polynomials:
    reading ``x*p_k = p_{k+1} + alpha_k p_k + beta_k p_{k-1}`` requires both
    ``p_k`` and ``p_{k+1}``, so the terminal ``alpha_{n-1}`` - which would
    need ``p_n`` or additional moments - is omitted.  ``beta[0]`` is an
    unused placeholder; ``beta[k] = <p_k,p_k>/<p_{k-1},p_{k-1}>`` for k >= 1.
    """
    polys = request.family.polynomials
    n = len(polys)

    def poly_to_frac_list(p: OrthogonalPolynomialTerm) -> list[Fraction]:
        return [_to_fraction(c) for c in p.coefficients]

    alphas: list[Fraction] = []
    betas: list[Fraction] = [Fraction(0)]

    for k in range(n):
        squared_norm_k = _to_fraction(polys[k].squared_norm)
        if k > 0:
            betas.append(squared_norm_k / _to_fraction(polys[k - 1].squared_norm))

        if k + 1 >= n:
            # alpha_k cannot be recovered without p_{k+1}; omit it rather
            # than inventing a terminal coefficient.
            break

        p_k = poly_to_frac_list(polys[k])
        p_next = poly_to_frac_list(polys[k + 1])

        x_pk: list[Fraction] = [Fraction(0)] * (len(p_k) + 1)
        for i in range(len(p_k)):
            x_pk[i + 1] = p_k[i]

        residual = [
            x_pk[i] - (p_next[i] if i < len(p_next) else Fraction(0))
            for i in range(len(x_pk))
        ]
        alphas.append(residual[k])

    # Norm ratios can be mathematically valid yet exceed the canonical
    # result range; reject them before constructing wire values.
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
    from jacobian.math._rational_height import RationalHeight

    for value in (*alphas, *betas):
        if RationalHeight.from_canonical(
            CanonicalRational.from_fraction(value)
        ).exceeds(MAX_CANONICAL_RATIONAL_DIGITS):
            raise ValueError(
                "recurrence coefficients exceed the canonical rational "
                "digit limit for this family"
            )
    return ThreeTermRecurrence(
        alpha=tuple(_from_fraction(a) for a in alphas),
        beta=tuple(_from_fraction(b) for b in betas),
        variable=request.family.variable,
    )


def compute_christoffel_darboux(
    request: ChristoffelDarbouxRequest,
) -> ChristoffelDarbouxKernel:
    """Compute the bivariate Christoffel-Darboux kernel K_m(x,y).

    K_m(x, y) = sum_{k=0}^m p_k(x) p_k(y) / h_k is returned as the exact
    bivariate coefficient matrix ``coefficients[i][j]`` of ``x^i y^j`` so
    off-diagonal evaluations and downstream composition stay faithful; the
    diagonal specialization K_m(x,x) is a derived evaluation, not the
    kernel itself.
    """
    polys = request.family.polynomials
    m = request.degree

    if m >= len(polys):
        raise ValueError(f"degree {m} exceeds family size {len(polys)}")

    size = m + 1
    coefficients = [[Fraction(0)] * size for _ in range(size)]

    for k in range(m + 1):
        p_k = [_to_fraction(c) for c in polys[k].coefficients]
        h_k = _to_fraction(polys[k].squared_norm)
        if h_k == 0:
            raise ValueError(
                f"family polynomial p_{k} has zero squared norm; the "
                "Christoffel-Darboux kernel is undefined"
            )
        for i in range(k + 1):
            for j in range(k + 1):
                coefficients[i][j] += p_k[i] * p_k[j] / h_k

    return ChristoffelDarbouxKernel(
        degree=m,
        coefficients=tuple(
            tuple(_from_fraction(coefficients[i][j]) for j in range(size))
            for i in range(size)
        ),
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
            squared_norm_prev = _to_fraction(polys[k - 1].squared_norm)

            x_pk = [Fraction(0)] * (len(p_k) + 1)
            for i in range(len(p_k)):
                x_pk[i + 1] = p_k[i]

            residual = [
                x_pk[i] - p_next[i] if i < len(p_next) else x_pk[i]
                for i in range(len(x_pk))
            ]
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
            # Monic-basis multiplication by x: x p_k = p_{k+1} + alpha_k p_k
            # + beta_k p_{k-1}, so the subdiagonal carries the monic
            # normalization 1 and the superdiagonal carries beta_{i+1}.
            matrix[i + 1][i] = Fraction(1)
            matrix[i][i + 1] = betas[i + 1]

    return JacobiMatrix(
        alphas=tuple(_from_fraction(a) for a in alphas),
        betas=tuple(_from_fraction(b) for b in betas),
        matrix=tuple(
            tuple(_from_fraction(matrix[i][j]) for j in range(matrix_size))
            for i in range(matrix_size)
        ),
        variable=family.variable,
    )


def _construct_quadrature_rule(
    p_n: list[Fraction], moments: list[Fraction], n: int
) -> tuple[list[Fraction], list[Fraction]]:
    """Exact nodes and weights shared by execution and admission replay.

    Nodes are the rational roots of p_n; weights solve the moment
    Vandermonde system V[k][i] = node_i**k against mu_k.
    """
    import sympy

    var_symbol = sympy.Symbol("x")
    poly_sym = sum(
        sympy.Rational(int(p_n[i].numerator), int(p_n[i].denominator)) * var_symbol**i
        for i in range(len(p_n))
    )
    roots = sorted(sympy.solve(poly_sym, var_symbol), key=lambda r: (r, str(r)))
    nodes_frac = []
    for r in roots:
        # Admission guarantees every root is a distinct rational; keep a
        # typed guard so a backend surprise cannot produce a wrong value.
        if not r.is_Rational:
            raise ValueError("orthogonal polynomial produced a non-rational node")
        nodes_frac.append(Fraction(int(r.p), int(r.q)))
    if len(nodes_frac) != n:
        raise ValueError(f"Expected {n} roots, got {len(nodes_frac)}")
    nodes_frac.sort()

    vandermonde = [[nodes_frac[i] ** k for i in range(n)] for k in range(n)]
    weights = _solve_linear_system(vandermonde, moments[:n])
    return nodes_frac, weights


def _build_quadrature_rule(prefix, order: int) -> tuple[list[Fraction], list[Fraction]]:
    """Pure nodes+weights construction shared by execution and validation."""
    from jacobian.math.moments_orthogonal._models import OrthogonalPolynomialRequest

    moments = [_to_fraction(m) for m in prefix.moments]
    family = compute_orthogonal_polynomials(
        OrthogonalPolynomialRequest(prefix=prefix, max_degree=order)
    )
    p_n = [_to_fraction(c) for c in family.polynomials[order].coefficients]
    return _construct_quadrature_rule(p_n, moments, order)


def compute_gaussian_quadrature(
    request: GaussianQuadratureRequest,
) -> GaussianQuadratureRule:
    """Compute an exact Gaussian quadrature rule from moments.

    For small orders, we use the fact that the nodes are roots of the
    degree-n orthogonal polynomial. We compute weights from the Vandermonde
    moment system.
    """
    n = request.order
    var = request.prefix.variable

    nodes_frac, weights = _build_quadrature_rule(request.prefix, n)

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
    import sympy  # noqa: F401 - availability guard for the exact backend

    if weights is None:
        raise ValueError("Vandermonde system is singular")

    # Check that all weights are positive
    for w in weights:
        if w <= 0:
            raise ValueError(f"Non-positive weight {w} in Gaussian quadrature")

    # Verify exactness through degree 2n-1 against the retained prefix.
    prefix_moments = [_to_fraction(m) for m in request.prefix.moments]
    for k in range(2 * n):
        approx = sum(weights[i] * nodes_frac[i] ** k for i in range(n))
        if k < len(prefix_moments) and approx != prefix_moments[k]:
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
        prefix=request.prefix,
    )


def _solve_linear_system(
    matrix: list[list[Fraction]], vector: list[Fraction]
) -> list[Fraction] | None:
    """Solve a linear system using Gaussian elimination with partial pivoting."""
    n = len(matrix)
    aug = [row[:] for row in matrix]
    for i, value in enumerate(vector):
        aug[i].append(value)
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
