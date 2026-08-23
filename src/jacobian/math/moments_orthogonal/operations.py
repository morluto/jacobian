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
    MomentFunctionalPrefix,
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


_CANONICAL_DIGIT_LIMIT = 10**32_768


def _fraction_exceeds_canonical_limit(value: Fraction) -> bool:
    """Decimal-height test on the exact Fraction, performed BEFORE any
    canonical conversion so an over-tall value is rejected with a typed
    error instead of failing inside ``CanonicalRational`` construction."""
    return (
        abs(value.numerator) >= _CANONICAL_DIGIT_LIMIT
        or value.denominator >= _CANONICAL_DIGIT_LIMIT
    )


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


def _require_canonical_family_values(
    polynomials: list[list[Fraction]], squared_norms: list[Fraction]
) -> None:
    """Typed height gate on derived Gram-Schmidt values.

    Derived coefficients and norms can leave the canonical range even
    when every input moment stays inside it; measure the exact Fractions
    before wire conversion so an over-tall family is reported as a typed
    domain error here instead of failing inside canonical construction.
    """
    for n, coefficients in enumerate(polynomials):
        if any(_fraction_exceeds_canonical_limit(c) for c in coefficients):
            raise ValueError(
                f"derived p_{n} coefficients exceed the canonical rational "
                "digit limit; supply a moment prefix whose orthogonal "
                "family stays representable"
            )
        if _fraction_exceeds_canonical_limit(squared_norms[n]):
            raise ValueError(
                f"derived squared norm h_{n} exceeds the canonical "
                "rational digit limit; supply a moment prefix whose "
                "orthogonal family stays representable"
            )


def _moment_inner(
    moments: list[Fraction],
    coeffs_a: list[Fraction],
    coeffs_b: list[Fraction],
) -> Fraction:
    """L(a*b) for polynomials given lowest-degree-first over the prefix."""
    product = [Fraction(0)] * (len(coeffs_a) + len(coeffs_b) - 1)
    for i, a in enumerate(coeffs_a):
        for j, b in enumerate(coeffs_b):
            product[i + j] += a * b
    result = Fraction(0)
    for k, coeff in enumerate(product):
        if k < len(moments):
            result += coeff * moments[k]
    return result


def _project_out(
    moments: list[Fraction],
    target: list[Fraction],
    basis_polynomial: list[Fraction],
    basis_norm: Fraction,
) -> None:
    """Subtract target's projection onto one basis polynomial."""
    projection = _moment_inner(moments, target, basis_polynomial) / basis_norm
    for i, coefficient in enumerate(basis_polynomial):
        target[i] -= projection * coefficient


def _construct_monic_orthogonal_polynomial(
    moments: list[Fraction], degree: int
) -> list[Fraction]:
    """Exact monic ``p_degree`` via Gram-Schmidt over the moment functional.

    Gaussian construction divides by squared norms only through
    ``p_{degree - 1}``: building ``p_n`` projects onto earlier polynomials,
    and a vanishing terminal norm ``h_degree`` is admissible (it is exactly
    the finite-support case where the measure sits on ``degree`` points).
    The terminal norm is therefore not computed here.
    """
    polynomials: list[list[Fraction]] = []
    squared_norms: list[Fraction] = []
    for n in range(degree):
        p_n = [Fraction(0)] * (n + 1)
        p_n[n] = Fraction(1)
        for k in range(n):
            _project_out(moments, p_n, polynomials[k], squared_norms[k])
        norm = _moment_inner(moments, p_n, p_n)
        _require_nonzero_norm(norm, n)
        polynomials.append(p_n)
        squared_norms.append(norm)

    p_degree = [Fraction(0)] * (degree + 1)
    p_degree[degree] = Fraction(1)
    for k in range(degree):
        _project_out(moments, p_degree, polynomials[k], squared_norms[k])
    return p_degree


def orthogonal_polynomials_from_moments(
    moments: list[Fraction], max_deg: int, var: str
) -> OrthogonalPolynomialFamily:
    """Pure Gram-Schmidt kernel over one bounded moment sequence.

    Shared by the MCP handler, the request admission replay, and the native
    API so no caller performs the exact projection twice.
    """

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

    # Derived coefficients and norms can leave the canonical range even
    # when every input moment stays inside it; measure the exact Fractions
    # before wire conversion so an over-tall family is reported as a typed
    # domain error here instead of failing inside canonical construction.
    _require_canonical_family_values(polynomials, squared_norms)

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


def _require_gram_schmidt_admission(
    prefix: MomentFunctionalPrefix, max_degree: int
) -> None:
    """Shared degree, moment-count, and height admission for the kernel.

    The wire request model enforces the same bounds through its fields and
    validators; native callers bypass that envelope, so this gate keeps
    the exported exact API from fabricating omitted moments (a short
    prefix silently reads missing moments as zero) or accepting
    unsupported degrees.
    """
    from jacobian.math.moments_orthogonal._models import (
        _require_gram_schmidt_heights_admissible,
    )
    from jacobian.math.moments_orthogonal.values import MAX_POLYNOMIAL_DEGREE

    if not 0 <= max_degree <= MAX_POLYNOMIAL_DEGREE:
        raise ValueError(f"max_degree must be between 0 and {MAX_POLYNOMIAL_DEGREE}")
    needed = 2 * max_degree + 1
    if len(prefix.moments) < needed:
        raise ValueError(
            f"need at least {needed} moments for degree {max_degree}, got "
            f"{len(prefix.moments)}"
        )
    _require_gram_schmidt_heights_admissible(prefix.moments, max_degree)


def compute_orthogonal_polynomials(
    request: OrthogonalPolynomialRequest,
) -> OrthogonalPolynomialFamily:
    """MCP adapter: validate the wire request, then run the shared kernel."""
    moments = [_to_fraction(m) for m in request.prefix.moments]
    return orthogonal_polynomials_from_moments(
        moments, request.max_degree, request.prefix.variable
    )


def recurrence_coefficients_from_family(
    family: OrthogonalPolynomialFamily,
) -> ThreeTermRecurrence:
    """Domain kernel: three-term recurrence coefficients of one family.

    p_{k+1}(x) = (x - alpha_k) p_k(x) - beta_k p_{k-1}(x)

    ``alpha`` carries only coefficients determined by adjacent polynomials:
    reading ``x*p_k = p_{k+1} + alpha_k p_k + beta_k p_{k-1}`` requires both
    ``p_k`` and ``p_{k+1}``, so the terminal ``alpha_{n-1}`` - which would
    need ``p_n`` or additional moments - is omitted.  ``beta[0]`` is an
    unused placeholder; ``beta[k] = <p_k,p_k>/<p_{k-1},p_{k-1}>`` for k >= 1.
    """
    polys = family.polynomials
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
    # result range; measure the exact Fraction height before conversion.
    for value in (*alphas, *betas):
        if _fraction_exceeds_canonical_limit(value):
            raise ValueError(
                "recurrence coefficients exceed the canonical rational "
                "digit limit for this family"
            )
    return ThreeTermRecurrence(
        alpha=tuple(_from_fraction(a) for a in alphas),
        beta=tuple(_from_fraction(b) for b in betas),
        variable=family.variable,
    )


def compute_recurrence(request: RecurrenceRequest) -> ThreeTermRecurrence:
    """MCP adapter: parse one request, call the domain kernel once."""
    return recurrence_coefficients_from_family(request.family)


def _kernel_coefficient_matrix(
    polynomials: tuple[OrthogonalPolynomialTerm, ...], m: int
) -> list[list[Fraction]]:
    """Exact bivariate coefficient matrix of K_m from p_0..p_m.

    Shared by execution and the kernel value's defining-sum replay.
    """
    size = m + 1
    coefficients = [[Fraction(0)] * size for _ in range(size)]
    for k in range(m + 1):
        p_k = [_to_fraction(c) for c in polynomials[k].coefficients]
        h_k = _to_fraction(polynomials[k].squared_norm)
        if h_k == 0:
            raise ValueError(
                f"family polynomial p_{k} has zero squared norm; the "
                "Christoffel-Darboux kernel is undefined"
            )
        for i in range(k + 1):
            for j in range(k + 1):
                coefficients[i][j] += p_k[i] * p_k[j] / h_k
    return coefficients


def christoffel_darboux_kernel_from_family(
    family: OrthogonalPolynomialFamily, degree: int
) -> ChristoffelDarbouxKernel:
    """Domain kernel: the bivariate Christoffel-Darboux kernel K_m(x,y).

    K_m(x, y) = sum_{k=0}^m p_k(x) p_k(y) / h_k is returned as the exact
    bivariate coefficient matrix ``coefficients[i][j]`` of ``x^i y^j`` so
    off-diagonal evaluations and downstream composition stay faithful; the
    diagonal specialization K_m(x,x) is a derived evaluation, not the
    kernel itself.
    """
    polys = family.polynomials
    m = degree

    if m >= len(polys):
        raise ValueError(f"degree {m} exceeds family size {len(polys)}")

    coefficients = _kernel_coefficient_matrix(polys, m)
    size = m + 1

    # Kernel entries can exceed the canonical range even when the family
    # is quasi-definite; measure the exact Fraction height before
    # canonical conversion.
    for row in coefficients:
        for value in row:
            if _fraction_exceeds_canonical_limit(value):
                raise ValueError(
                    "Christoffel-Darboux kernel coefficients exceed the "
                    "canonical rational digit limit for this family"
                )
    return ChristoffelDarbouxKernel(
        degree=m,
        coefficients=tuple(
            tuple(_from_fraction(coefficients[i][j]) for j in range(size))
            for i in range(size)
        ),
        variable=family.variable,
        family=family,
    )


def compute_christoffel_darboux(
    request: ChristoffelDarbouxRequest,
) -> ChristoffelDarbouxKernel:
    """MCP adapter: parse one request, call the domain kernel once."""
    return christoffel_darboux_kernel_from_family(request.family, request.degree)


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

            # Admission guarantees every norm feeding an emitted ratio is
            # nonzero.
            betas.append(squared_norm_k / squared_norm_prev)

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
) -> tuple[list[Fraction], list[Fraction] | None]:
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
    # Exact numeric ordering only: stringifying a root would trip CPython's
    # integer-string conversion limit inside the admitted canonical range.
    nodes_frac = []
    for r in sympy.solve(poly_sym, var_symbol):
        # Admission guarantees every root is a distinct rational; keep a
        # typed guard so a backend surprise cannot produce a wrong value.
        if not r.is_Rational:
            raise ValueError("orthogonal polynomial produced a non-rational node")
        nodes_frac.append(Fraction(int(r.p), int(r.q)))
    nodes_frac.sort()
    if len(nodes_frac) != n:
        raise ValueError(f"Expected {n} roots, got {len(nodes_frac)}")

    vandermonde = [[nodes_frac[i] ** k for i in range(n)] for k in range(n)]
    weights = _solve_linear_system(vandermonde, moments[:n])
    return nodes_frac, weights


def _build_quadrature_rule(
    prefix: MomentFunctionalPrefix, order: int
) -> tuple[list[Fraction], list[Fraction]]:
    """Pure nodes+weights construction shared by execution and validation."""
    moments = [_to_fraction(m) for m in prefix.moments]
    p_n = _construct_monic_orthogonal_polynomial(moments, order)
    nodes, weights = _construct_quadrature_rule(p_n, moments, order)
    if weights is None:
        raise ValueError("Vandermonde system is singular")
    return nodes, weights


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
