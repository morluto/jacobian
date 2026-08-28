"""Exact moment-functional and orthogonal-polynomial operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight
from jacobian.math.analysis.orthogonal_polynomials._jacobi import (
    JacobiMatrixAdmissionError,
    jacobi_matrix_from_family,
    require_jacobi_matrix_admission,
)
from jacobian.math.analysis.orthogonal_polynomials._models import (
    ChristoffelDarbouxRequest,
    GaussianQuadratureRequest,
    HankelRequest,
    JacobiMatrixRequest,
    OrthogonalPolynomialRequest,
    RecurrenceRequest,
    ShiftedHankelRequest,
)
from jacobian.math.analysis.orthogonal_polynomials.values import (
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


class HankelMatrixAdmissionError(ValueError):
    """A value-based Hankel admission failure with an owner-local code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class MomentsOrthogonalAdmissionError(ValueError):
    """A shared value-based admission failure for moment operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class GaussianQuadratureAdmissionError(MomentsOrthogonalAdmissionError):
    """A value-based Gaussian-quadrature admission failure."""


class ChristoffelDarbouxAdmissionError(MomentsOrthogonalAdmissionError):
    """A value-based Christoffel-Darboux admission failure."""


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


def _require_gram_schmidt_heights_admissible(
    moments: tuple[CanonicalRational, ...], max_degree: int
) -> None:
    """Bound moment heights before exact Gram-Schmidt projection begins."""
    if max_degree == 0:
        return
    side = max_degree + 1
    per_entry = (MAX_CANONICAL_RATIONAL_DIGITS - 2 * side) // (2 * side * (side + 1))
    bound = max(per_entry, 8)
    for value in moments[: 2 * max_degree + 1]:
        if RationalHeight.from_canonical(value).exceeds(bound):
            raise MomentsOrthogonalAdmissionError(
                "gram_schmidt_height",
                f"moment heights exceed the conservative {bound}-digit "
                f"bound for exact degree-{max_degree} Gram-Schmidt; supply "
                "a smaller or better-scaled moment prefix",
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


def require_hankel_matrix_admission(
    prefix: MomentFunctionalPrefix, order: int, *, shifted: bool
) -> None:
    """Validate one canonical prefix for a bounded Hankel construction."""
    from jacobian.math.analysis.orthogonal_polynomials.values import MAX_HANKEL_ORDER

    maximum = MAX_HANKEL_ORDER - int(shifted)
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or not 0 <= order <= maximum
    ):
        raise HankelMatrixAdmissionError(
            "order_out_of_range",
            f"Hankel order must be an integer between 0 and {maximum}",
        )
    needed = 2 * order + 1 + int(shifted)
    if len(prefix.moments) < needed:
        kind = "shifted " if shifted else ""
        raise HankelMatrixAdmissionError(
            "insufficient_moments",
            f"need at least {needed} moments for {kind}order {order}, got "
            f"{len(prefix.moments)}",
        )
    consumed = prefix.moments[1:] if shifted else prefix.moments
    per_entry = MAX_CANONICAL_RATIONAL_DIGITS // ((order + 1) ** 2)
    bound = max(per_entry - 2, 8)
    if any(
        RationalHeight.from_canonical(value).exceeds(bound)
        for value in consumed[: 2 * order + 1]
    ):
        raise HankelMatrixAdmissionError(
            "determinant_height",
            f"moment heights exceed the conservative {bound}-digit bound for "
            f"an exact order-{order} determinant",
        )


def hankel_matrix_from_prefix(
    prefix: MomentFunctionalPrefix, order: int, *, shifted: bool
) -> HankelMomentMatrix:
    """Compute one admitted ordinary or shifted exact Hankel matrix."""
    moments = [_to_fraction(moment) for moment in prefix.moments]
    offset = int(shifted)
    matrix = [
        [moments[i + j + offset] for j in range(order + 1)] for i in range(order + 1)
    ]
    determinant = _rational_det(matrix)
    rank = _rational_rank(matrix)
    return HankelMomentMatrix._from_kernel(
        order=order,
        entries=tuple(
            tuple(_from_fraction(matrix[i][j]) for j in range(order + 1))
            for i in range(order + 1)
        ),
        determinant=_from_fraction(determinant),
        rank=rank,
        variable=prefix.variable,
    )


def compute_hankel_matrix(request: HankelRequest) -> HankelMomentMatrix:
    """MCP adapter: parse one request, call the canonical-prefix kernel."""
    require_hankel_matrix_admission(request.prefix, request.order, shifted=False)
    return hankel_matrix_from_prefix(request.prefix, request.order, shifted=False)


def compute_shifted_hankel(request: ShiftedHankelRequest) -> HankelMomentMatrix:
    """MCP adapter: parse one request, call the canonical-prefix kernel."""
    require_hankel_matrix_admission(request.prefix, request.order, shifted=True)
    return hankel_matrix_from_prefix(request.prefix, request.order, shifted=True)


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

    Shared by request admission, MCP execution, and the native API so no caller
    performs the exact projection twice.
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

    return OrthogonalPolynomialFamily._from_kernel(
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
    from jacobian.math.analysis.orthogonal_polynomials.values import (
        MAX_POLYNOMIAL_DEGREE,
    )

    if not 0 <= max_degree <= MAX_POLYNOMIAL_DEGREE:
        raise MomentsOrthogonalAdmissionError(
            "degree_range",
            f"max_degree must be between 0 and {MAX_POLYNOMIAL_DEGREE}",
        )
    needed = 2 * max_degree + 1
    if len(prefix.moments) < needed:
        raise MomentsOrthogonalAdmissionError(
            "insufficient_moments",
            f"need at least {needed} moments for degree {max_degree}, got "
            f"{len(prefix.moments)}",
        )
    _require_gram_schmidt_heights_admissible(prefix.moments, max_degree)


def compute_orthogonal_polynomials(
    request: OrthogonalPolynomialRequest,
) -> OrthogonalPolynomialFamily:
    """MCP adapter: validate the wire request, then run the shared kernel."""
    try:
        _require_gram_schmidt_admission(request.prefix, request.max_degree)
        moments = [_to_fraction(m) for m in request.prefix.moments]
        return orthogonal_polynomials_from_moments(
            moments, request.max_degree, request.prefix.variable
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prefix", "max_degree"),
            code="moments_orthogonal.family_not_admitted",
            message=str(exc),
        ) from exc


def _require_quasi_definite_family(family: OrthogonalPolynomialFamily) -> None:
    """Recurrence ratios divide by every squared norm except the terminal
    one: ``beta_k = h_k / h_{k-1}`` for k >= 1 uses p_0..p_{n-2} as
    denominators, and ``alpha`` reads adjacent polynomial coefficients. A
    vanishing terminal norm therefore leaves the recurrence exactly defined,
    while any interior zero norm would leak a division failure from execution.

    Degenerate canonical families remain authorable values for composition;
    each consuming operation rejects them at admission. This guard keeps the
    native path on the same admitted domain as ``RecurrenceRequest``.
    """
    polynomials = family.polynomials[:-1]
    if any(term.squared_norm.as_fraction() == 0 for term in polynomials):
        raise MomentsOrthogonalAdmissionError(
            "zero_norm",
            "recurrence coefficients require every non-terminal squared "
            "norm to be nonzero",
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
    _require_quasi_definite_family(family)
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
            raise MomentsOrthogonalAdmissionError(
                "recurrence_height",
                "recurrence coefficients exceed the canonical rational "
                "digit limit for this family",
            )
    return ThreeTermRecurrence._from_kernel(
        alpha=tuple(_from_fraction(a) for a in alphas),
        beta=tuple(_from_fraction(b) for b in betas),
        variable=family.variable,
    )


def compute_recurrence(request: RecurrenceRequest) -> ThreeTermRecurrence:
    """MCP adapter: parse one request, call the domain kernel once."""
    try:
        return recurrence_coefficients_from_family(request.family)
    except MomentsOrthogonalAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("family",),
            code=f"moments_orthogonal.{exc.reason}",
            message=str(exc),
        ) from exc


def _kernel_coefficient_matrix(
    polynomials: tuple[OrthogonalPolynomialTerm, ...], m: int
) -> list[list[Fraction]]:
    """Exact bivariate coefficient matrix of K_m from p_0..p_m.

    Shared by execution and the explicit kernel-value verifier.
    """
    size = m + 1
    coefficients = [[Fraction(0)] * size for _ in range(size)]
    for k in range(m + 1):
        p_k = [_to_fraction(c) for c in polynomials[k].coefficients]
        h_k = _to_fraction(polynomials[k].squared_norm)
        if h_k == 0:
            raise ChristoffelDarbouxAdmissionError(
                "zero_norm",
                f"family polynomial p_{k} has zero squared norm; the "
                "Christoffel-Darboux kernel is undefined",
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
        raise ChristoffelDarbouxAdmissionError(
            "degree_range", f"degree {m} exceeds family size {len(polys)}"
        )

    coefficients = _kernel_coefficient_matrix(polys, m)
    size = m + 1

    # Kernel entries can exceed the canonical range even when the family
    # is quasi-definite; measure the exact Fraction height before
    # canonical conversion.
    for row in coefficients:
        for value in row:
            if _fraction_exceeds_canonical_limit(value):
                raise ChristoffelDarbouxAdmissionError(
                    "coefficient_height",
                    "Christoffel-Darboux kernel coefficients exceed the "
                    "canonical rational digit limit for this family",
                )
    return ChristoffelDarbouxKernel._from_kernel(
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
    try:
        return christoffel_darboux_kernel_from_family(request.family, request.degree)
    except ChristoffelDarbouxAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("family", "degree"),
            code=f"moments_orthogonal.christoffel_darboux.{exc.reason}",
            message=str(exc),
        ) from exc


def compute_jacobi_matrix(request: JacobiMatrixRequest) -> JacobiMatrix:
    """MCP adapter: parse one request, call the canonical-family kernel."""
    try:
        require_jacobi_matrix_admission(request.family)
    except JacobiMatrixAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("family",),
            code=f"moments_orthogonal.jacobi_matrix.{exc.reason}",
            message=str(exc),
        ) from exc
    return jacobi_matrix_from_family(request.family)


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


def require_gaussian_quadrature_admission(
    prefix: MomentFunctionalPrefix, order: int
) -> None:
    """Admit a canonical prefix for one exact rational Gaussian rule."""
    from jacobian.math.analysis.orthogonal_polynomials.values import (
        MAX_QUADRATURE_ORDER,
    )

    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or not 1 <= order <= MAX_QUADRATURE_ORDER
    ):
        raise GaussianQuadratureAdmissionError(
            "order_out_of_range",
            f"quadrature order must be an integer between 1 and {MAX_QUADRATURE_ORDER}",
        )
    try:
        _require_gram_schmidt_heights_admissible(prefix.moments, order)
    except MomentsOrthogonalAdmissionError as exc:
        raise GaussianQuadratureAdmissionError(exc.reason, str(exc)) from None
    needed = 2 * order
    if len(prefix.moments) < needed:
        raise GaussianQuadratureAdmissionError(
            "insufficient_moments",
            f"need at least {needed} moments for quadrature order {order}, got "
            f"{len(prefix.moments)}",
        )

    import sympy

    moments = [_to_fraction(value) for value in prefix.moments]
    coefficients = _construct_monic_orthogonal_polynomial(moments, order)
    variable = sympy.Symbol(prefix.variable)
    polynomial = sum(
        coefficient * variable**index for index, coefficient in enumerate(coefficients)
    )
    _, factors = sympy.factor_list(polynomial)
    if any(
        sympy.degree(factor, variable) != 1 or multiplicity != 1
        for factor, multiplicity in factors
    ):
        raise GaussianQuadratureAdmissionError(
            "rational_nodes",
            f"quadrature order {order} requires p_{order} to split into "
            "distinct linear factors over QQ so every node is an exact rational; "
            "this moment prefix yields algebraic or repeated nodes",
        )
    nodes, weights = _build_quadrature_rule(prefix, order)
    if any(_fraction_exceeds_canonical_limit(value) for value in (*nodes, *weights)):
        raise GaussianQuadratureAdmissionError(
            "quadrature_height",
            "derived quadrature nodes or weights exceed the canonical rational "
            "digit limit; supply a moment prefix whose exact rule stays "
            "representable",
        )
    if any(weight <= 0 for weight in weights):
        raise GaussianQuadratureAdmissionError(
            "positive_weights",
            "quadrature admission requires strictly positive weights; this "
            "moment prefix yields a nonpositive weight",
        )


def gaussian_quadrature_rule_from_prefix(
    prefix: MomentFunctionalPrefix, order: int
) -> GaussianQuadratureRule:
    """Construct one admitted exact Gaussian quadrature rule."""
    nodes_frac, weights = _build_quadrature_rule(prefix, order)
    for k in range(2 * order):
        approximation = sum(
            weights[index] * nodes_frac[index] ** k for index in range(order)
        )
        if approximation != _to_fraction(prefix.moments[k]):
            raise ValueError(f"quadrature is not exact at degree {k}")
    return GaussianQuadratureRule._from_kernel(
        order=order,
        nodes=tuple(
            QuadratureNode(node=_from_fraction(node), weight=_from_fraction(weight))
            for node, weight in zip(nodes_frac, weights, strict=True)
        ),
        variable=prefix.variable,
        prefix=prefix,
    )


def compute_gaussian_quadrature(
    request: GaussianQuadratureRequest,
) -> GaussianQuadratureRule:
    """MCP adapter: parse one request, call the canonical-prefix kernel."""
    try:
        require_gaussian_quadrature_admission(request.prefix, request.order)
        return gaussian_quadrature_rule_from_prefix(request.prefix, request.order)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prefix", "order"),
            code="moments_orthogonal.quadrature_not_admitted",
            message=str(exc),
        ) from exc


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


__all__ = [
    "compute_christoffel_darboux",
    "compute_gaussian_quadrature",
    "compute_hankel_matrix",
    "compute_jacobi_matrix",
    "compute_orthogonal_polynomials",
    "compute_recurrence",
    "compute_shifted_hankel",
]
