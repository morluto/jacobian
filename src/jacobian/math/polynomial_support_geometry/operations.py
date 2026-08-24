"""Exact polynomial support geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomial_support_geometry._models import (
    InitialFormRequest,
    NewtonPolytopeRequest,
    SupportRequest,
    WeightProfileRequest,
)
from jacobian.math.polynomial_support_geometry.values import (
    MAX_NEWTON_TERMS,
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _term_pairs(
    polynomial: RationalPolynomial,
) -> list[tuple[Fraction, tuple[int, ...]]]:
    """Extract (coefficient, exponents) from the canonical polynomial value.

    The canonical type already guarantees unique exponent tuples in
    descending lexicographic order with nonzero coefficients; zero terms
    are omitted by construction.
    """
    return [
        (term.coefficient.as_fraction(), tuple(term.exponents))
        for term in polynomial.polynomial.terms
        if term.coefficient.as_fraction() != 0
    ]


def _dot_product(weight: tuple[int, ...], exponents: tuple[int, ...]) -> int:
    return sum(w * e for w, e in zip(weight, exponents, strict=True))


def _require_weighted_polynomial_domain(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> None:
    """Mathematical preconditions shared by native and wire weighted calls.

    The minimum-weight kernels need a nonzero polynomial and a weight per
    declared variable; enforcing this at the domain level keeps the native
    API inside the same mathematical domain as the MCP operation without
    importing transport-size caps.
    """
    if len(weight) != len(polynomial.variables):
        raise ValueError("weight vector length must match variable count")
    if all(term.coefficient.as_fraction() == 0 for term in polynomial.polynomial.terms):
        raise ValueError(
            "the zero polynomial has no weight profile; supply a nonzero polynomial"
        )


def _compute_weight_layers(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> tuple[
    int,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
]:
    """Exact (minimum, minimizing exponents, sorted layers) shared by the
    weight-profile operation and its value validator."""
    _require_weighted_polynomial_domain(polynomial, weight)
    weighted = [
        (_dot_product(weight, tuple(term.exponents)), tuple(term.exponents))
        for term in polynomial.polynomial.terms
    ]
    min_weight = min(w for w, _ in weighted)
    minimizing = tuple(sorted(exp for w, exp in weighted if w == min_weight))

    layer_dict: dict[int, list[tuple[int, ...]]] = {}
    for w, exp in weighted:
        layer_dict.setdefault(w, []).append(exp)

    layers = tuple((w, tuple(sorted(layer_dict[w]))) for w in sorted(layer_dict.keys()))
    return min_weight, minimizing, layers


def _initial_form_terms(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> tuple[tuple[Fraction, tuple[int, ...]], ...]:
    """Exact minimum-weight face terms shared by the initial-form operation
    and its value validator."""
    _require_weighted_polynomial_domain(polynomial, weight)
    weighted = [
        (_dot_product(weight, tuple(term.exponents)), term)
        for term in polynomial.polynomial.terms
    ]
    min_weight = min(w for w, _ in weighted)
    face = [term for w, term in weighted if w == min_weight]
    return tuple(
        (term.coefficient.as_fraction(), tuple(term.exponents)) for term in face
    )


def _solve_convex_membership(
    point: tuple[int, ...],
    others: list[tuple[int, ...]],
) -> tuple[Fraction, ...] | None:
    """Decide exactly whether ``point`` lies in the convex hull of ``others``.

    Solves the feasibility system ``A lambda = b, lambda >= 0`` where each
    column of ``A`` is one other point augmented by a 1 (the convexity row)
    and ``b`` is ``point`` plus 1, using an exact Phase-1 rational simplex
    with Bland's entering and leaving rules (lowest index), which cannot
    cycle and therefore terminates deterministically.

    Returns an exact convex-combination coefficient tuple when feasible
    (replayed against its defining equations before returning), and
    ``None`` when infeasible - infeasibility of this system is precisely
    the certificate that ``point`` is a vertex of the hull.
    """
    from fractions import Fraction

    dimension = len(point)
    count = len(others)
    # Rows: one per coordinate plus the convexity row sum(lambda) = 1.
    rows = [[Fraction(others[j][i]) for j in range(count)] for i in range(dimension)]
    rows.append([Fraction(1)] * count)
    rhs = [Fraction(v) for v in point] + [Fraction(1)]

    total = count + len(rows)
    # Tableau: structural columns, artificial columns, then the RHS.
    tableau = [
        row + [Fraction(1 if i == r else 0) for i in range(len(rows))] + [rhs[r]]
        for r, row in enumerate(rows)
    ]
    basis = [count + i for i in range(len(rows))]
    costs = [Fraction(0)] * count + [Fraction(1)] * len(rows)

    def _price_out(objective: list[Fraction]) -> None:
        for i, basic in enumerate(basis):
            factor = objective[basic]
            if factor:
                row = tableau[i]
                for col in range(total + 1):
                    objective[col] -= factor * row[col]

    objective = [*costs, Fraction(0)]
    _price_out_basis(objective, tableau, basis, total)

    while True:
        entering = _bland_entering_column(objective, total)
        if entering is None:
            break
        leaving_row = _bland_leaving_row(tableau, basis, entering)
        if leaving_row is None:
            raise AssertionError("Phase-1 objective is bounded below by zero")
        _pivot_at(tableau, objective, basis, leaving_row, entering)

    # Phase-1 optimum: zero iff the original system is feasible.
    artificial_rows = [i for i, basic in enumerate(basis) if basic >= count]
    if any(tableau[i][-1] != 0 for i in artificial_rows):
        return None

    solution = _extract_solution(tableau, basis, count)
    _require_membership_witness_replay(rows, rhs, solution)
    return tuple(solution)


def _price_out_basis(
    objective: list[Fraction],
    tableau: list[list[Fraction]],
    basis: list[int],
    total: int,
) -> None:
    """Eliminate basic-variable columns from the Phase-1 objective row."""
    for i, basic in enumerate(basis):
        factor = objective[basic]
        if factor:
            row = tableau[i]
            for col in range(total + 1):
                objective[col] -= factor * row[col]


def _bland_entering_column(objective: list[Fraction], total: int) -> int | None:
    """Lowest-index column with negative reduced cost (Bland's rule)."""
    return next((col for col in range(total) if objective[col] < 0), None)


def _bland_leaving_row(
    tableau: list[list[Fraction]],
    basis: list[int],
    entering: int,
) -> int | None:
    """Minimum-ratio row; ties broken by lowest leaving index (Bland)."""
    leaving_row: int | None = None
    leaving_ratio: Fraction | None = None
    for i, row in enumerate(tableau):
        if row[entering] > 0:
            ratio = row[-1] / row[entering]
            if leaving_ratio is None or ratio < leaving_ratio:
                leaving_row = i
                leaving_ratio = ratio
            elif ratio == leaving_ratio and leaving_row is not None:
                if basis[i] < basis[leaving_row]:
                    leaving_row = i
                    leaving_ratio = ratio
    return leaving_row


def _pivot_at(
    tableau: list[list[Fraction]],
    objective: list[Fraction],
    basis: list[int],
    leaving_row: int,
    entering: int,
) -> None:
    """Normalize the pivot row, eliminate the column everywhere, re-basis."""
    tableau[leaving_row] = [
        value / tableau[leaving_row][entering] for value in tableau[leaving_row]
    ]
    pivot_row = tableau[leaving_row]
    for i, row in enumerate(tableau):
        if i != leaving_row and row[entering]:
            factor = row[entering]
            tableau[i] = [a - factor * b for a, b in zip(row, pivot_row, strict=True)]
    factor = objective[entering]
    if factor:
        objective[:] = [
            a - factor * b for a, b in zip(objective, pivot_row, strict=True)
        ]
    basis[leaving_row] = entering


def _extract_solution(
    tableau: list[list[Fraction]], basis: list[int], count: int
) -> list[Fraction]:
    from fractions import Fraction

    solution = [Fraction(0)] * count
    for i, basic in enumerate(basis):
        if basic < count:
            solution[basic] = tableau[i][-1]
    return solution


def _require_membership_witness_replay(
    rows: list[list[Fraction]],
    rhs: list[Fraction],
    solution: list[Fraction],
) -> None:
    replay = [
        sum(a * value for a, value in zip(row, solution, strict=True)) for row in rows
    ]
    if replay != rhs or any(value < 0 for value in solution):
        raise AssertionError(
            "exact membership witness failed to replay against its equations"
        )


def _is_vertex(point: tuple[int, ...], others: list[tuple[int, ...]]) -> bool:
    """Decide exactly whether ``point`` is a vertex of the convex hull.

    A point is a vertex of the hull of the support precisely when it is not
    a convex combination of the remaining support points. Coordinate
    domination decides most points cheaply; the rest go through the exact
    Phase-1 rational membership kernel. No sampled or heuristic direction
    test is involved.
    """
    if not others:
        return True
    dimension = len(point)
    for axis in range(dimension):
        values = [other[axis] for other in others]
        if point[axis] > max(values) or point[axis] < min(values):
            return True
    return _solve_convex_membership(point, others) is None


def _matrix_rank(matrix: list[list[int]]) -> int:
    """Compute the rank of an integer matrix using Gaussian elimination."""
    if not matrix or not matrix[0]:
        return 0
    mat = [list(row) for row in matrix]
    rows = len(mat)
    rank = 0
    for col in range(len(mat[0])):
        pivot = next((row for row in range(rank, rows) if mat[row][col] != 0), None)
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        _eliminate_column(mat, rank, col)
        rank += 1
        if rank == rows:
            break
    return rank


def _eliminate_column(mat: list[list[int]], rank: int, col: int) -> None:
    """Clear ``col`` below and above the pivot row using integer pivoting."""
    pivot_val = mat[rank][col]
    for row in range(len(mat)):
        if row == rank or mat[row][col] == 0:
            continue
        factor = mat[row][col]
        mat[row] = [
            value * pivot_val - mat[rank][j] * factor
            for j, value in enumerate(mat[row])
        ]


def support_from_polynomial(polynomial: RationalPolynomial) -> PolynomialSupport:
    """Domain kernel: the exponent support of one canonical polynomial.

    Shared by the MCP adapter and the native API so both surfaces apply
    exactly the same mathematical admission and return one value type.
    """
    terms = _term_pairs(polynomial)

    if not terms:
        return PolynomialSupport(
            is_zero=True,
            term_count=0,
            exponents=(),
            coefficients=(),
            variables=polynomial.variables,
        )

    exponents = [t[1] for t in terms]
    coefficients = [t[0] for t in terms]

    n = len(polynomial.variables)
    coord_min = tuple(min(e[i] for _, e in terms) for i in range(n))
    coord_max = tuple(max(e[i] for _, e in terms) for i in range(n))

    return PolynomialSupport(
        is_zero=False,
        term_count=len(terms),
        exponents=tuple(exponents),
        coefficients=tuple(CanonicalRational.from_fraction(c) for c in coefficients),
        variables=polynomial.variables,
        coordinate_min=coord_min,
        coordinate_max=coord_max,
        total_degree_min=min(sum(e) for _, e in terms),
        total_degree_max=max(sum(e) for _, e in terms),
    )


def compute_support(request: SupportRequest) -> PolynomialSupport:
    """MCP adapter: parse one request, call the domain kernel once."""
    return support_from_polynomial(request.polynomial)


def newton_polytope_from_polynomial(
    polynomial: RationalPolynomial,
) -> NewtonPolytope:
    """Domain kernel: the Newton polytope of one canonical polynomial.

    The per-point exact extremality work bound is a mathematical work
    obligation, not a transport cap, so the kernel enforces it for wire
    and native callers alike.
    """
    if len(polynomial.polynomial.terms) > MAX_NEWTON_TERMS:
        raise ValueError(
            f"Newton polytope requests are limited to {MAX_NEWTON_TERMS} terms"
        )
    terms = _term_pairs(polynomial)
    variables = polynomial.variables

    if not terms:
        return NewtonPolytope(
            is_zero=True,
            variables=variables,
            ambient_dimension=len(variables),
            affine_dimension=0,
        )

    exponents = [t[1] for t in terms]
    n = len(variables)

    # Classify every support point exactly: vertices are the points outside
    # the convex hull of the rest.
    vertices = []
    nonextreme = []
    for exp in exponents:
        others = [q for q in exponents if q != exp]
        if _is_vertex(exp, others):
            vertices.append(exp)
        else:
            nonextreme.append(exp)

    # Determine affine dimension
    if len(vertices) <= 1:
        affine_dim = 0
    else:
        # Compute dimension via rank of differences
        first = vertices[0]
        diffs = [[v[j] - first[j] for j in range(n)] for v in vertices[1:]]
        affine_dim = _matrix_rank(diffs)

    return NewtonPolytope(
        is_zero=False,
        variables=variables,
        ambient_dimension=n,
        affine_dimension=affine_dim,
        vertices=tuple(vertices),
        nonextreme=tuple(nonextreme),
        all_support_exponents=tuple(exponents),
    )


def compute_newton_polytope(request: NewtonPolytopeRequest) -> NewtonPolytope:
    """MCP adapter: parse one request, call the domain kernel once."""
    return newton_polytope_from_polynomial(request.polynomial)


def compute_weight_profile(request: WeightProfileRequest) -> PolynomialWeightProfile:
    """Compute the weight profile of a polynomial's support."""
    minimum_weight, minimizing, weight_layers = _compute_weight_layers(
        request.polynomial, request.weight
    )
    return PolynomialWeightProfile(
        polynomial=request.polynomial,
        weight=request.weight,
        minimum_weight=minimum_weight,
        minimizing_exponents=minimizing,
        weight_layers=weight_layers,
    )


def compute_initial_form(request: InitialFormRequest) -> PolynomialFaceData:
    """Compute the initial form of a polynomial under a weight vector."""
    source = request.polynomial
    face_terms = _initial_form_terms(source, request.weight)

    initial_form = RationalPolynomial(
        variables=source.variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(c),
                    exponents=e,
                )
                for c, e in face_terms
            )
        ),
    )

    return PolynomialFaceData(
        polynomial=source,
        weight=request.weight,
        initial_form=initial_form,
    )
