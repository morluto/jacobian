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
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)


def _extract_terms(
    request: SupportRequest | NewtonPolytopeRequest | WeightProfileRequest | InitialFormRequest,
) -> list[tuple[Fraction, tuple[int, ...]]]:
    """Extract (coefficient, exponents) from request terms."""
    result = []
    for term in request.terms:
        coeff = term["coefficient"]
        if isinstance(coeff, dict):
            num = int(coeff.get("num", "0"))
            den = int(coeff.get("den", "1"))
        else:
            num, den = int(coeff), 1
        frac = Fraction(num, den)
        if frac != 0:
            exponents = tuple(int(e) for e in term["exponents"])
            result.append((frac, exponents))
    return result


def _dot_product(weight: tuple[int, ...], exponents: tuple[int, ...]) -> int:
    return sum(w * e for w, e in zip(weight, exponents, strict=True))


def _is_extreme(
    point: tuple[int, ...],
    others: list[tuple[int, ...]],
) -> bool:
    """Check if a point is a vertex of the convex hull."""
    # Simple approach: a point is extreme if there exists a direction
    # in which it is the unique maximizer
    n = len(point)
    for direction in _generate_directions(n, len(others)):
        values = {_dot_product(direction, p) for p in others}
        max_val = max(values)
        point_val = _dot_product(direction, point)
        if point_val == max_val:
            count_max = sum(1 for p in others if _dot_product(direction, p) == max_val)
            if count_max == 1:
                return True
    return False


def _generate_directions(n: int, count: int):
    """Generate test directions for extreme point detection."""
    directions = []
    # Standard basis directions
    for i in range(n):
        d = [0] * n
        d[i] = 1
        directions.append(tuple(d))
        d = [0] * n
        d[i] = -1
        directions.append(tuple(d))
    # Random-ish directions based on prime numbers
    for i in range(min(count * 2, 20)):
        d = tuple(((i + j + 1) * 7) % 11 - 5 for j in range(n))
        if any(v != 0 for v in d):
            directions.append(d)
    return directions


def compute_support(request: SupportRequest) -> PolynomialSupport:
    """Compute the exponent support of a polynomial."""
    terms = _extract_terms(request)

    if not terms:
        return PolynomialSupport(
            is_zero=True,
            term_count=0,
            exponents=(),
            coefficients=(),
            variables=request.variables,
        )

    exponents = [t[1] for t in terms]
    coefficients = [t[0] for t in terms]

    n = len(request.variables)
    coord_min = tuple(min(e[i] for _, e in terms) for i in range(n))
    coord_max = tuple(max(e[i] for _, e in terms) for i in range(n))

    return PolynomialSupport(
        is_zero=False,
        term_count=len(terms),
        exponents=tuple(exponents),
        coefficients=tuple(
            CanonicalRational(num=str(c.numerator), den=str(c.denominator))
            for c in coefficients
        ),
        variables=request.variables,
        coordinate_min=coord_min,
        coordinate_max=coord_max,
        total_degree_min=min(sum(e) for _, e in terms),
        total_degree_max=max(sum(e) for _, e in terms),
    )


def compute_newton_polytope(request: NewtonPolytopeRequest) -> NewtonPolytope:
    """Compute the Newton polytope of a polynomial."""
    terms = _extract_terms(request)

    if not terms:
        return NewtonPolytope(
            is_zero=True,
            ambient_dimension=len(request.variables),
            affine_dimension=0,
        )

    exponents = [t[1] for t in terms]
    n = len(request.variables)

    # Find vertices (extreme points) of the convex hull
    vertices = []
    nonextreme = []
    for i, exp in enumerate(exponents):
        others = exponents[:]
        # Check if this point is extreme
        all_points = set(exponents)
        # A point is a vertex if it cannot be written as a convex combination
        # of other points
        if _is_vertex(exp, exponents):
            vertices.append(exp)
        else:
            nonextreme.append(exp)

    # Determine affine dimension
    if len(vertices) <= 1:
        affine_dim = 0
    else:
        # Compute dimension via rank of differences
        first = vertices[0]
        diffs = [
            [v[j] - first[j] for j in range(n)]
            for v in vertices[1:]
        ]
        affine_dim = _matrix_rank(diffs)

    return NewtonPolytope(
        is_zero=False,
        ambient_dimension=n,
        affine_dimension=affine_dim,
        vertices=tuple(vertices),
        nonextreme=tuple(nonextreme),
        all_support_exponents=tuple(exponents),
    )


def _is_vertex(point: tuple[int, ...], all_points: list[tuple[int, ...]]) -> bool:
    """Check if a point is a vertex of the convex hull.

    A point is a vertex iff there exists a direction d such that
    <d, point> strictly maximizes over all points.
    """
    others = [p for p in all_points if p != point]
    if not others:
        return True

    n = len(point)
    # Try standard basis directions
    for i in range(n):
        for sign in [1, -1]:
            d = [0] * n
            d[i] = sign
            point_val = sum(d[j] * point[j] for j in range(n))
            others_vals = [sum(d[j] * p[j] for j in range(n)) for p in others]
            if all(point_val > ov for ov in others_vals):
                return True

    # Try directions formed by pairs of points
    import itertools
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    d = [0] * n
                    d[i] = s1
                    d[j] = s2
                    point_val = sum(d[k] * point[k] for k in range(n))
                    others_vals = [sum(d[k] * p[k] for k in range(n)) for p in others]
                    if all(point_val > ov for ov in others_vals):
                        return True

    # Try random-ish directions
    for comb in itertools.product(range(-3, 4), repeat=min(n, 2)):
        d = list(comb) + [0] * (n - len(comb))
        if all(v == 0 for v in d):
            continue
        point_val = sum(d[k] * point[k] for k in range(n))
        others_vals = [sum(d[k] * p[k] for k in range(n)) for p in others]
        if all(point_val > ov for ov in others_vals):
            return True

    return False


def _matrix_rank(matrix: list[list[int]]) -> int:
    """Compute the rank of an integer matrix using Gaussian elimination."""
    if not matrix or not matrix[0]:
        return 0
    mat = [list(row) for row in matrix]
    rows = len(mat)
    cols = len(mat[0])
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
                factor = mat[row][col]
                pivot_val = mat[rank][col]
                for j in range(cols):
                    mat[row][j] = mat[row][j] * pivot_val - mat[rank][j] * factor
        rank += 1
        if rank == rows:
            break
    return rank


def compute_weight_profile(request: WeightProfileRequest) -> PolynomialWeightProfile:
    """Compute the weight profile of a polynomial's support."""
    terms = _extract_terms(request)

    if not terms:
        return PolynomialWeightProfile(
            minimum_weight=0,
            minimizing_exponents=(),
            weight_layers=(),
        )

    weight = request.weight
    weighted = [
        (_dot_product(weight, exp), exp) for _, exp in terms
    ]

    min_weight = min(w for w, _ in weighted)
    minimizing = tuple(sorted(exp for w, exp in weighted if w == min_weight))

    # Build weight layers
    layer_dict: dict[int, list[tuple[int, ...]]] = {}
    for w, exp in weighted:
        if w not in layer_dict:
            layer_dict[w] = []
        layer_dict[w].append(exp)

    weight_layers = tuple(
        (w, tuple(sorted(layer_dict[w])))
        for w in sorted(layer_dict.keys())
    )

    return PolynomialWeightProfile(
        minimum_weight=min_weight,
        minimizing_exponents=minimizing,
        weight_layers=weight_layers,
    )


def compute_initial_form(request: InitialFormRequest) -> PolynomialFaceData:
    """Compute the initial form of a polynomial under a weight vector."""
    terms = _extract_terms(request)
    weight = request.weight

    if not terms:
        return PolynomialFaceData(
            face_exponents=(),
            face_coefficients=(),
            face_polynomial_terms=(),
        )

    # Compute weights for all terms
    weighted = [(_dot_product(weight, exp), coeff, exp) for coeff, exp in terms]
    min_weight = min(w for w, _, _ in weighted)

    # Select terms at minimum weight
    face_terms = [(c, e) for w, c, e in weighted if w == min_weight]

    face_exponents = tuple(t[1] for t in face_terms)
    face_coefficients = [
        CanonicalRational(num=str(c.numerator), den=str(c.denominator))
        for c, _ in face_terms
    ]

    # Build polynomial term dicts
    face_polynomial_terms = tuple(
        {
            "coefficient": {"num": str(c.numerator), "den": str(c.denominator)},
            "exponents": list(e),
        }
        for c, e in face_terms
    )

    return PolynomialFaceData(
        face_exponents=face_exponents,
        face_coefficients=tuple(face_coefficients),
        face_polynomial_terms=face_polynomial_terms,
    )
