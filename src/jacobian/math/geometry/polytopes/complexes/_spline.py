"""Exact piecewise-polynomial compatibility and bounded spline spaces."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import comb
from typing import Any, NoReturn

import sympy as sp

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    PieceAssignment,
    PieceCompatibilityRow,
    PiecewiseEvaluationResult,
    PiecewisePolynomialResult,
    PolytopalComplexClosureResult,
    SplineSpaceResult,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.geometry.polytopes.values import Vertex
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("piecewise",), code=f"polytopal_complex.{code}", message=message
    )


def _poly_symbols(poly: RationalPolynomial) -> tuple[Any, ...]:
    return sp.symbols(" ".join(poly.variables), seq=True) if poly.variables else ()


def _to_poly(poly: RationalPolynomial, symbols: tuple[Any, ...]) -> Any:
    expr = 0
    for term in poly.polynomial.terms:
        value = sp.Rational(term.coefficient.num, term.coefficient.den)
        monomial = sp.Integer(1)
        for symbol, exponent in zip(symbols, term.exponents, strict=True):
            monomial *= symbol**exponent
        expr += value * monomial
    return sp.Poly(expr, *symbols, domain=sp.QQ)


def _from_poly(poly: sp.Poly, variables: tuple[str, ...]) -> RationalPolynomial:
    terms = []
    for exponents, coefficient in sorted(poly.terms(), reverse=True):
        coeff = Fraction(int(coefficient.p), int(coefficient.q))
        if coeff:
            terms.append(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coeff),
                    exponents=tuple(int(e) for e in exponents),
                )
            )
    return RationalPolynomial(
        variables=variables, polynomial=SparseRationalPolynomial(terms=tuple(terms))
    )


def _point_values(face: Any) -> tuple[Fraction, ...]:
    return tuple(c.as_fraction() for c in face.coordinates)


def _admit_complex(  # noqa: C901
    value: PolytopalComplexClosureResult,
) -> PolytopalComplexClosureResult:
    """Re-establish the complete closure relation at consumer boundaries.

    Closure results are intentionally built with a trusted kernel constructor,
    so native callers can otherwise forge a structurally valid but unrelated
    face ledger.  Reconstructing the source cells from the retained canonical
    vertices checks the complete face, incidence, intersection, and Euler
    claims without trusting any derived field.
    """

    if not isinstance(value, PolytopalComplexClosureResult):
        _reject("complex_type", "expected a canonical polytopal complex")
    try:
        payload = value.model_dump(mode="python")
        canonical = PolytopalComplexClosureResult.model_validate(payload)
    except Exception:
        _reject("complex_malformed", "polytopal complex carriers are malformed")
    if canonical.model_dump(mode="python") != payload:
        _reject("complex_malformed", "polytopal complex must be canonical")
    try:
        source_count = len(canonical.source_cell_map)
        if source_count > 64:
            _reject("complex_source", "polytopal source presentation is too large")
        source_rows = {
            row.source_index: row.cell_id for row in canonical.source_cell_map
        }
        if tuple(sorted(source_rows)) != tuple(range(source_count)):
            _reject("complex_source", "source-cell transport must be complete")
        cells_by_id = {cell.cell_id: cell for cell in canonical.maximal_cells}
        if set(source_rows.values()) != set(cells_by_id):
            _reject("complex_source", "source-cell transport names unknown cells")
        source_cells: list[RationalVPolytope | None] = [None] * source_count
        space = RationalCoordinateSpace(axes=tuple(canonical.space.axes))
        for cell in canonical.maximal_cells:
            if tuple(sorted(cell.source_indices)) != cell.source_indices or any(
                index < 0 or index >= source_count for index in cell.source_indices
            ):
                _reject("complex_source", "cell provenance indices are malformed")
            vertices = tuple(
                RationalPolytopeVertex(
                    vertex_id=f"v{position}", coordinates=point.coordinates
                )
                for position, point in enumerate(cell.vertices)
            )
            polytope = RationalVPolytope(space=space, vertices=vertices)
            for index in cell.source_indices:
                if source_cells[index] is not None:
                    _reject("complex_source", "source indices must be unique")
                source_cells[index] = polytope
        if any(cell is None for cell in source_cells):
            _reject("complex_source", "cell provenance must cover every source row")
        # Local import avoids the operations/_spline import cycle.
        from jacobian.math.geometry.polytopes.complexes.operations import (
            polytopal_complex_closure,
        )

        rebuilt = polytopal_complex_closure(tuple(source_cells))  # type: ignore[arg-type]
    except OperationDomainValidationError:
        raise
    except OperationResourceAdmissionError:
        raise
    except Exception:
        _reject(
            "complex_malformed", "polytopal complex closure could not be re-established"
        )
    if rebuilt.model_dump(mode="python") != payload:
        _reject(
            "complex_source_mismatch",
            "faces, cells, and closure claims must describe one canonical complex",
        )
    return canonical


def _affine_ideal(face: Any, symbols: tuple[Any, ...]) -> list[Any]:
    points = [_point_values(vertex) for vertex in face.vertices]
    dimension = len(points[0])
    if len(points) == 1:
        normals: list[tuple[Fraction, ...]] = [
            tuple(Fraction(1 if i == j else 0) for i in range(dimension))
            for j in range(dimension)
        ]
    else:
        base = points[0]
        matrix = sp.Matrix(
            [
                [
                    sp.Rational(value.numerator, value.denominator)
                    for value in (point[i] - base[i] for i in range(dimension))
                ]
                for point in points[1:]
            ]
        )
        normals = [
            tuple(Fraction(int(v.p), int(v.q)) for v in vector)
            for vector in matrix.nullspace()
        ]
    return [
        sum(
            sp.Rational(normal[i].numerator, normal[i].denominator)
            * (
                symbols[i]
                - sp.Rational(points[0][i].numerator, points[0][i].denominator)
            )
            for i in range(dimension)
        )
        for normal in normals
    ]


MAX_PIECE_COMPATIBILITY_WORK = 50_000_000
MAX_PIECE_RESULT_DIGITS = 64 * 1024 * 1024
MAX_RATIONAL_SCALAR_DIGITS = 2 * 32_768


def _admit_pieces(
    complex_value: PolytopalComplexClosureResult,
    pieces: tuple[PieceAssignment, ...],
) -> PolytopalComplexClosureResult:
    complex_value = _admit_complex(complex_value)
    if (
        not isinstance(pieces, tuple)
        or not pieces
        or any(not isinstance(row, PieceAssignment) for row in pieces)
    ):
        _reject("piece_type", "pieces must be canonical cell-polynomial assignments")
    try:
        canonical_pieces = tuple(
            PieceAssignment.model_validate(row.model_dump(mode="python"))
            for row in pieces
        )
    except Exception:
        _reject("piece_malformed", "piece assignments must be canonical values")
    if any(
        row.model_dump(mode="python") != source.model_dump(mode="python")
        for row, source in zip(canonical_pieces, pieces, strict=True)
    ):
        _reject("piece_malformed", "piece assignments must be canonical values")
    cells = tuple(cell.cell_id for cell in complex_value.maximal_cells)
    supplied = tuple(sorted(row.cell_id for row in canonical_pieces))
    if supplied != cells or len(set(supplied)) != len(supplied):
        _reject(
            "piece_axis",
            "exactly one polynomial piece is required for every canonical maximal cell",
        )
    variables = canonical_pieces[0].polynomial.variables
    if variables != tuple(complex_value.space.axes):
        _reject(
            "piece_ring", "polynomial variables must equal the complex coordinate axes"
        )
    if any(row.polynomial.variables != variables for row in canonical_pieces):
        _reject(
            "piece_ring", "all pieces must use one identical ordered polynomial ring"
        )
    if len(canonical_pieces) > 16:
        raise OperationResourceAdmissionError(
            location=("pieces",),
            code="polytopal_complex.piece_count",
            message="too many pieces",
        )
    term_count = max(len(row.polynomial.polynomial.terms) for row in canonical_pieces)
    dimension = len(complex_value.space.axes)
    max_degree = max(
        (
            sum(term.exponents)
            for row in canonical_pieces
            for term in row.polynomial.polynomial.terms
        ),
        default=0,
    )
    # Restriction to an affine face can expand a monomial.  Admit a complete
    # carrier-sized bound for every reduced polynomial, not just its source
    # term count, before Groebner reduction starts.
    reduced_term_bound = term_count * comb(max_degree + dimension, dimension)
    if reduced_term_bound > 4_096:
        raise OperationResourceAdmissionError(
            location=("pieces",),
            code="polytopal_complex.compatibility_terms",
            message="reduced compatibility polynomials exceed the admitted term envelope",
        )
    pair_count = sum(
        len(face.maximal_cell_ids) * (len(face.maximal_cell_ids) - 1) // 2
        for face in complex_value.faces
    )
    # Charge both polynomial conversion and the exact reduction, then admit
    # the complete ledger before creating a SymPy expression.
    work = pair_count * max(reduced_term_bound, 1) * (dimension + 1) ** 3
    if work > MAX_PIECE_COMPATIBILITY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.compatibility_work",
            message="piece compatibility reduction exceeds the admitted envelope",
        )
    row_digits = (2 * reduced_term_bound + 8) * MAX_RATIONAL_SCALAR_DIGITS
    estimated_digits = pair_count * row_digits
    if estimated_digits > MAX_PIECE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.compatibility_output",
            message="piece compatibility ledger exceeds the admitted output envelope",
        )
    return complex_value


def piecewise_polynomial_from_maximal_pieces(
    complex_value: PolytopalComplexClosureResult,
    pieces: tuple[PieceAssignment, ...],
) -> PiecewisePolynomialResult:
    complex_value = _admit_pieces(complex_value, pieces)
    by_id = {row.cell_id: row.polynomial for row in pieces}
    symbols = _poly_symbols(pieces[0].polynomial)
    polynomials = {
        cell_id: _to_poly(polynomial, symbols) for cell_id, polynomial in by_id.items()
    }
    rows = []
    first_obstruction: tuple[str, RationalPolynomial] | None = None
    for face in complex_value.faces:
        supports = tuple(sorted(face.maximal_cell_ids))
        if len(supports) < 2 or face.dimension < 0:
            continue
        ideal = _affine_ideal(face, symbols)
        basis = sp.groebner(ideal, *symbols, domain=sp.QQ)
        for first, second in combinations(supports, 2):
            difference = polynomials[first] - polynomials[second]
            remainder = basis.reduce(difference.as_expr())[1]
            reduced = _from_poly(
                sp.Poly(remainder, *symbols, domain=sp.QQ),
                tuple(complex_value.space.axes),
            )
            compatible = not remainder
            row = PieceCompatibilityRow(
                first_cell_id=first,
                second_cell_id=second,
                face_id=face.face_id,
                reduced_difference=reduced,
                compatible=compatible,
            )
            rows.append(row)
            if not compatible and first_obstruction is None:
                first_obstruction = (face.face_id, reduced)
    ordered_pieces = tuple(sorted(pieces, key=lambda x: x.cell_id))
    if first_obstruction is not None:
        face_id, difference = first_obstruction
        return PiecewisePolynomialResult(
            complex=complex_value,
            pieces=ordered_pieces,
            compatibility=tuple(rows),
            status="INCOMPATIBLE",
            obstruction_face_id=face_id,
            obstruction_difference=difference,
        )
    return PiecewisePolynomialResult(
        complex=complex_value,
        pieces=ordered_pieces,
        compatibility=tuple(rows),
        status="COMPATIBLE",
    )


def _contains(cell: Any, point: tuple[Fraction, ...]) -> bool:
    vertices = tuple(Vertex(coordinates=vertex.coordinates) for vertex in cell.vertices)
    profile = facet_incidence(vertices, cell.dimension)
    return all(
        sum(
            a.as_fraction() * x
            for a, x in zip(facet.halfspace.coefficients, point, strict=True)
        )
        <= facet.halfspace.offset.as_fraction()
        for facet in profile.facets
    )


def piecewise_polynomial_evaluate(
    function: PiecewisePolynomialResult, point: ComplexPoint
) -> PiecewiseEvaluationResult:
    if not isinstance(function, PiecewisePolynomialResult) or not isinstance(
        point, ComplexPoint
    ):
        _reject("evaluation_type", "evaluation requires a canonical function and point")
    function_complex = _admit_pieces(function.complex, function.pieces)
    if function.status != "COMPATIBLE":
        _reject(
            "incompatible_function",
            "an incompatible piecewise function cannot be evaluated",
        )
    canonical_function = piecewise_polynomial_from_maximal_pieces(
        function_complex, function.pieces
    )
    if canonical_function != function:
        _reject(
            "function_source_mismatch",
            "piecewise status and compatibility ledger must match the source pieces",
        )
    function = canonical_function
    try:
        point = ComplexPoint.model_validate(point.model_dump(mode="python"))
    except Exception:
        _reject("point_shape", "evaluation point must be a canonical complex point")
    if len(point.coordinates) != len(function_complex.space.axes):
        _reject("point_axis", "evaluation point must use the complex coordinate axis")
    values = tuple(c.as_fraction() for c in point.coordinates)
    cells = tuple(
        cell for cell in function_complex.maximal_cells if _contains(cell, values)
    )
    if not cells:
        return PiecewiseEvaluationResult(
            function=function,
            status="OUTSIDE_SUPPORT",
            point=point,
            containing_cell_ids=(),
            value=None,
        )
    by_id = {row.cell_id: row.polynomial for row in function.pieces}
    symbols = _poly_symbols(function.pieces[0].polynomial)
    evaluated = []
    substitutions = {
        symbol: sp.Rational(value.numerator, value.denominator)
        for symbol, value in zip(symbols, values, strict=True)
    }
    for cell in cells:
        _admit_evaluation_growth(by_id[cell.cell_id], values)
        poly = _to_poly(by_id[cell.cell_id], symbols)
        result = poly.as_expr().subs(substitutions)
        rational = sp.Rational(result)
        evaluated.append(Fraction(int(rational.p), int(rational.q)))
    if any(value != evaluated[0] for value in evaluated[1:]):
        raise ArithmeticError("compatible pieces disagree at a common boundary point")
    return PiecewiseEvaluationResult(
        function=function,
        status="EVALUATED",
        point=point,
        containing_cell_ids=tuple(cell.cell_id for cell in cells),
        value=CanonicalRational.from_fraction(evaluated[0]),
    )


def _integer_digits(value: int) -> int:
    return len(str(abs(value)))


def _admit_evaluation_growth(
    polynomial: RationalPolynomial, values: tuple[Fraction, ...]
) -> None:
    """Bound exact substitution and rational aggregation before SymPy work."""

    term_bounds: list[tuple[int, int]] = []
    for term in polynomial.polynomial.terms:
        if any(
            value.numerator == 0 and exponent > 0
            for value, exponent in zip(values, term.exponents, strict=True)
        ):
            continue
        numerator_digits = _integer_digits(term.coefficient.num)
        denominator_digits = _integer_digits(term.coefficient.den)
        for value, exponent in zip(values, term.exponents, strict=True):
            numerator_digits += exponent * _integer_digits(value.numerator)
            denominator_digits += exponent * _integer_digits(value.denominator)
        term_bounds.append((numerator_digits, denominator_digits))
    if not term_bounds:
        return
    denominator_digits = sum(bound[1] for bound in term_bounds)
    numerator_digits = (
        max(
            numerator + denominator_digits - denominator
            for numerator, denominator in term_bounds
        )
        + len(str(len(term_bounds)))
        + 1
    )
    if max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("point",),
            code="polytopal_complex.evaluation_growth",
            message=(
                "piecewise polynomial evaluation exceeds the canonical rational "
                "output envelope"
            ),
        )


def _monomials(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            (
                exponents
                for exponents in product(range(degree + 1), repeat=dimension)
                if sum(exponents) <= degree
            ),
            reverse=True,
        )
    )


def _linear_form(face: Any, symbols: tuple[Any, ...]) -> Any:
    ideal = _affine_ideal(face, symbols)
    if len(ideal) != 1:
        _reject(
            "spline_facet",
            "every interior spline interface must be a codimension-one facet",
        )
    return sp.Poly(ideal[0], *symbols, domain=sp.QQ)


MAX_SPLINE_CONSTRAINT_CELLS = 1_048_576
MAX_SPLINE_RESULT_CELLS = 1_000_000
MAX_SPLINE_RESULT_DIGITS = 64 * 1024 * 1024
MAX_SPLINE_SCALAR_DIGITS = 2 * 32_768


def _admit_spline(
    complex_value: PolytopalComplexClosureResult, degree: int, smoothness: int
) -> tuple[PolytopalComplexClosureResult, tuple[Any, ...], int]:
    complex_value = _admit_complex(complex_value)
    if type(degree) is not int or type(smoothness) is not int:
        _reject(
            "spline_type",
            "spline degree and smoothness must be exact integers",
        )
    if degree < 0 or degree > 12 or smoothness < -1 or smoothness > 4:
        _reject(
            "spline_parameters",
            "degree must be in [0, 12] and smoothness must be in [-1, 4]",
        )
    if any(
        not isinstance(axis, str) or not axis.strip()
        for axis in complex_value.space.axes
    ):
        _reject("spline_axes", "complex coordinate axes must be nonempty symbols")
    cells = tuple(sorted(complex_value.maximal_cells, key=lambda cell: cell.cell_id))
    dimension = len(complex_value.space.axes)
    monomial_count = comb(dimension + degree, degree)
    width = len(cells) * monomial_count
    if width > 4096:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_width",
            message="spline coefficient axis exceeds the admitted envelope",
        )
    interface_count = sum(
        1
        for face in complex_value.faces
        if len(face.maximal_cell_ids) == 2
        and face.dimension == dimension - 1
        and smoothness >= 0
    )
    # Each interface contributes at most one dense row per monomial.  Reserve
    # the full row-by-column materialization before SymPy or RationalMatrix
    # allocation; this bounds both the exact constraint output and elimination.
    constraint_cells = interface_count * monomial_count * width
    if constraint_cells > MAX_SPLINE_CONSTRAINT_CELLS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.spline_constraints",
            message="spline compatibility matrix exceeds the admitted envelope",
        )
    # The unconstrained case has a width-by-width nullspace basis.  Admit the
    # complete exact result (both matrices and all rational components) before
    # SymPy materializes a dense basis.
    result_cells = constraint_cells + width * width
    estimated_digits = result_cells * MAX_SPLINE_SCALAR_DIGITS
    if result_cells > MAX_SPLINE_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_result_cells",
            message="spline exact result has too many matrix cells",
        )
    if estimated_digits > MAX_SPLINE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_result_output",
            message="spline exact result exceeds the admitted digit envelope",
        )
    return complex_value, cells, width


def spline_space(
    complex_value: PolytopalComplexClosureResult, degree: int, smoothness: int
) -> SplineSpaceResult:
    complex_value, cells, width = _admit_spline(complex_value, degree, smoothness)
    dimension = len(complex_value.space.axes)
    monomials = _monomials(dimension, degree)
    coefficient_axis = tuple(
        (cell.cell_id, monomial) for cell in cells for monomial in monomials
    )
    if len(coefficient_axis) != width:
        raise ArithmeticError("spline coefficient-axis admission mismatch")
    symbols = tuple(sp.Symbol(axis) for axis in complex_value.space.axes)
    constraint_rows: list[list[Fraction]] = []
    cell_index = {cell.cell_id: i for i, cell in enumerate(cells)}
    for face in complex_value.faces:
        supports = tuple(sorted(face.maximal_cell_ids))
        if len(supports) != 2 or face.dimension != dimension - 1 or smoothness < 0:
            continue
        ell = _linear_form(face, symbols)
        divisor = sp.Poly(ell.as_expr() ** (smoothness + 1), *symbols, domain=sp.QQ)
        remainders = []
        for cell_id, sign in ((supports[0], 1), (supports[1], -1)):
            for monomial in monomials:
                expr = sp.Poly(
                    sp.prod(symbols[i] ** monomial[i] for i in range(dimension)),
                    *symbols,
                    domain=sp.QQ,
                )
                rem = expr.div(divisor)[1] * sign
                remainders.append((cell_id, monomial, rem))
        support_coeffs: set[tuple[int, ...]] = set()
        for _, _, rem in remainders:
            support_coeffs.update(exp for exp, _ in rem.terms())
        for exponent in sorted(support_coeffs, reverse=True):
            row = [Fraction(0) for _ in range(width)]
            for cell_id, monomial, rem in remainders:
                coeff = rem.coeff_monomial(exponent)
                if coeff:
                    col = cell_index[cell_id] * len(monomials) + monomials.index(
                        monomial
                    )
                    row[col] += Fraction(int(coeff.p), int(coeff.q))
            if any(row):
                constraint_rows.append(row)
    matrix = RationalMatrix(
        row_count=len(constraint_rows),
        column_count=width,
        entries=tuple(
            tuple(CanonicalRational.from_fraction(v) for v in row)
            for row in constraint_rows
        ),
    )
    # Retain the coefficient-axis width even when there are no interface
    # constraints.  ``sp.Matrix([])`` is 0x0 and would silently erase the
    # polynomial coefficient domain; a shaped zero-row matrix has the correct
    # nullspace (the width standard basis vectors).
    smatrix = (
        sp.zeros(0, width)
        if not constraint_rows
        else sp.Matrix(
            [
                [sp.Rational(v.numerator, v.denominator) for v in row]
                for row in constraint_rows
            ]
        )
    )
    rank = int(smatrix.rank())
    basis = smatrix.nullspace() if width else []
    basis_rows = []
    for vector in basis:
        basis_rows.append(
            tuple(
                CanonicalRational.from_fraction(Fraction(int(v.p), int(v.q)))
                for v in vector
            )
        )
    nullity = width - rank
    if len(basis_rows) != nullity:
        raise ArithmeticError("spline nullspace basis dimension mismatch")
    nullspace = RationalMatrix(
        row_count=len(basis_rows), column_count=width, entries=tuple(basis_rows)
    )
    return SplineSpaceResult(
        complex=complex_value,
        degree=degree,
        smoothness=smoothness,
        coefficient_axis=coefficient_axis,
        compatibility_matrix=matrix,
        rank=rank,
        nullity=nullity,
        nullspace_basis=nullspace,
    )


__all__ = [
    "piecewise_polynomial_evaluate",
    "piecewise_polynomial_from_maximal_pieces",
    "spline_space",
]
