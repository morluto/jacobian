"""Exact piecewise-polynomial compatibility and bounded spline spaces."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import comb
from typing import Any, NoReturn

import sympy as sp

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import decimal_digit_width
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
    MAX_COMPLEX_CELLS,
    ComplexPoint,
    PieceAssignment,
    PieceCompatibilityRow,
    PiecewiseEvaluationResult,
    PiecewiseFacetSmoothness,
    PiecewisePolynomialAdditionRequest,
    PiecewisePolynomialMultiplicationRequest,
    PiecewisePolynomialResult,
    PiecewiseSmoothnessRequest,
    PiecewiseSmoothnessResult,
    PolytopalComplexClosureResult,
    SplineCoordinatesRequest,
    SplineCoordinatesResult,
    SplineDimensionRequest,
    SplineDimensionResult,
    SplineEvaluationRequest,
    SplineEvaluationResult,
    SplineSpaceResult,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.geometry.polytopes.values import Vertex
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials._multiply_kernel import rational_polynomial_multiply
from jacobian.math.polynomials._multiply_models import (
    MAX_MULTIPLY_PRODUCT_WORK,
    _maximum_product_coefficient_digits,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
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
                    vertex_id=f"v{position:03d}", coordinates=point.coordinates
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
MAX_PIECE_MULTIPLICATION_WORK = 1_000_000
MAX_PIECE_MULTIPLICATION_OUTPUT_DIGITS = 10 * 1024 * 1024
"""Decimal-digit ceiling summed over all stored rational components of a product."""
MAX_RATIONAL_SCALAR_DIGITS = 2 * 32_768


def _admit_pieces(
    complex_value: PolytopalComplexClosureResult,
    pieces: tuple[PieceAssignment, ...],
) -> PolytopalComplexClosureResult:
    return _admit_piece_assignments(_admit_complex(complex_value), pieces)


def _admit_piece_assignments(
    complex_value: PolytopalComplexClosureResult,
    pieces: tuple[PieceAssignment, ...],
) -> PolytopalComplexClosureResult:
    """Admit piece axes and compatibility work on an already checked complex."""
    canonical_pieces = _canonical_piece_assignments(complex_value, pieces)
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


def _canonical_piece_assignments(
    complex_value: PolytopalComplexClosureResult,
    pieces: tuple[PieceAssignment, ...],
) -> tuple[PieceAssignment, ...]:
    """Validate source-to-cell polynomial axes without computing continuity."""
    if not isinstance(pieces, tuple) or not pieces:
        _reject("piece_type", "pieces must be canonical cell-polynomial assignments")
    if len(pieces) > MAX_COMPLEX_CELLS:
        raise OperationResourceAdmissionError(
            location=("pieces",),
            code="polytopal_complex.piece_count",
            message="too many pieces",
        )
    if any(not isinstance(row, PieceAssignment) for row in pieces):
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
    return canonical_pieces


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


def piecewise_polynomial_add(  # noqa: C901
    request: PiecewisePolynomialAdditionRequest,
) -> PiecewisePolynomialResult:
    """Add two compatible C0 functions on the identical canonical complex."""
    if not isinstance(request, PiecewisePolynomialAdditionRequest):
        _reject("addition_type", "expected two canonical piecewise-polynomial values")
    left, right = request.left, request.right
    try:
        left_payload = left.model_dump(mode="python")
        right_payload = right.model_dump(mode="python")
        left = PiecewisePolynomialResult.model_validate(left_payload)
        right = PiecewisePolynomialResult.model_validate(right_payload)
    except Exception:
        _reject(
            "addition_input",
            "both summands must be canonical piecewise-polynomial values",
        )
    if left_payload != left.model_dump(
        mode="python"
    ) or right_payload != right.model_dump(mode="python"):
        _reject(
            "addition_input",
            "both summands must be canonical piecewise-polynomial values",
        )
    if left.complex.model_dump(mode="python") != right.complex.model_dump(
        mode="python"
    ):
        _reject("addition_complex", "summands must use the identical canonical complex")
    if left.status != "COMPATIBLE" or right.status != "COMPATIBLE":
        _reject(
            "addition_continuity",
            "only compatible C0 piecewise-polynomial functions can be added",
        )

    left_by_id = {piece.cell_id: piece.polynomial for piece in left.pieces}
    right_by_id = {piece.cell_id: piece.polynomial for piece in right.pieces}
    if set(left_by_id) != set(right_by_id) or not left_by_id:
        _reject(
            "addition_piece_axis", "summands must assign exactly the same maximal cells"
        )
    if any(
        left_by_id[cell].variables != right_by_id[cell].variables for cell in left_by_id
    ):
        _reject(
            "addition_piece_ring",
            "summand pieces must use the same ordered polynomial ring",
        )

    # Preflight the exact union support and every rational sum before any
    # coefficient arithmetic or compatibility reduction is performed.
    output_term_count = 0
    max_piece_term_count = 0
    output_digit_bound = 0
    maximum_degree = 0
    for cell_id in sorted(left_by_id):
        first = left_by_id[cell_id].polynomial.terms
        second = right_by_id[cell_id].polynomial.terms
        first_by_exponent = {term.exponents: term.coefficient for term in first}
        second_by_exponent = {term.exponents: term.coefficient for term in second}
        exponents = first_by_exponent.keys() | second_by_exponent.keys()
        if len(exponents) > MAX_POLYNOMIAL_TERMS:
            raise OperationResourceAdmissionError(
                location=("pieces", cell_id),
                code="polytopal_complex.addition_terms",
                message=f"a sum piece may contain at most {MAX_POLYNOMIAL_TERMS} terms",
            )
        output_term_count += len(exponents)
        max_piece_term_count = max(max_piece_term_count, len(exponents))
        for exponent in exponents:
            maximum_degree = max(maximum_degree, sum(exponent))
            left_coefficient = first_by_exponent.get(exponent)
            right_coefficient = second_by_exponent.get(exponent)
            if left_coefficient is not None and right_coefficient is not None:
                if (
                    left_coefficient.num == -right_coefficient.num
                    and left_coefficient.den == right_coefficient.den
                ):
                    # Canonical reduced form makes exact cancellation an
                    # equality test on the components.  The zero sum never
                    # enters the output, so the growth bound does not apply.
                    continue
                numerator_digits = (
                    max(
                        _decimal_digits_upper(left_coefficient.num)
                        + _decimal_digits_upper(right_coefficient.den),
                        _decimal_digits_upper(right_coefficient.num)
                        + _decimal_digits_upper(left_coefficient.den),
                    )
                    + 1
                )
                denominator_digits = _decimal_digits_upper(
                    left_coefficient.den
                ) + _decimal_digits_upper(right_coefficient.den)
                if (
                    max(numerator_digits, denominator_digits)
                    > MAX_CANONICAL_RATIONAL_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=("pieces", cell_id),
                        code="polytopal_complex.addition_scalar_growth",
                        message="a sum coefficient may exceed the canonical rational digit envelope",
                    )
                output_digit_bound += numerator_digits + denominator_digits
            elif left_coefficient is not None:
                output_digit_bound += _decimal_digits_upper(
                    left_coefficient.num
                ) + _decimal_digits_upper(left_coefficient.den)
            elif right_coefficient is not None:
                output_digit_bound += _decimal_digits_upper(
                    right_coefficient.num
                ) + _decimal_digits_upper(right_coefficient.den)
            else:
                raise ArithmeticError("sum support disagrees with its operand union")
    if output_digit_bound > MAX_PIECE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("pieces",),
            code="polytopal_complex.addition_output",
            message="the sum polynomial output exceeds the admitted digit envelope",
        )

    dimension = len(left.complex.space.axes)
    reduced_terms = max_piece_term_count * comb(maximum_degree + dimension, dimension)
    if reduced_terms > 4_096:
        raise OperationResourceAdmissionError(
            location=("pieces",),
            code="polytopal_complex.addition_compatibility_terms",
            message="the sum's reduced compatibility polynomials exceed the admitted term envelope",
        )
    pair_count = sum(
        len(face.maximal_cell_ids) * (len(face.maximal_cell_ids) - 1) // 2
        for face in left.complex.faces
    )
    aggregate_work = 3 * pair_count * max(reduced_terms, 1) * (dimension + 1) ** 3
    if aggregate_work > MAX_PIECE_COMPATIBILITY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.addition_work",
            message="input validation and sum compatibility exceed the admitted work envelope",
        )

    left_checked = piecewise_polynomial_from_maximal_pieces(left.complex, left.pieces)
    right_checked = piecewise_polynomial_from_maximal_pieces(
        right.complex, right.pieces
    )
    if left_checked.status != "COMPATIBLE" or right_checked.status != "COMPATIBLE":
        _reject("addition_continuity", "summand compatibility claims must be exact")
    if tuple(
        (row.first_cell_id, row.second_cell_id, row.face_id)
        for row in left_checked.compatibility
    ) != tuple(
        (row.first_cell_id, row.second_cell_id, row.face_id)
        for row in right_checked.compatibility
    ):
        _reject(
            "addition_compatibility_profile",
            "summands do not share one canonical interface profile",
        )

    sum_pieces = []
    for cell_id in sorted(left_by_id):
        first_coefficients = {
            term.exponents: term.coefficient.as_fraction()
            for term in left_by_id[cell_id].polynomial.terms
        }
        second_coefficients = {
            term.exponents: term.coefficient.as_fraction()
            for term in right_by_id[cell_id].polynomial.terms
        }
        terms = []
        for exponent in sorted(
            first_coefficients.keys() | second_coefficients.keys(), reverse=True
        ):
            coefficient = first_coefficients.get(
                exponent, Fraction(0)
            ) + second_coefficients.get(exponent, Fraction(0))
            if coefficient:
                terms.append(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_fraction(coefficient),
                        exponents=exponent,
                    )
                )
        sum_pieces.append(
            PieceAssignment(
                cell_id=cell_id,
                polynomial=RationalPolynomial(
                    variables=left_by_id[cell_id].variables,
                    polynomial=SparseRationalPolynomial(terms=tuple(terms)),
                ),
            )
        )
    # Normal-form restriction is linear. Since both input ledgers have been
    # recomputed and every row is zero, the sum has the same exact zero ledger.
    return PiecewisePolynomialResult(
        complex=left_checked.complex,
        pieces=tuple(sum_pieces),
        compatibility=left_checked.compatibility,
        status="COMPATIBLE",
    )


def piecewise_polynomial_multiply(  # noqa: C901
    request: PiecewisePolynomialMultiplicationRequest,
) -> PiecewisePolynomialResult:
    """Multiply two compatible functions cellwise on one exact complex.

    Restriction to a shared face is a ring homomorphism, so products of two
    functions with equal restrictions again have equal restrictions. The
    returned ledger records those exact zero differences after both source
    ledgers have been recomputed.
    """
    if not isinstance(request, PiecewisePolynomialMultiplicationRequest):
        _reject(
            "multiplication_type", "expected two canonical piecewise-polynomial values"
        )
    left, right = request.left, request.right
    try:
        left_payload = left.model_dump(mode="python")
        right_payload = right.model_dump(mode="python")
        left = PiecewisePolynomialResult.model_validate(left_payload)
        right = PiecewisePolynomialResult.model_validate(right_payload)
    except Exception:
        _reject(
            "multiplication_input",
            "both factors must be canonical piecewise-polynomial values",
        )
    if left_payload != left.model_dump(
        mode="python"
    ) or right_payload != right.model_dump(mode="python"):
        _reject(
            "multiplication_input",
            "both factors must be canonical piecewise-polynomial values",
        )
    if left.complex.model_dump(mode="python") != right.complex.model_dump(
        mode="python"
    ):
        _reject(
            "multiplication_complex", "factors must use the identical canonical complex"
        )
    if left.status != "COMPATIBLE" or right.status != "COMPATIBLE":
        _reject(
            "multiplication_continuity",
            "only compatible C0 piecewise-polynomial functions can be multiplied",
        )

    left_by_id = {piece.cell_id: piece.polynomial for piece in left.pieces}
    right_by_id = {piece.cell_id: piece.polynomial for piece in right.pieces}
    if set(left_by_id) != set(right_by_id) or not left_by_id:
        _reject(
            "multiplication_piece_axis",
            "factors must assign exactly the same maximal cells",
        )
    if any(
        left_by_id[cell].variables != right_by_id[cell].variables for cell in left_by_id
    ):
        _reject(
            "multiplication_piece_ring",
            "factor pieces must use the same ordered polynomial ring",
        )

    # Preflight aggregate sparse convolution work, support, degree, coefficient
    # height, and serialized output before multiplying any cell polynomials.
    maximum_result_terms = 0
    maximum_coefficient_digits = 1
    maximum_dimension = 0
    aggregate_convolution_work = 0
    for cell_id in sorted(left_by_id):
        first = left_by_id[cell_id]
        second = right_by_id[cell_id]
        first_terms = first.polynomial.terms
        second_terms = second.polynomial.terms
        convolution_work = len(first_terms) * len(second_terms)
        aggregate_convolution_work += convolution_work
        maximum_dimension = max(maximum_dimension, len(first.variables))
        if aggregate_convolution_work > MAX_PIECE_MULTIPLICATION_WORK:
            raise OperationResourceAdmissionError(
                location=("pieces", cell_id),
                code="polytopal_complex.multiplication_work",
                message="aggregate piecewise polynomial convolution exceeds the admitted work envelope",
            )
        if convolution_work > MAX_MULTIPLY_PRODUCT_WORK:
            raise OperationResourceAdmissionError(
                location=("pieces", cell_id),
                code="polytopal_complex.multiplication_work",
                message="one piece polynomial convolution exceeds the admitted work envelope",
            )
        maximum_exponents = tuple(
            max((term.exponents[axis] for term in first_terms), default=0)
            + max((term.exponents[axis] for term in second_terms), default=0)
            for axis in range(len(first.variables))
        )
        if any(exponent > MAX_POLYNOMIAL_EXPONENT for exponent in maximum_exponents):
            raise OperationResourceAdmissionError(
                location=("pieces", cell_id),
                code="polytopal_complex.multiplication_exponent",
                message="a product exponent exceeds the canonical polynomial limit",
            )
        support_bound = 1
        for exponent in maximum_exponents:
            support_bound *= exponent + 1
        result_term_bound = min(convolution_work, support_bound)
        if result_term_bound > MAX_POLYNOMIAL_TERMS:
            raise OperationResourceAdmissionError(
                location=("pieces", cell_id),
                code="polytopal_complex.multiplication_terms",
                message="a product piece may exceed the canonical polynomial term limit",
            )
        maximum_result_terms = max(maximum_result_terms, result_term_bound)
        coefficient_digits = _maximum_product_coefficient_digits(first, second)
        if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationResourceAdmissionError(
                location=("pieces", cell_id),
                code="polytopal_complex.multiplication_scalar_growth",
                message="a product coefficient may exceed the canonical rational digit limit",
            )
        maximum_coefficient_digits = max(maximum_coefficient_digits, coefficient_digits)

    pair_count = sum(
        len(face.maximal_cell_ids) * (len(face.maximal_cell_ids) - 1) // 2
        for face in left.complex.faces
    )
    compatibility_term_bound = maximum_result_terms * comb(
        max(
            (
                sum(term.exponents)
                for polynomial in left_by_id.values()
                for term in polynomial.polynomial.terms
            ),
            default=0,
        )
        + max(
            (
                sum(term.exponents)
                for polynomial in right_by_id.values()
                for term in polynomial.polynomial.terms
            ),
            default=0,
        )
        + maximum_dimension,
        maximum_dimension,
    )
    compatibility_work = (
        3 * pair_count * max(compatibility_term_bound, 1) * (maximum_dimension + 1) ** 3
    )
    if compatibility_work > MAX_PIECE_COMPATIBILITY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.multiplication_compatibility_work",
            message="product compatibility validation exceeds the admitted work envelope",
        )

    # The product ledger stores two reduced rational components per retained
    # term coefficient and at most max_exponent_digits decimal digits per
    # exponent. The zero compatibility ledger, cell IDs, and transported
    # complex are fixed-cardinality structural data admitted by the complex
    # and work envelopes above; transport byte policy is enforced downstream.
    max_exponent_digits = 5
    result_term_count = len(left_by_id) * maximum_result_terms
    product_output_digit_bound = result_term_count * (
        2 * maximum_coefficient_digits + maximum_dimension * max_exponent_digits
    )
    if product_output_digit_bound > MAX_PIECE_MULTIPLICATION_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("pieces",),
            code="polytopal_complex.multiplication_output",
            message="piecewise polynomial product may exceed the admitted digit envelope",
        )

    left_checked = piecewise_polynomial_from_maximal_pieces(left.complex, left.pieces)
    right_checked = piecewise_polynomial_from_maximal_pieces(
        right.complex, right.pieces
    )
    if left_checked.status != "COMPATIBLE" or right_checked.status != "COMPATIBLE":
        _reject(
            "multiplication_continuity",
            "factor compatibility claims must be exact",
        )
    profile = tuple(
        (row.first_cell_id, row.second_cell_id, row.face_id)
        for row in left_checked.compatibility
    )
    if profile != tuple(
        (row.first_cell_id, row.second_cell_id, row.face_id)
        for row in right_checked.compatibility
    ):
        _reject(
            "multiplication_compatibility_profile",
            "factors do not share one canonical interface profile",
        )

    product_pieces = tuple(
        PieceAssignment(
            cell_id=cell_id,
            polynomial=rational_polynomial_multiply(
                left_by_id[cell_id], right_by_id[cell_id]
            ),
        )
        for cell_id in sorted(left_by_id)
    )
    variables = product_pieces[0].polynomial.variables
    zero = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(terms=()),
    )
    compatibility = tuple(
        PieceCompatibilityRow(
            first_cell_id=row.first_cell_id,
            second_cell_id=row.second_cell_id,
            face_id=row.face_id,
            reduced_difference=zero,
            compatible=True,
        )
        for row in left_checked.compatibility
    )
    return PiecewisePolynomialResult(
        complex=left_checked.complex,
        pieces=product_pieces,
        compatibility=compatibility,
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


def _facet_remainder_dimension(dimension: int, degree: int, smoothness: int) -> int:
    """Count degree-bounded monomials modulo a facet equation to power r+1."""
    normal_orders = range(min(degree, smoothness) + 1)
    return sum(
        comb(dimension - 1 + degree - normal_order, dimension - 1)
        for normal_order in normal_orders
    )


def _require_pure_spline_complex(
    complex_value: PolytopalComplexClosureResult, smoothness: int
) -> None:
    if smoothness >= 0 and any(
        cell.dimension != complex_value.dimension
        for cell in complex_value.maximal_cells
    ):
        _reject(
            "spline_purity",
            "C^r spline spaces require every maximal cell to have full complex dimension",
        )


def _linear_form(face: Any, symbols: tuple[Any, ...]) -> Any:
    ideal = _affine_ideal(face, symbols)
    if len(ideal) != 1:
        _reject(
            "spline_facet",
            "every interior spline interface must be a codimension-one facet",
        )
    return sp.Poly(ideal[0], *symbols, domain=sp.QQ)


def piecewise_polynomial_smoothness(
    request: PiecewiseSmoothnessRequest,
) -> PiecewiseSmoothnessResult:
    """Return exact C^r orders on every interior facet of a piecewise polynomial."""
    if not isinstance(request, PiecewiseSmoothnessRequest):
        _reject("smoothness_request", "expected a canonical smoothness request")
    try:
        request = PiecewiseSmoothnessRequest.model_validate(
            request.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValueError):
        _reject(
            "smoothness_request", "request fields must satisfy the smoothness schema"
        )
    function = request.function
    complex_value = _admit_pieces(function.complex, function.pieces)
    if any(
        cell.dimension != complex_value.dimension
        for cell in complex_value.maximal_cells
    ):
        _reject("smoothness_purity", "facet smoothness profiles require a pure complex")
    # Recompute the supplied continuity claim from its exact cell pieces.
    function = piecewise_polynomial_from_maximal_pieces(complex_value, function.pieces)
    cells = {row.cell_id: row.polynomial for row in function.pieces}
    dimension = complex_value.dimension
    facets = tuple(
        face
        for face in complex_value.faces
        if face.dimension == dimension - 1 and len(face.maximal_cell_ids) == 2
    )
    term_count = max((len(poly.polynomial.terms) for poly in cells.values()), default=1)
    work = (
        len(facets) * term_count * (dimension + 1) ** 3 * (request.max_smoothness + 1)
    )
    if work > MAX_PIECE_COMPATIBILITY_WORK:
        raise OperationResourceAdmissionError(
            location=("function",),
            code="polytopal_complex.smoothness_work",
            message="facet smoothness reduction exceeds the admitted envelope",
        )
    symbols = _poly_symbols(next(iter(cells.values())))
    rows: list[PiecewiseFacetSmoothness] = []
    for face in facets:
        first, second = face.maximal_cell_ids
        difference = _to_poly(cells[first], symbols) - _to_poly(cells[second], symbols)
        defining_form = _linear_form(face, symbols)
        order_found = -1
        for order in range(request.max_smoothness + 1):
            _, remainder = sp.div(difference, defining_form ** (order + 1))
            if not remainder.is_zero:
                break
            order_found = order
        rows.append(
            PiecewiseFacetSmoothness(
                first_cell_id=first,
                second_cell_id=second,
                face_id=face.face_id,
                smoothness=order_found,
            )
        )
    global_order = min((row.smoothness for row in rows), default=request.max_smoothness)
    return PiecewiseSmoothnessResult(
        function=function,
        max_smoothness=request.max_smoothness,
        facets=tuple(rows),
        smoothness=global_order,
    )


MAX_SPLINE_CONSTRAINT_CELLS = 1_048_576
MAX_SPLINE_RESULT_CELLS = 1_000_000
MAX_SPLINE_RESULT_DIGITS = 64 * 1024 * 1024
MAX_SPLINE_SCALAR_DIGITS = 2 * 32_768
MAX_SPLINE_DIMENSION_CONSTRAINT_CELLS = 1_048_576
MAX_SPLINE_DIMENSION_RANK_WORK = 32_000_000
MAX_SPLINE_DIMENSION_INTERMEDIATE_DIGITS = 32_768
MAX_SPLINE_DIMENSION_OUTPUT_DIGITS = 10 * 1024 * 1024
"""Decimal-digit ceiling summed over stored rational cells of the matrix."""
MAX_SPLINE_DIMENSION_INTERMEDIATE_BYTES = 512 * 1024 * 1024
MAX_SPLINE_COORDINATE_OUTPUT_BYTES = CanonicalLimits().max_output_bytes


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
    _require_pure_spline_complex(complex_value, smoothness)
    dimension = len(complex_value.space.axes)
    if any(cell.dimension != dimension for cell in cells):
        _reject(
            "spline_ambient_dimension",
            "spline coordinates require maximal cells full-dimensional in the ambient space",
        )
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
    row_bound = interface_count * _facet_remainder_dimension(
        dimension, degree, smoothness
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
    rank_work = 2 * row_bound * width * min(row_bound, width)
    if rank_work > MAX_SPLINE_DIMENSION_RANK_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.spline_rank_work",
            message="spline rank and nullspace work exceeds the admitted envelope",
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
    complex_value, coefficient_axis, constraint_rows, width = _spline_constraint_data(
        complex_value, degree, smoothness
    )
    return _spline_space_from_data(
        complex_value, degree, smoothness, coefficient_axis, constraint_rows, width
    )


def _spline_space_from_data(
    complex_value: PolytopalComplexClosureResult,
    degree: int,
    smoothness: int,
    coefficient_axis: tuple[tuple[str, tuple[int, ...]], ...],
    constraint_rows: tuple[tuple[Fraction, ...], ...],
    width: int,
) -> SplineSpaceResult:
    """Materialize an admitted spline basis from its exact coefficient rows."""
    matrix = _spline_constraint_matrix(constraint_rows, width)
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


def _spline_basis_coordinates(
    space: SplineSpaceResult, vector: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Recover exact coordinates from the unit columns of a nullspace basis."""
    basis = tuple(
        tuple(value.as_fraction() for value in row)
        for row in space.nullspace_basis.entries
    )
    free_columns: list[int | None] = [None] * len(basis)
    for column in range(len(space.coefficient_axis)):
        one_rows = [row for row, values in enumerate(basis) if values[column] == 1]
        if len(one_rows) == 1 and all(
            value == 0 or row == one_rows[0]
            for row, values in enumerate(basis)
            for value in (values[column],)
        ):
            row_index = one_rows[0]
            if free_columns[row_index] is None:
                free_columns[row_index] = column
    if any(column is None for column in free_columns):
        raise ArithmeticError(
            "canonical spline nullspace lost its free-coordinate axes"
        )
    coordinates = tuple(vector[column] for column in free_columns if column is not None)
    reconstructed = tuple(
        sum(
            (coordinates[row] * values[column] for row, values in enumerate(basis)),
            Fraction(0),
        )
        for column in range(len(space.coefficient_axis))
    )
    if reconstructed != vector:
        _reject(
            "spline_coordinates_not_member",
            "piece polynomials are not in the exact span of the retained spline basis",
        )
    return coordinates


def _admit_spline_coordinate_materialization(
    complex_value: PolytopalComplexClosureResult,
    coefficient_axis: tuple[tuple[str, tuple[int, ...]], ...],
    constraint_rows: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
    width: int,
) -> None:
    """Bound retained basis bytes and exact work before nullspace expansion."""
    maximum_entry_digits = max(
        (
            _decimal_digits_upper(value.numerator)
            + _decimal_digits_upper(value.denominator)
            for row in constraint_rows
            for value in row
        ),
        default=1,
    )
    rank_bound = min(len(constraint_rows), width)
    row_height = (width + 1) * maximum_entry_digits + len(str(width + 1))
    determinant_digits = rank_bound * row_height + rank_bound * len(
        str(max(rank_bound, 1))
    )
    if determinant_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_coordinates_height",
            message="the canonical spline basis may exceed the exact rational component bound",
        )
    basis_scalar_digits = max(1, 2 * determinant_digits + 4)
    coordinate_scalar_digits = max(
        (
            _decimal_digits_upper(value.numerator)
            + _decimal_digits_upper(value.denominator)
            for value in vector
        ),
        default=1,
    )
    result_bound = (
        (len(constraint_rows) * width + width * width) * (2 * basis_scalar_digits + 32)
        + width * (coordinate_scalar_digits + 32)
        + len(encode_strict_json(complex_value.model_dump(mode="json")))
        + 512 * len(coefficient_axis)
        + 4096
    )
    if result_bound > MAX_SPLINE_COORDINATE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_coordinates_output",
            message="the source-bound spline space and coordinates exceed the output envelope",
        )
    matrix_work = len(constraint_rows) * width
    if matrix_work + 3 * width * width > 32_000_000:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_coordinates_work",
            message="spline membership and basis-coordinate work exceed the admitted envelope",
        )
    reconstruction_digits = width * (
        coordinate_scalar_digits + basis_scalar_digits + len(str(width)) + 2
    )
    if reconstruction_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("function",),
            code="polytopal_complex.spline_coordinates_reconstruction_height",
            message="reconstructing the basis coordinates may exceed the exact scalar envelope",
        )
    coefficient_digits = coordinate_scalar_digits
    for row in constraint_rows:
        value_digits = max(
            (
                _decimal_digits_upper(value.numerator)
                + _decimal_digits_upper(value.denominator)
                for value in row
            ),
            default=1,
        )
        if (
            width * (value_digits + coefficient_digits + len(str(width)) + 2)
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("function",),
                code="polytopal_complex.spline_coordinates_height",
                message="spline membership products exceed the exact scalar envelope",
            )


def spline_coordinates(
    request: SplineCoordinatesRequest,
) -> SplineCoordinatesResult:
    """Express one supplied piecewise polynomial in the canonical spline basis.

    The operation ignores caller-provided compatibility claims and derives the
    exact cell coefficient vector from the source pieces.  Membership is
    established by the retained spline compatibility matrix, then coordinates
    are recovered from the unit columns of the canonical nullspace basis.
    """
    if not isinstance(request, SplineCoordinatesRequest):
        _reject(
            "spline_coordinates_type", "expected a canonical spline-coordinate request"
        )
    function = request.function
    if not isinstance(function, PiecewisePolynomialResult):
        _reject("spline_coordinates_function", "expected a piecewise-polynomial value")

    complex_value, cells, width = _admit_spline(
        function.complex, request.degree, request.smoothness
    )
    pieces = _canonical_piece_assignments(complex_value, function.pieces)
    cell_order = tuple(
        sorted(complex_value.maximal_cells, key=lambda cell: cell.cell_id)
    )
    monomials = _monomials(len(complex_value.space.axes), request.degree)
    coefficient_axis = tuple(
        (cell.cell_id, monomial) for cell in cell_order for monomial in monomials
    )
    columns = {axis: index for index, axis in enumerate(coefficient_axis)}
    coefficients = [Fraction(0) for _ in coefficient_axis]
    pieces_by_cell = {piece.cell_id: piece.polynomial for piece in pieces}
    variables = tuple(complex_value.space.axes)
    for cell_id in (cell.cell_id for cell in cell_order):
        polynomial = pieces_by_cell[cell_id]
        if polynomial.variables != variables:
            _reject(
                "spline_coordinates_ring", "piece variables must match the complex axes"
            )
        for term in polynomial.polynomial.terms:
            if sum(term.exponents) > request.degree:
                _reject(
                    "spline_coordinates_degree",
                    "every piece degree must be at most the requested spline degree",
                )
            column = columns.get((cell_id, term.exponents))
            if column is None:
                _reject(
                    "spline_coordinates_axis",
                    "a piece term is outside the monomial axis",
                )
            coefficients[column] = term.coefficient.as_fraction()

    complex_value, axis, constraint_rows, width = _spline_constraint_rows(
        complex_value, cells, width, request.degree, request.smoothness
    )
    if axis != coefficient_axis:
        raise ArithmeticError("spline coordinate coefficient-axis mismatch")
    vector = tuple(coefficients)
    _admit_spline_coordinate_materialization(
        complex_value, coefficient_axis, constraint_rows, vector, width
    )
    if any(
        sum(
            (entry * value for entry, value in zip(row, vector, strict=True)),
            Fraction(0),
        )
        for row in constraint_rows
    ):
        _reject(
            "spline_coordinates_not_member",
            "piece polynomials do not satisfy the requested exact C^r interface conditions",
        )

    space = _spline_space_from_data(
        complex_value,
        request.degree,
        request.smoothness,
        coefficient_axis,
        constraint_rows,
        width,
    )
    basis_coordinates = _spline_basis_coordinates(space, vector)
    return SplineCoordinatesResult(
        spline_space=space,
        basis_coordinates=tuple(
            CanonicalRational.from_fraction(value) for value in basis_coordinates
        ),
    )


def _spline_constraint_data(
    complex_value: PolytopalComplexClosureResult,
    degree: int,
    smoothness: int,
    *,
    dimension_only: bool = False,
) -> tuple[
    PolytopalComplexClosureResult,
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[Fraction, ...], ...],
    int,
]:
    """Build the admitted exact spline matrix once for basis and dimension paths."""
    if dimension_only:
        complex_value, width, _, _ = _admit_spline_dimension(
            complex_value, degree, smoothness
        )
        cells = tuple(
            sorted(complex_value.maximal_cells, key=lambda cell: cell.cell_id)
        )
    else:
        complex_value, cells, width = _admit_spline(complex_value, degree, smoothness)
    return _spline_constraint_rows(complex_value, cells, width, degree, smoothness)


def _spline_constraint_rows(
    complex_value: PolytopalComplexClosureResult,
    cells: tuple[Any, ...],
    width: int,
    degree: int,
    smoothness: int,
) -> tuple[
    PolytopalComplexClosureResult,
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[Fraction, ...], ...],
    int,
]:
    """Build exact interface rows after the complex and dimensions are admitted."""
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
    upper_rows = sum(
        1
        for face in complex_value.faces
        if len(face.maximal_cell_ids) == 2
        and face.dimension == dimension - 1
        and smoothness >= 0
    ) * _facet_remainder_dimension(dimension, degree, smoothness)
    if len(constraint_rows) > upper_rows:
        raise ArithmeticError("spline constraint row admission mismatch")
    return (
        complex_value,
        coefficient_axis,
        tuple(tuple(row) for row in constraint_rows),
        width,
    )


def _spline_constraint_matrix(
    rows: tuple[tuple[Fraction, ...], ...], width: int
) -> RationalMatrix:
    return RationalMatrix(
        row_count=len(rows),
        column_count=width,
        entries=tuple(
            tuple(CanonicalRational.from_fraction(value) for value in row)
            for row in rows
        ),
    )


def _admit_spline_dimension(
    complex_value: PolytopalComplexClosureResult,
    degree: int,
    smoothness: int,
) -> tuple[PolytopalComplexClosureResult, int, int, int]:
    """Preflight compact matrix construction and exact rank work."""
    complex_value = _admit_complex(complex_value)
    if type(degree) is not int or type(smoothness) is not int:
        _reject("spline_type", "spline degree and smoothness must be exact integers")
    if degree < 0 or degree > 12 or smoothness < -1 or smoothness > 4:
        _reject(
            "spline_parameters", "degree must be in [0, 12] and smoothness in [-1, 4]"
        )
    dimension = len(complex_value.space.axes)
    if any(not axis.strip() for axis in complex_value.space.axes):
        _reject("spline_axes", "complex coordinate axes must be nonempty symbols")
    _require_pure_spline_complex(complex_value, smoothness)
    monomial_count = comb(dimension + degree, degree)
    width = len(complex_value.maximal_cells) * monomial_count
    if width > 4096:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_width",
            message="spline coefficient axis exceeds the admitted envelope",
        )
    interfaces = sum(
        len(face.maximal_cell_ids) == 2
        and face.dimension == dimension - 1
        and smoothness >= 0
        for face in complex_value.faces
    )
    row_bound = interfaces * _facet_remainder_dimension(dimension, degree, smoothness)
    matrix_cells = row_bound * width
    rank_work = 2 * row_bound * width * min(row_bound, width)
    if matrix_cells > MAX_SPLINE_DIMENSION_CONSTRAINT_CELLS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.spline_dimension_matrix",
            message="spline dimension compatibility matrix exceeds its cell envelope",
        )
    if rank_work > MAX_SPLINE_DIMENSION_RANK_WORK:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="polytopal_complex.spline_dimension_work",
            message="spline dimension exact rank exceeds its admitted work envelope",
        )
    return complex_value, width, row_bound, rank_work


def _spline_dimension_output_digit_bound(
    rows: tuple[tuple[Fraction, ...], ...],
    width: int,
    max_entry_digits: int,
) -> int:
    """Bound the stored rational height of the exact compatibility matrix.

    The complex and coefficient axis are fixed-cardinality structural data
    already admitted by the input and width envelopes; the matrix is the only
    component that can grow, and every stored cell is a reduced rational with
    numerator and denominator within the measured entry-height bound.
    """
    return len(rows) * width * 2 * max_entry_digits


def spline_dimension(request: SplineDimensionRequest) -> SplineDimensionResult:
    """Return the exact spline dimension without constructing a nullspace basis."""
    if not isinstance(request, SplineDimensionRequest):
        _reject(
            "spline_dimension_type", "expected a canonical spline dimension request"
        )
    # Matrix construction has a row bound admitted above.  Measure exact scalar
    # heights and serialized matrix output before handing the matrix to rank().
    complex_value, axis, rows, width = _spline_constraint_data(
        request.complex,
        request.degree,
        request.smoothness,
        dimension_only=True,
    )
    dimension = len(complex_value.space.axes)
    row_bound = sum(
        len(face.maximal_cell_ids) == 2
        and face.dimension == dimension - 1
        and request.smoothness >= 0
        for face in complex_value.faces
    ) * _facet_remainder_dimension(dimension, request.degree, request.smoothness)
    if len(rows) > row_bound or any(len(row) != width for row in rows):
        raise ArithmeticError("spline dimension matrix shape admission mismatch")
    max_entry_digits = max(
        (
            max(
                decimal_digit_width(value.numerator),
                decimal_digit_width(value.denominator),
            )
            for row in rows
            for value in row
        ),
        default=1,
    )
    output_digits = _spline_dimension_output_digit_bound(rows, width, max_entry_digits)
    if output_digits > MAX_SPLINE_DIMENSION_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_dimension_output",
            message="spline dimension exact matrix exceeds its output envelope",
        )
    pivot_size = min(len(rows), width)
    intermediate_digits = (pivot_size + 1) * (
        max_entry_digits + len(str(max(len(rows), width))) + 2
    )
    intermediate_bytes = len(rows) * width * (2 * intermediate_digits + 16)
    if (
        intermediate_digits > MAX_SPLINE_DIMENSION_INTERMEDIATE_DIGITS
        or intermediate_bytes > MAX_SPLINE_DIMENSION_INTERMEDIATE_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="polytopal_complex.spline_dimension_height",
            message="spline dimension rank intermediates exceed their height or storage envelope",
        )
    matrix = _spline_constraint_matrix(rows, width)
    if rows:
        from flint import fmpq, fmpq_mat

        backend = fmpq_mat(
            [
                [fmpq(value.numerator, value.denominator) for value in row]
                for row in rows
            ]
        )
        rank = int(backend.rank())
    else:
        rank = 0
    nullity = width - rank
    return SplineDimensionResult(
        complex=complex_value,
        degree=request.degree,
        smoothness=request.smoothness,
        coefficient_axis=axis,
        compatibility_matrix=matrix,
        rank=rank,
        nullity=nullity,
    )


def _canonical_spline_evaluation_inputs(
    request: SplineEvaluationRequest,
) -> tuple[ComplexPoint, tuple[CanonicalRational, ...]]:
    point = request.point
    try:
        point = ComplexPoint.model_validate(point.model_dump(mode="python"))
    except Exception:
        _reject("point_shape", "evaluation point must be a canonical complex point")
    if (
        not isinstance(request.basis_coefficients, tuple)
        or len(request.basis_coefficients) > 4096
        or any(
            not isinstance(value, CanonicalRational)
            or value.num.bit_length() > 4 * MAX_CANONICAL_RATIONAL_DIGITS
            or value.den.bit_length() > 4 * MAX_CANONICAL_RATIONAL_DIGITS
            for value in request.basis_coefficients
        )
    ):
        _reject(
            "spline_basis_coordinates",
            "basis coefficients must be a bounded canonical rational tuple",
        )
    try:
        basis_coefficients = tuple(
            CanonicalRational.model_validate(value.model_dump(mode="python"))
            for value in request.basis_coefficients
        )
    except Exception:
        _reject(
            "spline_basis_coordinates",
            "basis coefficients must be a bounded canonical rational tuple",
        )
    return point, basis_coefficients


def _decimal_digits_upper(value: int) -> int:
    if value == 0:
        return 1
    # log10(2) < 30103/100000, so avoid stringifying caller-sized integers.
    return (abs(value).bit_length() * 30_103) // 100_000 + 1


def _containing_cell_ids(
    complex_value: PolytopalComplexClosureResult, point: ComplexPoint
) -> frozenset[str]:
    coordinates = tuple(value.as_fraction() for value in point.coordinates)
    return frozenset(
        cell.cell_id
        for cell in complex_value.maximal_cells
        if _contains(cell, coordinates)
    )


def _admit_spline_evaluation_growth(
    spline: SplineSpaceResult,
    coefficients: tuple[CanonicalRational, ...],
    point: ComplexPoint,
    containing_ids: frozenset[str],
) -> None:
    """Bound the full rational sum before constructing its products."""

    estimated_digits = 0
    for basis_index, scalar in enumerate(coefficients):
        scalar_digits = _decimal_digits_upper(scalar.num) + _decimal_digits_upper(
            scalar.den
        )
        if not scalar.num:
            continue
        for column, (cell_id, exponents) in enumerate(spline.coefficient_axis):
            if cell_id not in containing_ids:
                continue
            basis_scalar = spline.nullspace_basis.entries[basis_index][column]
            if not basis_scalar.num:
                continue
            numerator_digits = scalar_digits + _decimal_digits_upper(basis_scalar.num)
            denominator_digits = _decimal_digits_upper(basis_scalar.den)
            for coordinate, power in zip(point.coordinates, exponents, strict=True):
                numerator_digits += power * _decimal_digits_upper(coordinate.num)
                denominator_digits += power * _decimal_digits_upper(coordinate.den)
            estimated_digits += numerator_digits + denominator_digits + 1
            if estimated_digits > MAX_CANONICAL_RATIONAL_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("basis_coefficients",),
                    code="polytopal_complex.spline_evaluation_growth",
                    message=(
                        "spline evaluation exceeds the canonical rational output envelope"
                    ),
                )


def _evaluate_spline_basis_combination(
    spline: SplineSpaceResult,
    coefficients: tuple[CanonicalRational, ...],
    point: ComplexPoint,
    containing_ids: frozenset[str],
) -> tuple[tuple[str, ...], Fraction | None]:
    cells = tuple(
        cell for cell in spline.complex.maximal_cells if cell.cell_id in containing_ids
    )
    point_fractions = tuple(value.as_fraction() for value in point.coordinates)
    if not cells:
        return (), None
    coefficient_fractions = tuple(value.as_fraction() for value in coefficients)
    basis_values = tuple(
        tuple(value.as_fraction() for value in row)
        for row in spline.nullspace_basis.entries
    )
    combined = [Fraction(0) for _ in spline.coefficient_axis]
    for basis_index, scalar in enumerate(coefficient_fractions):
        if not scalar:
            continue
        for column, entry in enumerate(basis_values[basis_index]):
            if entry:
                combined[column] += scalar * entry
    by_cell: dict[str, list[Fraction]] = {cell.cell_id: [] for cell in cells}
    for coefficient, (cell_id, exponents) in zip(
        combined, spline.coefficient_axis, strict=True
    ):
        if cell_id not in by_cell or not coefficient:
            continue
        monomial = Fraction(1)
        for coordinate, power in zip(point_fractions, exponents, strict=True):
            monomial *= coordinate**power
        by_cell[cell_id].append(coefficient * monomial)
    values = tuple(sum(terms, Fraction(0)) for terms in by_cell.values())
    if any(other != values[0] for other in values[1:]):
        raise ArithmeticError("canonical spline basis disagrees on a shared face")
    return tuple(cell.cell_id for cell in cells), values[0]


def spline_evaluate(
    request: SplineEvaluationRequest,
) -> SplineEvaluationResult:
    """Evaluate a linear combination of the canonical spline-space basis.

    The request carries the complex and spline parameters rather than a
    caller-constructed basis, so the coefficient coordinates are interpreted
    against the exact nullspace produced by this operation.  Admission and
    basis construction are the same bounded path used by ``spline_space``.
    """

    if not isinstance(request, SplineEvaluationRequest):
        _reject("spline_evaluation_type", "expected a canonical spline evaluation")
    point, basis_coefficients = _canonical_spline_evaluation_inputs(request)
    spline = spline_space(request.complex, request.degree, request.smoothness)
    if len(basis_coefficients) != spline.nullity:
        _reject(
            "spline_basis_coordinates",
            "basis_coefficients must have one entry per canonical spline basis row",
        )
    axes = tuple(spline.complex.space.axes)
    if len(point.coordinates) != len(axes):
        _reject("point_axis", "evaluation point must use the complex coordinate axes")
    containing_ids = _containing_cell_ids(spline.complex, point)
    if containing_ids:
        _admit_spline_evaluation_growth(
            spline, basis_coefficients, point, containing_ids
        )
    cell_ids, value = _evaluate_spline_basis_combination(
        spline, basis_coefficients, point, containing_ids
    )
    return SplineEvaluationResult(
        complex=spline.complex,
        degree=spline.degree,
        smoothness=spline.smoothness,
        basis_coefficients=basis_coefficients,
        point=point,
        containing_cell_ids=cell_ids,
        value=None if value is None else CanonicalRational.from_fraction(value),
    )


__all__ = [
    "piecewise_polynomial_add",
    "piecewise_polynomial_evaluate",
    "piecewise_polynomial_from_maximal_pieces",
    "piecewise_polynomial_multiply",
    "spline_evaluate",
    "spline_space",
]
