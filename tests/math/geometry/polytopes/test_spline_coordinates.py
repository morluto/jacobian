from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _spline as spline_kernel
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    PieceAssignment,
    PiecewisePolynomialResult,
    SplineCoordinatesRequest,
    SplineCoordinatesResult,
    SplineEvaluationRequest,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    spline_coordinates,
    spline_evaluate,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_from_maximal_pieces,
    polytopal_complex_closure,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _interval(left: int, right: int, prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=(CanonicalRational(num=point, den=1),),
            )
            for index, point in enumerate((left, right))
        ),
    )


def _polynomial(terms: tuple[tuple[int, int], ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=(exponent,),
                )
                for exponent, coefficient in sorted(terms, reverse=True)
                if coefficient
            )
        ),
    )


def _two_interval_function(
    left_terms: tuple[tuple[int, int], ...],
    right_terms: tuple[tuple[int, int], ...],
) -> PiecewisePolynomialResult:
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    cells = tuple(sorted(complex_value.maximal_cells, key=lambda cell: cell.cell_id))
    return piecewise_polynomial_from_maximal_pieces(
        complex_value,
        (
            PieceAssignment(
                cell_id=cells[0].cell_id, polynomial=_polynomial(left_terms)
            ),
            PieceAssignment(
                cell_id=cells[1].cell_id, polynomial=_polynomial(right_terms)
            ),
        ),
    )


def test_piecewise_function_roundtrips_through_source_bound_spline_coordinates():
    function = _two_interval_function((), ((1, 1), (0, -1)))
    result = spline_coordinates(
        SplineCoordinatesRequest(function=function, degree=1, smoothness=0)
    )
    revived = SplineCoordinatesResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )

    space = revived.spline_space
    coordinate_vector = [Fraction(0) for _ in space.coefficient_axis]
    for piece in function.pieces:
        for term in piece.polynomial.polynomial.terms:
            coordinate_vector[
                space.coefficient_axis.index((piece.cell_id, term.exponents))
            ] = term.coefficient.as_fraction()
    basis = tuple(
        tuple(value.as_fraction() for value in row)
        for row in space.nullspace_basis.entries
    )
    reconstructed = tuple(
        sum(
            (
                scalar.as_fraction() * basis[row][column]
                for row, scalar in enumerate(revived.basis_coordinates)
            ),
            Fraction(0),
        )
        for column in range(len(space.coefficient_axis))
    )
    assert reconstructed == tuple(coordinate_vector)
    evaluated = spline_evaluate(
        SplineEvaluationRequest(
            complex=space.complex,
            degree=space.degree,
            smoothness=space.smoothness,
            basis_coefficients=revived.basis_coordinates,
            point=ComplexPoint(coordinates=(CanonicalRational(num=3, den=2),)),
        )
    )
    assert evaluated.value == CanonicalRational(num=1, den=2)
    assert revived == result


def test_two_dimensional_source_pieces_reconstruct_from_spline_coordinates():
    complex_value = polytopal_complex_closure(
        (
            RationalVPolytope(
                space=RationalCoordinateSpace(axes=("x", "y")),
                vertices=tuple(
                    RationalPolytopeVertex(
                        vertex_id=f"a{index}",
                        coordinates=tuple(
                            CanonicalRational(num=v, den=1) for v in point
                        ),
                    )
                    for index, point in enumerate(((0, 0), (1, 0), (1, 1)))
                ),
            ),
            RationalVPolytope(
                space=RationalCoordinateSpace(axes=("x", "y")),
                vertices=tuple(
                    RationalPolytopeVertex(
                        vertex_id=f"b{index}",
                        coordinates=tuple(
                            CanonicalRational(num=v, den=1) for v in point
                        ),
                    )
                    for index, point in enumerate(((0, 0), (1, 1), (0, 1)))
                ),
            ),
        )
    )
    cells = tuple(sorted(complex_value.maximal_cells, key=lambda cell: cell.cell_id))
    polynomial = RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(1, 0)
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=2), exponents=(0, 1)
                ),
            )
        ),
    )
    function = piecewise_polynomial_from_maximal_pieces(
        complex_value,
        tuple(
            PieceAssignment(cell_id=cell.cell_id, polynomial=polynomial)
            for cell in cells
        ),
    )
    result = spline_coordinates(
        SplineCoordinatesRequest(function=function, degree=1, smoothness=0)
    )

    axis = result.spline_space.coefficient_axis
    source_vector = tuple(
        next(
            (
                term.coefficient.as_fraction()
                for piece in function.pieces
                if piece.cell_id == cell_id
                for term in piece.polynomial.polynomial.terms
                if term.exponents == monomial
            ),
            Fraction(0),
        )
        for cell_id, monomial in axis
    )
    basis = tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in result.spline_space.nullspace_basis.entries
    )
    reconstructed = tuple(
        sum(
            (
                coordinate.as_fraction() * basis[row][column]
                for row, coordinate in enumerate(result.basis_coordinates)
            ),
            Fraction(0),
        )
        for column in range(len(axis))
    )
    assert result.spline_space.complex == complex_value
    assert result.spline_space.complex.dimension == 2
    assert reconstructed == source_vector


def test_piece_count_is_rejected_before_assignment_canonicalization(monkeypatch):
    function = _two_interval_function(((0, 1),), ((0, 1),))
    malformed = PieceAssignment.model_construct(cell_id="invalid", polynomial=None)
    forged = PiecewisePolynomialResult.model_construct(
        complex=function.complex,
        pieces=(malformed,) * 17,
        compatibility=(),
        status="COMPATIBLE",
        obstruction_face_id=None,
        obstruction_difference=None,
    )

    def should_not_canonicalize(*_args, **_kwargs):
        raise AssertionError(
            "piece assignments were canonicalized before count admission"
        )

    monkeypatch.setattr(PieceAssignment, "model_validate", should_not_canonicalize)
    with pytest.raises(OperationResourceAdmissionError, match="too many pieces"):
        spline_coordinates(
            SplineCoordinatesRequest(function=forged, degree=1, smoothness=0)
        )


def test_coordinates_reject_exact_smoothness_and_degree_failures():
    continuous_not_c1 = _two_interval_function((), ((1, 1), (0, -1)))
    with pytest.raises(OperationDomainValidationError, match=r"C\^r interface"):
        spline_coordinates(
            SplineCoordinatesRequest(function=continuous_not_c1, degree=1, smoothness=1)
        )

    too_high_degree = _two_interval_function(((2, 1),), ((2, 1),))
    with pytest.raises(OperationDomainValidationError, match="piece degree"):
        spline_coordinates(
            SplineCoordinatesRequest(function=too_high_degree, degree=1, smoothness=0)
        )


def test_discontinuous_piecewise_value_uses_the_unconstrained_spline_slice():
    function = _two_interval_function((), ((0, 1),))
    assert function.status == "INCOMPATIBLE"
    result = spline_coordinates(
        SplineCoordinatesRequest(function=function, degree=0, smoothness=-1)
    )
    assert result.spline_space.nullity == 2
    assert result.basis_coordinates == (
        CanonicalRational(num=0, den=1),
        CanonicalRational(num=1, den=1),
    )
    evaluated = spline_evaluate(
        SplineEvaluationRequest(
            complex=result.spline_space.complex,
            degree=result.spline_space.degree,
            smoothness=result.spline_space.smoothness,
            basis_coefficients=result.basis_coordinates,
            point=ComplexPoint(coordinates=(CanonicalRational(num=1, den=4),)),
        )
    )
    assert evaluated.value == CanonicalRational(num=0, den=1)


def test_forged_continuity_status_does_not_establish_spline_membership():
    discontinuous = _two_interval_function((), ((0, 1),))
    forged = PiecewisePolynomialResult.model_construct(
        complex=discontinuous.complex,
        pieces=discontinuous.pieces,
        compatibility=(),
        status="COMPATIBLE",
        obstruction_face_id=None,
        obstruction_difference=None,
    )
    with pytest.raises(OperationDomainValidationError, match="interface conditions"):
        spline_coordinates(
            SplineCoordinatesRequest(function=forged, degree=0, smoothness=0)
        )


def test_coordinate_output_is_admitted_before_nullspace_materialization(monkeypatch):
    function = _two_interval_function(((0, 1),), ((0, 1),))
    monkeypatch.setattr(spline_kernel, "MAX_SPLINE_COORDINATE_OUTPUT_BYTES", 1)

    def unexpected_basis(*_args, **_kwargs):
        raise AssertionError("basis must not be materialized after output rejection")

    monkeypatch.setattr(spline_kernel, "_spline_space_from_data", unexpected_basis)
    with pytest.raises(OperationResourceAdmissionError, match="output envelope"):
        spline_kernel.spline_coordinates(
            SplineCoordinatesRequest(function=function, degree=1, smoothness=0)
        )


def test_catalog_exposes_piecewise_to_spline_coordinates():
    function = _two_interval_function(((0, 1),), ((0, 1),))
    operation_id = "polyhedral_complex.spline.coordinates.compute"
    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    invoked = invoke_operation(
        operation.operation_id,
        {
            "function": function.model_dump(mode="json"),
            "degree": 1,
            "smoothness": 0,
        },
        catalog,
    )
    assert invoked.output["spline_space"]["nullity"] == 3
    assert len(invoked.output["basis_coordinates"]) == 3
