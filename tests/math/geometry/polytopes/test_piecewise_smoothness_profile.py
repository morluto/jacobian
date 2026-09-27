from jacobian._exact import CanonicalRational
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    PieceAssignment,
    PiecewiseSmoothnessRequest,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_smoothness,
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
                vertex_id=f"{prefix}{i}",
                coordinates=(CanonicalRational(num=x, den=1),),
            )
            for i, x in enumerate((left, right))
        ),
    )


def _polynomial(terms: tuple[tuple[int, int], ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=(degree,),
                )
                for coefficient, degree in terms
            )
        ),
    )


def test_facet_smoothness_uses_exact_divisibility_not_point_sampling():
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    left_cell, right_cell = sorted(
        complex_value.maximal_cells,
        key=lambda cell: min(point.coordinates[0].num for point in cell.vertices),
    )
    # The right piece (x-1)^2 has equal value and first derivative at x=1,
    # but its second derivative differs from the zero left piece.
    pieces = (
        PieceAssignment(cell_id=left_cell.cell_id, polynomial=_polynomial(())),
        PieceAssignment(
            cell_id=right_cell.cell_id,
            polynomial=_polynomial(((1, 2), (-2, 1), (1, 0))),
        ),
    )
    function = piecewise_polynomial_from_maximal_pieces(complex_value, pieces)
    result = piecewise_polynomial_smoothness(
        PiecewiseSmoothnessRequest(function=function, max_smoothness=4)
    )
    assert function.status == "COMPATIBLE"
    assert len(result.facets) == 1
    assert result.facets[0].smoothness == result.smoothness == 1
