"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/polytopes/test_spline_coordinates.py``. The
preamble below is carried over so the moved test resolves every name it uses.
"""

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    PieceAssignment,
    PiecewisePolynomialResult,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_from_maximal_pieces,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
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


def _two_interval_function() -> PiecewisePolynomialResult:
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    cells = tuple(sorted(complex_value.maximal_cells, key=lambda cell: cell.cell_id))
    return piecewise_polynomial_from_maximal_pieces(
        complex_value,
        tuple(
            PieceAssignment(
                cell_id=cell.cell_id,
                polynomial=RationalPolynomial(
                    variables=("x",),
                    polynomial=SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=(1,),
                            ),
                        )
                    ),
                ),
            )
            for cell in cells
        ),
    )


def test_catalog_exposes_piecewise_to_spline_coordinates():
    function = _two_interval_function()
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
