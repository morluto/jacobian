"""Advertised polytope and piecewise-polynomial examples through the catalog."""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import (
    polytopal_complex_affine_transform,
    polytopal_complex_closure,
    spline_dimension,
)
from jacobian.math.geometry.polytopes.complexes import (
    polytopal_complex_affine_transform as exported_affine_transform,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    PiecewisePolynomialResult,
    SplineDimensionRequest,
    SplineDimensionResult,
)
from jacobian.math.matrices.cyclic_linear import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _zero(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            {"num": 0, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


_OPERATION_ID = "polytope.face_lattice.compute"


def _coefficient_map(polynomial: RationalPolynomial):
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _interval(left: int, right: int, prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=(CanonicalRational(num=value, den=1),),
            )
            for index, value in enumerate((left, right))
        ),
    )


def test_common_refinement_catalog_example_is_composable():
    catalog = Catalog.open()
    operation = catalog.operation("polytopal_complex.common_refinement.compute")
    example = operation.examples[0]
    result = invoke_operation(operation.operation_id, example.input, catalog)
    assert result.output["cell_pairs"][0]["refined_cell_id"] == "M0"


def test_affine_transform_catalog_example_roundtrips():
    catalog = Catalog.open()
    operation = catalog.operation("polytopal_complex.affine_transform.compute")
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    target = result.output["target"]
    assert target["maximal_cells"][0]["vertices"] == [
        {"coordinates": [{"num": "3", "den": "1"}]},
        {"coordinates": [{"num": "5", "den": "1"}]},
    ]
    assert len(result.output["cell_transport"]) == 1
    assert exported_affine_transform is polytopal_complex_affine_transform


def test_face_lattice_operation_is_catalogued_and_runs_its_example() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_OPERATION_ID)
    assert operation is not None
    result = invoke_operation(_OPERATION_ID, operation.examples[0].input, catalog)
    assert result.output["faces"]
    assert len(result.output["covers"]) == 32


def test_catalog_addition_example_executes_through_typed_contract():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.add.compute")
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert _coefficient_map(validated.pieces[0].polynomial) == {(0,): Fraction(3)}


def test_catalog_piecewise_multiplication_example_executes():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.multiply.compute")
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert _coefficient_map(validated.pieces[0].polynomial) == {
        (2,): Fraction(1),
        (1,): Fraction(1),
    }


def test_one_cell_zero_row_dimension_roundtrips_and_catalog_invokes():
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    request = SplineDimensionRequest(complex=complex_value, degree=5, smoothness=1)

    result = spline_dimension(request)
    replayed = SplineDimensionResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )

    assert result.compatibility_matrix.row_count == 0
    assert result.compatibility_matrix.column_count == 6
    assert result.rank == 0 and result.nullity == 6
    assert replayed == result
    invoked = invoke_operation(
        "polyhedral_complex.spline_dimension.compute",
        {
            "complex": complex_value.model_dump(mode="json"),
            "degree": 0,
            "smoothness": 0,
        },
        Catalog.open(),
    )
    assert invoked.output["nullity"] == 1 and invoked.output["rank"] == 0
