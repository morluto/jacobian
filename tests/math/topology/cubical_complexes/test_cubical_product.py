"""Cartesian products of cubical complexes."""

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalProductRequest,
    CubicalProductResult,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS
from jacobian.math.topology.cubical_complexes.operations import f_vector, product


def test_interval_times_point_is_exact_canonical_and_composable() -> None:
    result = product(
        (CubicalCell(intervals=((0, 1),)),),
        (CubicalCell(intervals=((5, 5),)),),
    )

    assert result.left_ambient_dimension == 1
    assert result.right_ambient_dimension == 1
    assert result.complex.ambient_dimension == 2
    assert result.complex.cells == tuple(
        sorted(
            (
                CubicalCell(intervals=((0, 0), (5, 5))),
                CubicalCell(intervals=((1, 1), (5, 5))),
                CubicalCell(intervals=((0, 1), (5, 5))),
            ),
            key=lambda cell: cell.intervals,
        )
    )

    decoded = CubicalProductResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    # Existing consumers accept the product's canonical cell representation.
    chain = f_vector(decoded.complex.cells)
    assert chain.f_vector.counts == (2, 1, 0)
    assert chain.euler_characteristic == 1


def test_square_times_interval_has_the_cube_f_vector() -> None:
    result = product(
        (CubicalCell(intervals=((0, 1), (0, 1))),),
        (CubicalCell(intervals=((3, 4),)),),
    )

    assert result.complex.ambient_dimension == 3
    vector = f_vector(result.complex.cells)
    assert vector.f_vector.counts == (8, 12, 6, 1)
    assert vector.euler_characteristic == 1


def test_product_tool_is_published_with_typed_example() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "cubical.product.compute"
    )
    request = CubicalProductRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.complex.ambient_dimension == 2
    assert f_vector(result.complex.cells).f_vector.counts == (2, 1, 0)


def test_product_rejects_ambient_axis_growth_before_expansion() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        product(
            (CubicalCell(intervals=((0, 1),) * 6),),
            (CubicalCell(intervals=((0, 1),) * 5),),
        )
    assert (
        error.value.errors()[0]["type"] == "cubical_complex.product_ambient_dimension"
    )


def test_product_admits_aggregate_cell_growth_before_product_allocation() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        product(
            (CubicalCell(intervals=((0, 1),) * 5),),
            (CubicalCell(intervals=((0, 1),) * 5),),
        )
    assert error.value.errors()[0]["type"] == "cubical_complex.product_cell_budget"


def test_product_admits_coordinate_output_size_before_product_allocation() -> None:
    large_coordinate = 10**999
    with pytest.raises(OperationResourceAdmissionError) as error:
        product(
            (CubicalCell(intervals=((large_coordinate, large_coordinate + 1),) * 4),),
            (CubicalCell(intervals=((large_coordinate, large_coordinate + 1),) * 4),),
        )
    assert error.value.errors()[0]["type"] == "cubical_complex.product_result_size"
