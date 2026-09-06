"""Smith admission retains zero axes through the full declared operation limit."""

import pytest
from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.operations import smith_normal_form_result
from jacobian.math.matrices.values import IntegerMatrix, integer_matrix_axis_schema


@pytest.mark.parametrize("shape", [(0, 33), (33, 0), (0, 64), (64, 0)])
def test_empty_smith_at_admitted_boundary(shape: tuple[int, int]) -> None:
    rows, columns = shape
    matrix = IntegerMatrix(
        row_count=rows, column_count=columns, entries=tuple(() for _ in range(rows))
    )
    native = smith_normal_form_result(matrix)
    output = invoke_operation(
        "matrix.normal_form.smith.compute",
        {"matrix": matrix.model_dump(mode="json")},
        Catalog.open(),
    ).output
    assert output == native.model_dump(mode="json")
    assert native.normal_form == matrix
    assert native.rank == 0
    assert native.invariant_factors == ()


@pytest.mark.parametrize("shape", [(0, 65), (65, 0)])
def test_empty_smith_rejects_excessive_axes(shape: tuple[int, int]) -> None:
    rows, columns = shape
    matrix = IntegerMatrix(
        row_count=rows, column_count=columns, entries=tuple(() for _ in range(rows))
    )
    with pytest.raises(OperationDomainValidationError, match="64 rows and columns"):
        smith_normal_form_result(matrix)


@pytest.mark.parametrize("maximum", [32, 64])
@pytest.mark.parametrize("empty_rows", [False, True])
def test_integer_schema_caps_explicit_axes(maximum: int, empty_rows: bool) -> None:
    validator = Draft202012Validator(integer_matrix_axis_schema(maximum))
    for size in (maximum, maximum + 1):
        rows, columns = (0, size) if empty_rows else (size, 0)
        value = {
            "domain": "ZZ",
            "row_count": rows,
            "column_count": columns,
            "entries": [[] for _ in range(rows)],
        }
        assert bool(list(validator.iter_errors(value))) == (size > maximum)
