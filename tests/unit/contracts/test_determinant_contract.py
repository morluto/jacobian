from __future__ import annotations

from jacobian.contracts.matrix_operations import MatrixDeterminantRequest
from jacobian.schema_registry import model_schema


def test_determinant_schema_publishes_64_by_64_input_bound() -> None:
    schema = model_schema(MatrixDeterminantRequest)
    matrix = schema["$defs"]["DeterminantRationalMatrix"]
    assert matrix["properties"]["entries"]["maxItems"] == 64
    row = matrix["properties"]["entries"]["items"]
    assert row["maxItems"] == 64
