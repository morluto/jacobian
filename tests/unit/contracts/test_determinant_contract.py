from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.matrix_operations import (
    MatrixDeterminantRequest,
    _determinant_input_scalar_digits,
)
from jacobian.schema_registry import model_schema


def test_determinant_schema_publishes_64_by_64_input_bound() -> None:
    schema = model_schema(MatrixDeterminantRequest)
    matrix = schema["$defs"]["DeterminantRationalMatrix"]
    assert matrix["properties"]["entries"]["maxItems"] == 64
    row = matrix["properties"]["entries"]["items"]
    assert row["maxItems"] == 64


def test_determinant_scalar_budget_scales_to_the_inline_output_cap() -> None:
    assert _determinant_input_scalar_digits(1) == 256
    assert _determinant_input_scalar_digits(64) == 7

    denominator = str(10**255 + 1000)
    entries = tuple(
        tuple({"num": "1", "den": denominator} for _ in range(33))
        for _ in range(33)
    )

    with pytest.raises(ValidationError, match="determinant input"):
        MatrixDeterminantRequest.model_validate({"matrix": {"entries": entries}})
