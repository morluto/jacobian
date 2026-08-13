"""Public validation diagnostics must be bounded and input-safe."""

import pytest
from pydantic import ValidationError, create_model

from jacobian.validation_diagnostics import (
    bounded_validation_exception_message,
    project_validation_errors,
    validation_error_message,
)


def test_projection_hides_rejected_values_and_bounds_error_count() -> None:
    invalid_model = create_model(
        "InvalidDiagnosticFixture",
        **{f"field_{index}": (int, ...) for index in range(16)},
    )
    with pytest.raises(ValidationError) as exc_info:
        invalid_model.model_validate({"field_0": "private_marker"})

    errors, count = project_validation_errors(exc_info.value)

    assert count == 16
    assert len(errors) == 8
    assert all("input" not in error for error in errors)
    assert "private_marker" not in str(errors)
    assert "private_marker" not in validation_error_message(exc_info.value)


def test_non_pydantic_detail_is_bounded() -> None:
    message = bounded_validation_exception_message(ValueError("x" * 2_000))

    assert len(message) == 1_024
    assert message.endswith("... [validation detail truncated]")
