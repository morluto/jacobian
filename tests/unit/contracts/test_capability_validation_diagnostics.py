from __future__ import annotations

import pytest

from jacobian.capability_errors import PayloadValidationError
from jacobian.capability_validation import validate_payload


def test_payload_validation_reports_exact_maximum_constraint() -> None:
    with pytest.raises(PayloadValidationError) as error:
        validate_payload(
            {
                "type": "object",
                "properties": {"n": {"type": "integer", "maximum": 1000}},
                "required": ["n"],
                "additionalProperties": False,
            },
            {"n": 2310},
        )

    assert error.value.path == "n"
    assert error.value.actual_type == "integer"
    assert error.value.expected == "a number less than or equal to 1000"
    assert error.value.details == {"validator": "maximum", "constraint": 1000}


def test_payload_validation_bounds_long_pattern_text() -> None:
    """A pattern longer than 1024 chars must not exceed the diagnostic limit."""

    long_pattern = "a" * 2000
    with pytest.raises(PayloadValidationError) as error:
        validate_payload(
            {
                "type": "object",
                "properties": {"s": {"type": "string", "pattern": long_pattern}},
                "required": ["s"],
                "additionalProperties": False,
            },
            {"s": "b"},
        )

    assert error.value.expected is not None
    assert len(error.value.expected) <= 1024
    assert error.value.expected.startswith("a string matching pattern ")
    assert error.value.expected.endswith("...")
