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
