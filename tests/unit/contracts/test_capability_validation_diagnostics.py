from __future__ import annotations

import pytest

from jacobian.capability_errors import PayloadValidationError
from jacobian.capability_validation import validate_payload
from jacobian.contracts.capabilities import CapabilityDiagnostic


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


@pytest.mark.parametrize(
    ("validator_name", "schema_property", "payload_value"),
    [
        ("pattern", {"type": "string", "pattern": "a" * 2000}, "b"),
        ("enum", {"type": "string", "enum": ["a" * 2000]}, "b"),
        ("const", {"type": "string", "const": "a" * 2000}, "b"),
    ],
)
def test_payload_validation_bounds_long_constraint_details(
    validator_name: str, schema_property: dict[str, object], payload_value: str
) -> None:
    """A large pattern/enum/const must not be copied unbounded into details."""

    with pytest.raises(PayloadValidationError) as error:
        validate_payload(
            {
                "type": "object",
                "properties": {"s": schema_property},
                "required": ["s"],
                "additionalProperties": False,
            },
            {"s": payload_value},
        )

    # The validator identity must survive the summary.
    assert error.value.details["validator"] == validator_name
    constraint = error.value.details["constraint"]
    assert isinstance(constraint, str)
    # The constraint is summarized, not copied verbatim.
    assert len(constraint) <= 1024
    assert constraint.endswith("...")


def test_payload_validation_bounds_serialized_diagnostic_size() -> None:
    """The serialized diagnostic stays small even for a large schema pattern."""

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

    diagnostic = CapabilityDiagnostic(
        code="INVALID_REQUEST",
        stage="capability_input_validation",
        message="The capability input does not match its advertised schema at s.",
        path="s",
        expected=error.value.expected,
        actual_type=error.value.actual_type,
        hint="Correct the reported field.",
        details=error.value.details,
    )
    # failed_result serializes this diagnostic in both output.error and
    # diagnostics, so a bounded dump keeps the whole response bounded.
    assert len(diagnostic.model_dump_json()) <= 4096


def test_payload_validation_reports_missing_nested_required_member() -> None:
    """A nested object omitting a required member must name the missing key."""

    with pytest.raises(PayloadValidationError) as error:
        validate_payload(
            {
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {"inner": {"type": "integer"}},
                        "required": ["inner"],
                    }
                },
                "required": ["outer"],
                "additionalProperties": False,
            },
            {"outer": {}},
        )

    assert error.value.path == "outer"
    assert error.value.details == {
        "validator": "required",
        "constraint": ["inner"],
    }


def test_payload_validation_reports_additional_properties_constraint() -> None:
    """An unexpected key must surface the additionalProperties constraint."""

    with pytest.raises(PayloadValidationError) as error:
        validate_payload(
            {
                "type": "object",
                "properties": {"known": {"type": "integer"}},
                "required": ["known"],
                "additionalProperties": False,
            },
            {"known": 1, "extra": "x"},
        )

    assert error.value.details == {
        "validator": "additionalProperties",
        "constraint": False,
    }


def test_payload_validation_bounds_large_required_constraint() -> None:
    """A large required list must be summarized, not copied unbounded."""

    required_keys = [f"key{i}" for i in range(2000)]
    with pytest.raises(PayloadValidationError) as error:
        validate_payload(
            {
                "type": "object",
                "properties": {"key0": {"type": "integer"}},
                "required": required_keys,
            },
            {"key0": 1},
        )

    assert error.value.details["validator"] == "required"
    constraint = error.value.details["constraint"]
    assert isinstance(constraint, str)
    assert len(constraint) <= 1024
    assert constraint.endswith("...")
