"""Schema validation owned by capability registration and dispatch."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.capability_errors import CapabilityError, PayloadValidationError
from jacobian.schema_validation import check_draft202012_schema


@lru_cache(maxsize=1024)
def compiled_validator(canonical_schema: bytes) -> Draft202012Validator:
    normalized = loads_strict_json(canonical_schema)
    try:
        check_draft202012_schema(canonical_schema)
    except SchemaError as exc:
        raise CapabilityError("capability JSON Schema is invalid") from exc
    return Draft202012Validator(normalized)


def validator(schema: dict[str, object]) -> Draft202012Validator:
    return compiled_validator(canonicalize_json(schema))


def validate_payload(
    schema: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    try:
        normalized = loads_strict_json(canonicalize_json(payload))
    except CanonicalizationError as exc:
        raise PayloadValidationError(
            str(exc),
            path="$",
            actual_type=json_value_type(payload),
            expected="bounded canonical JSON",
        ) from exc
    if not isinstance(normalized, dict):
        raise CapabilityError("capability payload must normalize to an object")
    errors = sorted(
        validator(schema).iter_errors(normalized),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "$"
        raise PayloadValidationError(
            f"{location}: {first.message}",
            path=location,
            actual_type=json_value_type(first.instance),
            expected=schema_expectation(first),
            details=schema_violation_details(first),
        )
    return normalized


def json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def schema_expectation(error: JsonSchemaValidationError) -> str:
    if error.validator == "enum" and isinstance(error.validator_value, list):
        allowed = ", ".join(
            json.dumps(value, ensure_ascii=False) for value in error.validator_value
        )
        return f"one of: {allowed}"
    if error.validator == "const":
        return f"the constant {json.dumps(error.validator_value, ensure_ascii=False)}"
    if error.validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            return "JSON type " + " or ".join(str(item) for item in expected)
        return f"JSON type {expected}"
    constraint_labels = {
        "minimum": "a number greater than or equal to",
        "exclusiveMinimum": "a number greater than",
        "maximum": "a number less than or equal to",
        "exclusiveMaximum": "a number less than",
        "minLength": "a string with minimum length",
        "maxLength": "a string with maximum length",
        "minItems": "an array with minimum length",
        "maxItems": "an array with maximum length",
        "multipleOf": "a number that is a multiple of",
        "pattern": "a string matching pattern",
    }
    label = constraint_labels.get(str(error.validator))
    if label is not None:
        rendered = json.dumps(error.validator_value, ensure_ascii=False)
        # Truncate to stay within CapabilityDiagnostic.expected's 1024-char limit
        max_value = 1024 - len(label) - 1
        if len(rendered) > max_value:
            rendered = rendered[: max(0, max_value - 3)] + "..."
        return f"{label} {rendered}"
    return "input matching the capability descriptor JSON Schema"


def schema_violation_details(error: JsonSchemaValidationError) -> dict[str, object]:
    """Return the exact public schema constraint that rejected one value."""

    details: dict[str, object] = {"validator": str(error.validator)}
    if error.validator in {
        "minimum",
        "exclusiveMinimum",
        "maximum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "multipleOf",
        "pattern",
        "enum",
        "const",
        "type",
    }:
        constraint_value = error.validator_value
        if isinstance(constraint_value, str):
            if len(constraint_value) > 1024:
                constraint_value = constraint_value[:1021] + "..."
        else:
            rendered = json.dumps(constraint_value, ensure_ascii=False)
            if len(rendered) > 1024:
                constraint_value = rendered[:1021] + "..."
        details["constraint"] = constraint_value
    return details


__all__ = [
    "compiled_validator",
    "json_value_type",
    "schema_expectation",
    "schema_violation_details",
    "validate_payload",
    "validator",
]
