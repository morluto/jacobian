"""Exact native integers and their strict, lossless JSON boundary."""

import json
from typing import Annotated, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from jacobian._exact import DecimalIntegerEncoding
from jacobian.canonical import encode_strict_json

INTEGER: TypeAdapter[int] = TypeAdapter(
    Annotated[int, DecimalIntegerEncoding(max_digits=20)]
)


@pytest.mark.parametrize("value", [0, 42, -42, 2**53 + 1, -(2**53 + 1)])
def test_native_integer_and_json_string_round_trip(value: int) -> None:
    native = INTEGER.validate_python(value)
    assert type(native) is int
    assert INTEGER.dump_python(native) == value
    assert INTEGER.dump_python(native, mode="json") == str(value)
    encoded = encode_strict_json(INTEGER.dump_python(native, mode="json"))
    assert INTEGER.validate_json(encoded, strict=True) == value
    assert INTEGER.validate_json(INTEGER.dump_json(native)) == value


@pytest.mark.parametrize("value", ["42", 42.0, True, None])
def test_python_validation_does_not_decode_or_coerce(value: object) -> None:
    with pytest.raises(ValidationError):
        INTEGER.validate_python(value)


@pytest.mark.parametrize(
    "value",
    [
        42,
        42.0,
        True,
        None,
        "01",
        "-0",
        "+1",
        "1e3",
        " 1",
        "1\n",
        "",
        "\u0661",
        "\uff11",
    ],
)
def test_json_requires_canonical_decimal_strings(value: object) -> None:
    with pytest.raises(ValidationError):
        INTEGER.validate_json(json.dumps(value))


@pytest.mark.parametrize("sign", [1, -1])
def test_digit_bound_matches_between_python_and_json(sign: int) -> None:
    accepted = sign * (10**20 - 1)
    assert INTEGER.validate_python(accepted) == accepted
    assert INTEGER.validate_json(encode_strict_json(str(accepted))) == accepted
    rejected = sign * 10**20
    with pytest.raises(ValidationError):
        INTEGER.validate_python(rejected)
    with pytest.raises(ValidationError):
        INTEGER.validate_json(encode_strict_json(str(rejected)))


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_json_schemas_publish_the_encoding(
    mode: Literal["validation", "serialization"],
) -> None:
    schema = INTEGER.json_schema(mode=mode)
    assert schema["type"] == "string"
    assert schema["maxLength"] == 21
    validator = Draft202012Validator(schema)
    for value in ("0", "9" * 20, "-" + "9" * 20):
        assert validator.is_valid(value)
    for invalid in ("9" * 21, "-" + "9" * 21, "1\n", "01", "-0", "\u0661", 42):
        assert not validator.is_valid(invalid)
