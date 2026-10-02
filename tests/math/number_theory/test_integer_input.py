"""Request spelling convenience preserves exact integer and owner bounds."""

import json
from typing import Annotated, Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from jacobian.math.number_theory._derived_models import FloorSquareRootRequest
from jacobian.math.number_theory._divisibility_models import IntegerPairRequest
from jacobian.math.number_theory._integer_input import (
    IntegerInput,
    IntegerInputEncoding,
)
from jacobian.math.number_theory._integer_models import (
    MAX_SAFE_INTEGER,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.math.number_theory._prime_models import (
    PreviousPrimeRequest,
    PrimorialRequest,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue

INTEGER = TypeAdapter(IntegerInput)


@pytest.mark.parametrize("value", (0, 6, -6, MAX_SAFE_INTEGER, -MAX_SAFE_INTEGER))
def test_safe_json_integers_and_strings_have_one_native_value(value: int) -> None:
    for source in (value, str(value)):
        parsed = INTEGER.validate_json(json.dumps(source), strict=True)
        assert type(parsed) is int
        assert parsed == value
        assert INTEGER.dump_python(parsed) == value
        assert INTEGER.dump_python(parsed, mode="json") == str(value)


@pytest.mark.parametrize("sign", (-1, 1))
def test_large_exact_strings_remain_lossless_but_numeric_tokens_are_refused(
    sign: int,
) -> None:
    for magnitude in (MAX_SAFE_INTEGER + 1, 10**256 - 1):
        value = sign * magnitude
        assert INTEGER.validate_json(json.dumps(str(value))) == value
        assert INTEGER.validate_python(value) == value
        with pytest.raises(ValidationError):
            INTEGER.validate_json(json.dumps(value))
    with pytest.raises(ValidationError):
        INTEGER.validate_json(json.dumps(str(sign * 10**256)))
    with pytest.raises(ValidationError):
        INTEGER.validate_python(sign * 10**256)


@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        6.0,
        6.5,
        None,
        "",
        "+6",
        "06",
        "-0",
        " 6",
        "6 ",
        "6\n",
        "1e3",
        "\uff16",
        "\u0666",
    ),
)
def test_json_request_codec_keeps_the_canonical_integer_grammar(value: Any) -> None:
    with pytest.raises(ValidationError):
        INTEGER.validate_json(json.dumps(value))


@pytest.mark.parametrize("value", ("6", True, False, 6.0, 6.5, None))
def test_python_request_validation_remains_strict(value: Any) -> None:
    with pytest.raises(ValidationError):
        INTEGER.validate_python(value)


@pytest.mark.parametrize(
    ("model", "minimum", "maximum"),
    (
        (NonnegativeIntegerRequest, 0, 10_000),
        (PositiveIntegerRequest, 1, 10_000),
        (PreviousPrimeRequest, 3, MAX_SAFE_INTEGER),
        (PrimorialRequest, 1, 1001),
        (FloorSquareRootRequest, 0, MAX_SAFE_INTEGER),
    ),
)
def test_each_consumer_preserves_its_exact_native_and_wire_bounds(
    model: Any, minimum: int, maximum: int
) -> None:
    schema = Draft202012Validator(model.model_json_schema())
    for accepted in (minimum, maximum):
        native = model(n=accepted)
        assert native.model_dump() == {"n": accepted}
        for spelling in (accepted, str(accepted)):
            assert schema.is_valid({"n": spelling})
            decoded = model.model_validate_json(
                json.dumps({"n": spelling}), strict=True
            )
            assert decoded == native
            assert model.model_validate_json(decoded.model_dump_json()) == native
    for rejected in (minimum - 1, maximum + 1):
        with pytest.raises(ValidationError):
            model(n=rejected)
        for spelling in (rejected, str(rejected)):
            assert not schema.is_valid({"n": spelling})
            with pytest.raises(ValidationError):
                model.model_validate_json(json.dumps({"n": spelling}), strict=True)
    # Size admission runs before decimal decoding, even for zero-looking input.
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps({"n": "0" * 100_000}))


def test_request_convenience_does_not_weaken_canonical_result_decoding() -> None:
    request = IntegerPairRequest.model_validate_json('{"left":12,"right":"18"}')
    assert request.model_dump() == {"left": 12, "right": 18}
    assert request.model_dump(mode="json") == {"left": "12", "right": "18"}
    assert IntegerValue.model_validate_json('{"value":"6"}').value == 6
    for invalid in (6, 6.0, True, "+6", "06"):
        with pytest.raises(ValidationError):
            IntegerValue.model_validate_json(json.dumps({"value": invalid}))


def test_request_and_result_schemas_publish_their_distinct_wire_contracts() -> None:
    validation = Draft202012Validator(INTEGER.json_schema(mode="validation"))
    serialization = Draft202012Validator(INTEGER.json_schema(mode="serialization"))
    for accepted in ("-6", "0", "6", "9" * 256, 6, -MAX_SAFE_INTEGER):
        assert validation.is_valid(accepted)
    for rejected in (
        True,
        6.5,
        "06",
        "-0",
        "+6",
        "6\n",
        "9" * 257,
        MAX_SAFE_INTEGER + 1,
    ):
        assert not validation.is_valid(rejected)
    assert serialization.is_valid("6")
    assert not serialization.is_valid(6)
    for model, minimum, maximum in (
        (NonnegativeIntegerRequest, 0, 10_000),
        (PositiveIntegerRequest, 1, 10_000),
        (PreviousPrimeRequest, 3, MAX_SAFE_INTEGER),
        (PrimorialRequest, 1, 1001),
        (FloorSquareRootRequest, 0, MAX_SAFE_INTEGER),
    ):
        field = model.model_json_schema()["properties"]["n"]
        numeric = next(
            branch for branch in field["anyOf"] if branch["type"] == "integer"
        )
        assert numeric["minimum"] == minimum
        assert numeric["maximum"] == maximum
        assert f"[{minimum}, {maximum}]" in field["description"]


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    ((None, None), (0, 0), (3, 17), (-17, -3), (-17, 23), (17, None), (None, -3)),
)
def test_exhaustive_small_integer_interval_schema_matches_runtime(
    minimum: int | None, maximum: int | None
) -> None:
    adapter: TypeAdapter[int] = TypeAdapter(
        Annotated[
            int, IntegerInputEncoding(max_digits=2, minimum=minimum, maximum=maximum)
        ]
    )
    validator = Draft202012Validator(adapter.json_schema())
    serialization = Draft202012Validator(adapter.json_schema(mode="serialization"))
    for value in range(-105, 106):
        expected = (
            abs(value) < 100
            and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum)
        )
        for spelling in (value, str(value)):
            assert validator.is_valid(spelling) is expected
            if expected:
                assert adapter.validate_json(json.dumps(spelling)) == value
            else:
                with pytest.raises(ValidationError):
                    adapter.validate_json(json.dumps(spelling))
        assert serialization.is_valid(str(value)) is expected
