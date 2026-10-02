"""Both public polynomial evaluators retain canonical identity points."""

import json

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation


@pytest.mark.parametrize(
    "operation_id", ("polynomial.rational.compute.evaluate", "polynomial.map.evaluate")
)
@pytest.mark.parametrize(
    "kind",
    ("reciprocal", "neighbor", "balanced", "integer", "negative", "max_balanced"),
)
def test_identity_evaluation_accepts_canonical_component_boundaries(
    operation_id: str, kind: str
) -> None:
    if kind in {"reciprocal", "neighbor"}:
        point = CanonicalRational(
            num=1,
            den=10
            ** (MAX_CANONICAL_RATIONAL_DIGITS - (1 if kind == "reciprocal" else 2)),
        )
    elif kind in {"balanced", "negative", "max_balanced"}:
        numerator = 10 ** (
            MAX_CANONICAL_RATIONAL_DIGITS - 1
            if kind == "max_balanced"
            else MAX_CANONICAL_RATIONAL_DIGITS // 2
        )
        point = CanonicalRational(
            num=numerator if kind != "negative" else -numerator, den=numerator + 1
        )
    else:
        point = CanonicalRational(num=10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1), den=1)
    wire_point = point.model_dump(mode="json")
    payload = {
        "polynomial": {
            "variables": ["x"],
            "polynomial": {
                "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [1]}]
            },
        },
        "point": wire_point
        if operation_id == "polynomial.rational.compute.evaluate"
        else {"variables": ["x"], "values": [wire_point]},
    }
    result = invoke_operation(operation_id, payload, Catalog.open())
    decoded = CanonicalRational.model_validate_json(json.dumps(result.output["value"]))
    assert decoded == point


@pytest.mark.parametrize(
    "operation_id", ("polynomial.rational.compute.evaluate", "polynomial.map.evaluate")
)
@pytest.mark.parametrize("overflow", (False, True))
def test_square_evaluation_preserves_true_growth_boundary(
    operation_id: str, overflow: bool
) -> None:
    native_point = CanonicalRational(
        num=1, den=10 ** (MAX_CANONICAL_RATIONAL_DIGITS // 2 - int(not overflow))
    )
    point = native_point.model_dump(mode="json")
    payload = {
        "polynomial": {
            "variables": ["x"],
            "polynomial": {
                "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [2]}]
            },
        },
        "point": point
        if operation_id == "polynomial.rational.compute.evaluate"
        else {"variables": ["x"], "values": [point]},
    }
    if not overflow:
        result = invoke_operation(operation_id, payload, Catalog.open())
        decoded = CanonicalRational.model_validate_json(
            json.dumps(result.output["value"])
        )
        assert decoded.as_fraction() == native_point.as_fraction() ** 2
        return
    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(operation_id, payload, Catalog.open())
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.evaluation_result_exceeds_component_bound"
    )
