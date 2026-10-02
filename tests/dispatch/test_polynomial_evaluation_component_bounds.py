"""Both public evaluation regimes retain canonical identity points."""

import json

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation


@pytest.mark.parametrize("variables", (("x",), ("x", "y")))
@pytest.mark.parametrize(
    "kind",
    ("reciprocal", "neighbor", "balanced", "integer", "negative", "max_balanced"),
)
def test_identity_evaluation_accepts_canonical_component_boundaries(
    variables: tuple[str, ...], kind: str
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
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [1] + [0] * (len(variables) - 1),
                    }
                ]
            },
        },
        "point": {
            "variables": list(variables),
            "values": [wire_point] + [{"num": "0", "den": "1"}] * (len(variables) - 1),
        },
    }
    result = invoke_operation("polynomial.map.evaluate", payload, Catalog.open())
    decoded = CanonicalRational.model_validate_json(json.dumps(result.output["value"]))
    assert decoded == point


@pytest.mark.parametrize("variables", (("x",), ("x", "y")))
@pytest.mark.parametrize("overflow", (False, True))
def test_square_evaluation_preserves_true_growth_boundary(
    variables: tuple[str, ...], overflow: bool
) -> None:
    native_point = CanonicalRational(
        num=1, den=10 ** (MAX_CANONICAL_RATIONAL_DIGITS // 2 - int(not overflow))
    )
    point = native_point.model_dump(mode="json")
    payload = {
        "polynomial": {
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2] + [0] * (len(variables) - 1),
                    }
                ]
            },
        },
        "point": {
            "variables": list(variables),
            "values": [point] + [{"num": "0", "den": "1"}] * (len(variables) - 1),
        },
    }
    if not overflow:
        result = invoke_operation("polynomial.map.evaluate", payload, Catalog.open())
        decoded = CanonicalRational.model_validate_json(
            json.dumps(result.output["value"])
        )
        assert decoded.as_fraction() == native_point.as_fraction() ** 2
        return
    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation("polynomial.map.evaluate", payload, Catalog.open())
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.evaluation_result_exceeds_component_bound"
    )
