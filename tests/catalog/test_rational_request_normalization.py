"""Registry-complete rational request schemas stay distinct from canonical values."""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.dispatch._rational_request_cases import (
    RATIONAL_REQUEST_CASES,
    RationalRequestCase,
    ratio,
)

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.math.matrices._operation_models import (
    MatrixRankRequest,
    RationalLinearSolveRequest,
    SquareRationalMatrixRequest,
    _MatrixRequest,
)
from jacobian.math.matrices.rational_linear._models import LinearRationalSystem
from jacobian.math.matrices.values import RationalMatrix, SparseRationalMatrix


def _contains(annotation: Any, carrier: type[Any]) -> bool:
    return annotation is carrier or any(
        _contains(arg, carrier) for arg in get_args(annotation)
    )


def test_inventory_covers_every_published_sibling_and_argument_branch() -> None:
    found: set[tuple[str, str, bool]] = set()
    for operation in BUILTIN_TOOLS:
        request = operation.request_type
        if issubclass(request, _MatrixRequest):
            for name, field in request.model_fields.items():
                for carrier, sparse in [
                    (RationalMatrix, False),
                    (SparseRationalMatrix, True),
                    (CanonicalRational, False),
                ]:
                    if _contains(field.annotation, carrier):
                        found.add((operation.operation_id, name, sparse))
        elif request.__module__ == "jacobian.math.matrices.rational_linear._models":
            for field in request.model_fields.values():
                assert field.annotation is LinearRationalSystem
                found.update(
                    {
                        (operation.operation_id, "coefficients", True),
                        (operation.operation_id, "rhs", False),
                    }
                )
    assert found == {
        (case.operation_id, case.argument, case.sparse)
        for case in RATIONAL_REQUEST_CASES
    }
    assert len(found) == 19


@pytest.mark.parametrize("case", RATIONAL_REQUEST_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "num,den", [("2", "4"), ("2", "-4"), ("-2", "-4"), ("-" + "9" * 256, "9" * 256)]
)
def test_published_schema_accepts_exact_request_presentations(
    case: RationalRequestCase, num: str, den: str
) -> None:
    descriptor = Catalog.open().inspect(case.operation_id)
    assert descriptor is not None
    schema = descriptor.input_schema
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(case.payload(ratio(num, den)))
    operation = Catalog.open().operation(case.operation_id)
    assert operation is not None
    operation.request_type.model_validate_json(
        json.dumps(case.payload(ratio(num, den))), strict=True
    )


@pytest.mark.parametrize("case", RATIONAL_REQUEST_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "num,den",
    [
        ("1", "0"),
        ("1", "-0"),
        ("1", "+4"),
        ("01", "4"),
        ("1", "04"),
        ("٢", "4"),
        ("1", "4\n"),
        ("9" * 257, "9" * 257),
    ],
)
def test_published_schema_rejects_invalid_raw_components(
    case: RationalRequestCase, num: str, den: str
) -> None:
    operation = Catalog.open().operation(case.operation_id)
    assert operation is not None
    payload = case.payload(ratio(num, den))
    assert not Draft202012Validator(
        operation.request_type.model_json_schema(mode="validation")
    ).is_valid(payload)
    with pytest.raises(ValidationError):
        operation.request_type.model_validate_json(json.dumps(payload), strict=True)


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    return schema["$defs"][node["$ref"].rsplit("/", 1)[1]] if "$ref" in node else node


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in RATIONAL_REQUEST_CASES
        if case.operation_id.startswith("matrix.") and case.argument != "rhs"
    ],
)
def test_raw_axis_overlays_are_owner_and_representation_specific(
    case: RationalRequestCase,
) -> None:
    operation = Catalog.open().operation(case.operation_id)
    assert operation is not None
    schema = operation.request_type.model_json_schema(mode="validation")
    matrix = schema["properties"][case.argument]
    if "anyOf" in matrix:
        matrix = matrix["anyOf"][1 if case.sparse else 0]
    matrix = _resolve(schema, matrix)
    maximum = 8192 if case.sparse else operation.request_type._raw_matrix_axis_limit
    assert matrix["properties"]["row_count"]["maximum"] == maximum
    assert matrix["properties"]["column_count"]["maximum"] == maximum
    entries = matrix["properties"]["entries"]
    assert entries["maxItems"] == (32768 if case.sparse else maximum)
    if not case.sparse:
        assert entries["items"]["maxItems"] == maximum
    # Request serialization still describes the full canonical carrier.
    encoded = operation.request_type.model_json_schema(mode="serialization")
    encoded_matrix = encoded["properties"][case.argument]
    if "anyOf" in encoded_matrix:
        encoded_matrix = encoded_matrix["anyOf"][1 if case.sparse else 0]
    assert (
        _resolve(encoded, encoded_matrix)["properties"]["column_count"]["maximum"]
        == 8192
    )


def test_rhs_schema_limits_and_unused_square_request_remain_explicit() -> None:
    assert (
        RationalLinearSolveRequest.model_json_schema()["properties"]["rhs"]["maxItems"]
        == 32
    )
    schema = SquareRationalMatrixRequest.model_json_schema()
    assert schema["properties"]["matrix"]["properties"]["column_count"]["maximum"] == 32
    for operation_id in (
        "linear.rational_solution.compute",
        "linear.rational_inconsistency.compute",
    ):
        operation = Catalog.open().operation(operation_id)
        assert operation is not None
        schema = operation.request_type.model_json_schema()
        system = _resolve(schema, schema["properties"]["system"])
        assert system["properties"]["rhs"]["maxItems"] == 8192
        assert system["properties"]["variables"]["maxItems"] == 8192


def test_union_schema_does_not_alias_dense_and_sparse_envelopes() -> None:
    schema = MatrixRankRequest.model_json_schema()
    validator = Draft202012Validator(schema)
    assert validator.is_valid(
        {
            "matrix": {
                "row_count": 1,
                "column_count": 8192,
                "entries": [{"row": 0, "column": 8191, "value": ratio("2", "-4")}],
            }
        }
    )
    assert not validator.is_valid(
        {
            "matrix": {
                "row_count": 1,
                "column_count": 129,
                "entries": [[ratio("2", "-4")] * 129],
            }
        }
    )
