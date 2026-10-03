"""Bounded owner diagnostics distinguish complete and truncated error lists."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp.types import TextContent
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, parse_operation_input
from jacobian.mcp.models import (
    OperationInvalidRequestData,
    OperationResourceAdmissionData,
    OperationValidationIssue,
)
from jacobian.mcp.server import create_server
from jacobian.mcp.tools import _invalid_request_error
from mcp import Client


def test_mcp_validation_error_counts_and_recovery() -> None:
    operation = Catalog.open().operation("integer.compute.gcd")
    assert operation is not None
    valid = {"left": 84, "right": 30}
    private_value = "private-diagnostic-sentinel"

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            for count in (0, 1, 64, 65, 81, 0):
                payload = {
                    **valid,
                    **{f"extra_{index}": private_value for index in range(count)},
                }
                if count:
                    with pytest.raises(ValidationError) as native:
                        parse_operation_input(operation.request_type, payload)
                    assert native.value.error_count() == count
                    records = native.value.errors(
                        include_url=False, include_context=False, include_input=False
                    )
                    assert len(records) == count

                response = await client.call_tool(
                    "math.run",
                    {"operation_id": operation.operation_id, "payload": payload},
                )
                if not count:
                    assert response.is_error is False
                    assert response.structured_content is not None
                    assert response.structured_content["output"] == {"value": "6"}
                    continue

                assert response.is_error is True
                assert response.structured_content is None
                assert len(response.content) == 1
                content = response.content[0]
                assert isinstance(content, TextContent)
                prefix = "Error executing tool math.run: "
                assert content.text.startswith(prefix)
                diagnostic = json.loads(content.text.removeprefix(prefix))
                assert diagnostic["code"] == "INVALID_REQUEST"
                assert diagnostic["stage"] == "operation_validation"
                assert diagnostic["operation_id"] == operation.operation_id
                assert diagnostic["errors"] == [
                    {
                        "location": list(record["loc"]),
                        "code": record["type"],
                        "message": record["msg"],
                    }
                    for record in records[:64]
                ]
                reported = len(diagnostic["errors"])
                omitted = diagnostic["omitted_error_count"]
                assert reported == min(count, 64)
                assert type(omitted) is int
                assert omitted == max(0, count - 64)
                assert reported + omitted == native.value.error_count()
                assert private_value not in content.text

    asyncio.run(scenario())


def test_validation_count_uses_records_once_not_distinct_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ValidationError.from_exception_data(
        "Repeated location",
        [
            {"type": "int_type", "loc": ("value",), "input": "private-value"}
            for _ in range(81)
        ],
    )
    error = OperationRequestValidationError(cause)
    original_errors = error.errors
    materializations = 0

    def errors() -> Sequence[Mapping[str, Any]]:
        nonlocal materializations
        materializations += 1
        return original_errors()

    monkeypatch.setattr(error, "errors", errors)
    projected = str(_invalid_request_error("integer.compute.gcd", error))
    diagnostic = json.loads(projected)

    assert materializations == 1
    assert len(diagnostic["errors"]) == 64
    assert diagnostic["omitted_error_count"] == 17
    assert len(diagnostic["errors"]) + diagnostic["omitted_error_count"] == (
        cause.error_count()
    )
    assert {tuple(issue["location"]) for issue in diagnostic["errors"]} == {("value",)}
    assert "private-value" not in projected


@pytest.mark.parametrize(
    "model", (OperationInvalidRequestData, OperationResourceAdmissionData)
)
@pytest.mark.parametrize("omitted", (None, 0, 1, 17))
def test_diagnostic_count_models_and_schemas_round_trip(
    model: type[OperationInvalidRequestData] | type[OperationResourceAdmissionData],
    omitted: int | None,
) -> None:
    issue = OperationValidationIssue(
        location=("value",), code="int_type", message="bad"
    )
    data = model(
        operation_id="integer.compute.gcd",
        errors=(issue,) * 64,
        omitted_error_count=omitted,
    )
    assert model.model_validate_json(data.model_dump_json(), strict=True) == data
    assert model.model_validate(data.model_dump(), strict=True) == data
    Draft202012Validator(model.model_json_schema()).validate(
        data.model_dump(mode="json")
    )

    legacy = data.model_dump(exclude={"omitted_error_count"})
    assert model.model_validate(legacy, strict=True).omitted_error_count is None
    legacy_json = data.model_dump_json(exclude={"omitted_error_count"})
    assert (
        model.model_validate_json(legacy_json, strict=True).omitted_error_count is None
    )
    assert (
        model(operation_id="integer.compute.gcd", errors=(issue,)).omitted_error_count
        is None
    )


@pytest.mark.parametrize(
    "model", (OperationInvalidRequestData, OperationResourceAdmissionData)
)
@pytest.mark.parametrize("omitted", (-1, True, "1", 1.5))
def test_diagnostic_models_reject_invalid_omitted_counts(
    model: type[OperationInvalidRequestData] | type[OperationResourceAdmissionData],
    omitted: Any,
) -> None:
    payload = {
        "operation_id": "integer.compute.gcd",
        "errors": [{"location": ["value"], "code": "int_type", "message": "bad"}],
        "omitted_error_count": omitted,
    }
    with pytest.raises(ValidationError) as rejected:
        model.model_validate_json(json.dumps(payload))
    assert rejected.value.errors()[0]["loc"] == ("omitted_error_count",)
    assert not Draft202012Validator(model.model_json_schema()).is_valid(payload)
