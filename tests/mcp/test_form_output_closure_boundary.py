"""Wider exact form results compose publicly; actual overflow stays operational."""

import asyncio
import json
from typing import Any

import pytest
from mcp.types import TextContent

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.differential_forms import (
    PolynomialDifferentialForm,
    PrimitiveResult,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _polynomial(
    axis: list[str], powers: list[int], num: int, den: int = 1
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": axis,
        "polynomial": {
            "terms": [
                {
                    "coefficient": {
                        "num": format_canonical_integer(num),
                        "den": format_canonical_integer(den),
                    },
                    "exponents": powers,
                }
            ]
            if num
            else []
        },
    }


def _form(
    axis: list[str],
    degree: int,
    indices: list[int],
    powers: list[int],
    num: int,
    den: int = 1,
) -> dict[str, Any]:
    return {
        "variables": axis,
        "degree": str(degree),
        "components": [
            {"indices": indices, "coefficient": _polynomial(axis, powers, num, den)}
        ]
        if num
        else [],
    }


@pytest.mark.parametrize(
    "kind", ("contract", "pullback", "primitive", "lie_derivative")
)
@pytest.mark.parametrize("overflow", (False, True))
def test_producer_decoder_consumer_closure_and_typed_refusal(
    kind: str, overflow: bool
) -> None:
    digits = MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS if overflow else 4096
    operation = f"differential_form.{kind}.compute"
    if kind == "primitive":
        denominator = 6 * 10 ** (digits - 1)
        source = _form(["x"], 1, [0], [1], 1, denominator)
        payload = {"form": source}
        expected_form = _form(["x"], 0, [], [2], 1, 2 * denominator)
        expected_derivative = source
    elif kind == "lie_derivative":
        coefficient = 5 * 10 ** (digits - 1)
        source = _form(["x", "y"], 1, [1], [1, 0], coefficient)
        payload = {
            "form": source,
            "field": {
                "variables": ["x", "y"],
                "components": [
                    _polynomial(["x", "y"], [1, 0], 1),
                    _polynomial(["x", "y"], [0, 1], 1),
                ],
            },
        }
        expected_form = _form(["x", "y"], 1, [1], [1, 0], 2 * coefficient)
        expected_derivative = _form(["x", "y"], 2, [0, 1], [0, 0], 2 * coefficient)
    else:
        coefficient = 10 ** (digits // 2)
        source = _form(["x"], 1, [0], [0], coefficient)
        payload = {"form": source}
        if kind == "contract":
            payload["field"] = {
                "variables": ["x"],
                "components": [_polynomial(["x"], [0], coefficient)],
            }
            expected_form = _form(["x"], 0, [], [0], coefficient**2)
            expected_derivative = _form(["x"], 1, [0], [0], 0)
        else:
            payload["mapping"] = {
                "source_variables": ["x"],
                "target_variables": ["x"],
                "images": [_polynomial(["x"], [1], coefficient)],
            }
            expected_form = _form(["x"], 1, [0], [0], coefficient**2)
            expected_derivative = _form(["x"], 2, [0, 1], [0], 0)
    catalog = Catalog.open()
    if overflow:
        with pytest.raises(OperationResourceAdmissionError) as native:
            invoke_operation(operation, payload, catalog)
        assert (
            native.value.errors()[0]["type"]
            == "differential_form.calculus.coefficient_budget"
        )
    else:
        output = invoke_operation(operation, payload, catalog).output
        if kind == "primitive":
            decoded = PrimitiveResult.model_validate_json(json.dumps(output))
            assert decoded.source == PolynomialDifferentialForm.model_validate_json(
                json.dumps(source)
            )
            assert decoded.outcome == "CONSTRUCTED"
            actual_form = output["primitive"]
        else:
            actual_form = output
        assert actual_form == expected_form
        PolynomialDifferentialForm.model_validate_json(json.dumps(actual_form))
        assert (
            invoke_operation(
                "differential_form.exterior_derivative.compute",
                {"form": actual_form},
                catalog,
            ).output
            == expected_derivative
        )

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": payload}
            )
            if overflow:
                assert result.is_error
                content = result.content[0]
                assert isinstance(content, TextContent)
                diagnostic = json.loads(
                    content.text.removeprefix("Error executing tool math.run: ")
                )
                assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
                assert diagnostic["stage"] == "resource_admission"
                assert (
                    diagnostic["errors"][0]["code"]
                    == "differential_form.calculus.coefficient_budget"
                )
                return
            assert not result.is_error
            assert result.structured_content is not None
            output = result.structured_content["output"]
            form = output["primitive"] if kind == "primitive" else output
            assert form == expected_form
            consumer = await client.call_tool(
                "math.run",
                {
                    "operation_id": "differential_form.exterior_derivative.compute",
                    "payload": {"form": form},
                },
            )
            assert not consumer.is_error
            assert consumer.structured_content is not None
            assert consumer.structured_content["output"] == expected_derivative

    asyncio.run(scenario())
