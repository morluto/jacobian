"""Compact locus outputs remain typed and compose through real MCP execution."""

import asyncio
import json

import pytest
from mcp.types import TextContent
from sympy import symbols

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.pullback._models import (
    RationalMetricPullbackProfile,
    RationalMetricPullbackRequest,
)
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize(
    "kind", ("identity", "constant", "singular", "undefined", "growth")
)
def test_locus_boundary_public_results_and_tensor_handoff(kind: str) -> None:
    u, x, y = symbols("u x y")
    source = ("x", "y") if kind == "identity" else ("x",)
    target = ("u", "v") if kind == "identity" else ("u",)
    entries = (
        (u**33, 0, 0, u**33)
        if kind == "identity"
        else (
            1 / u
            if kind == "undefined"
            else u**64
            if kind == "constant"
            else u**2
            if kind == "singular"
            else 10**127 * u,
        )
    )
    components = tuple(rational_function_from_sympy(v, target) for v in entries)
    guards = tuple(
        c.denominator
        for c in components
        if any(any(t.exponents) for t in c.denominator.terms)
    )
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=target,
            variance=("COVARIANT", "COVARIANT"),
            components=components,
            retained_nonzero_denominators=guards,
        )
    )
    images = (
        (x, y)
        if kind == "identity"
        else (
            10**127
            if kind == "constant"
            else 0
            if kind in {"singular", "undefined"}
            else 10**127 * x,
        )
    )
    mapping = RationalFunctionMap(
        source_variables=source,
        target_coordinates=target,
        components=tuple(rational_function_from_sympy(v, source) for v in images),
    )
    payload = RationalMetricPullbackRequest(metric=metric, map=mapping).model_dump(
        mode="json"
    )
    operation = "differential_geometry.rational_metric.pullback.compute"
    if kind in {"identity", "constant"}:
        output = invoke_operation(operation, payload, Catalog.open()).output
        decoded = RationalMetricPullbackProfile.model_validate_json(json.dumps(output))
        expected = (x**33, 0, 0, x**33) if kind == "identity" else (0,)
        assert (
            tuple(rational_function_to_sympy(c) for c in decoded.pullback.components)
            == expected
        )
        assert len(decoded.pullback_locus_guard) == (1 if kind == "identity" else 0)
    else:
        with pytest.raises(
            OperationResourceAdmissionError
            if kind == "growth"
            else OperationDomainValidationError
        ):
            invoke_operation(operation, payload, Catalog.open())
        output = None

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": payload}
            )
            if output is None:
                assert result.is_error
                content = result.content[0]
                assert isinstance(content, TextContent)
                diagnostic = json.loads(
                    content.text.removeprefix("Error executing tool math.run: ")
                )
                assert diagnostic["code"] == (
                    "RESOURCE_ADMISSION_REJECTED"
                    if kind == "growth"
                    else "INVALID_REQUEST"
                )
                if kind != "growth":
                    assert (
                        diagnostic["errors"][0]["code"]
                        == f"differential_geometry.rational_metric.pullback.{kind}_metric{'_locus' if kind == 'undefined' else ''}"
                    )
                return
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == output
            vector = RationalCoordinateTensor(
                coordinate_axis=source,
                variance=("CONTRAVARIANT",),
                components=tuple(
                    rational_function_from_sympy(0, source) for _ in source
                ),
            )
            consumed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "differential_geometry.rational_tensor.lie_derivative.compute",
                    "payload": {
                        "vector_field": vector.model_dump(mode="json"),
                        "tensor": output["pullback"],
                    },
                },
            )
            assert not consumed.is_error
            assert consumed.structured_content is not None
            tensor = RationalCoordinateTensor.model_validate_json(
                json.dumps(consumed.structured_content["output"]["lie_derivative"])
            )
            assert all(not c.numerator.terms for c in tensor.components)
            assert (
                tensor.retained_nonzero_denominators
                == decoded.pullback.retained_nonzero_denominators
            )

    asyncio.run(scenario())
