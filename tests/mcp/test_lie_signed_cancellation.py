"""Newly admitted exact Lie cancellations and refusals on the real MCP surface."""

import asyncio
import json
from typing import Any

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.differential import (
    RationalLieDerivativeProfile,
    verify_lie_derivative,
)
from jacobian.mcp.server import create_server
from mcp import Client

_OPERATION = "differential_geometry.rational_tensor.lie_derivative.compute"


def _tensor(
    exponent: int,
    coefficient: int = 1,
    *,
    reciprocal: bool = False,
    extra: bool = False,
    axes: tuple[str, ...] = ("x",),
) -> dict[str, Any]:
    def polynomial(rows: list[tuple[int, tuple[int, ...]]]) -> dict[str, Any]:
        return {
            "terms": [
                {"coefficient": {"num": str(c), "den": "1"}, "exponents": list(e)}
                for c, e in rows
            ]
        }

    components = []
    guards = []
    for index in range(len(axes)):
        powers = tuple(exponent if axis == index else 0 for axis in range(len(axes)))
        numerator_rows = (
            [(coefficient, (0,) * len(axes))] if reciprocal else [(coefficient, powers)]
        )
        if extra:
            numerator_rows.append(
                (
                    1,
                    tuple(
                        power - int(axis == index) for axis, power in enumerate(powers)
                    ),
                )
            )
        denominator = polynomial([(1, powers if reciprocal else (0,) * len(axes))])
        components.append(
            {
                "variables": list(axes),
                "numerator": polynomial(numerator_rows),
                "denominator": denominator,
            }
        )
        if reciprocal:
            guards.append(denominator)
    return {
        "coordinate_axis": list(axes),
        "variance": ["CONTRAVARIANT"],
        "components": components,
        "retained_nonzero_denominators": guards,
    }


def test_live_cancellation_discovery_dispatch_and_unchanged_handoff() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            found = await client.call_tool("math.find", {"operation_id": _OPERATION})
            assert not found.is_error
            for vector, tensor in (
                (_tensor(33), _tensor(33)),
                (_tensor(128), _tensor(128)),
                (_tensor(33, 7), _tensor(33, 21)),
                (_tensor(64, 7, reciprocal=True), _tensor(64, 21, reciprocal=True)),
                (_tensor(33, axes=("y", "x")), _tensor(33, axes=("y", "x"))),
                (_tensor(33), _tensor(33, extra=True)),
            ):
                payload = {"vector_field": vector, "tensor": tensor}
                dispatched = invoke_operation(_OPERATION, payload, Catalog.open())
                result = await client.call_tool(
                    "math.run", {"operation_id": _OPERATION, "payload": payload}
                )
                assert not result.is_error
                assert result.structured_content is not None
                output = result.structured_content["output"]
                assert output == dispatched.output
                decoded = RationalLieDerivativeProfile.model_validate_json(
                    json.dumps(output)
                )
                assert verify_lie_derivative(decoded)
                derived = output["lie_derivative"]
                assert derived["coordinate_axis"] == tensor["coordinate_axis"]
                assert (
                    derived["retained_nonzero_denominators"]
                    == tensor["retained_nonzero_denominators"]
                )
                if len(tensor["components"][0]["numerator"]["terms"]) == 2:
                    assert derived["components"][0]["numerator"]["terms"] == [
                        {"coefficient": {"num": "-1", "den": "1"}, "exponents": [64]}
                    ]
                else:
                    assert all(
                        not component["numerator"]["terms"]
                        for component in derived["components"]
                    )
                    handed = await client.call_tool(
                        "math.run",
                        {
                            "operation_id": _OPERATION,
                            "payload": {"vector_field": vector, "tensor": derived},
                        },
                    )
                    assert not handed.is_error
                    assert handed.structured_content is not None
                    assert (
                        handed.structured_content["output"]["lie_derivative"] == derived
                    )

    asyncio.run(scenario())


def test_live_signed_presolve_preserves_genuine_growth_and_source_rejections() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            unreduced = _tensor(34)
            denominator = {
                "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [1]}]
            }
            unreduced["components"][0]["denominator"] = denominator
            unreduced["retained_nonzero_denominators"] = [denominator]
            for vector, tensor, code in (
                (_tensor(128), _tensor(127), "RESOURCE_ADMISSION_REJECTED"),
                (unreduced, unreduced, "INVALID_REQUEST"),
            ):
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": _OPERATION,
                        "payload": {"vector_field": vector, "tensor": tensor},
                    },
                )
                assert result.is_error
                assert isinstance(result.content[0], TextContent)
                text = result.content[0].text
                diagnostic = json.loads(text[text.index("{") :])
                assert diagnostic["code"] == code
                if code == "INVALID_REQUEST":
                    assert diagnostic["errors"][0]["code"].endswith(
                        "component_not_canonical"
                    )

    asyncio.run(scenario())
