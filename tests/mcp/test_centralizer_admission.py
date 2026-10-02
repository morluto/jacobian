"""Live MCP centralizer admission and resource-error recovery."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.math.matrices.canonical_forms._models import CentralizerResult
from jacobian.mcp.server import create_server
from mcp import Client


def test_centralizer_structured_regimes_and_resource_refusal() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            for size, regime in (
                (23, "scalar"),
                (17, "jordan"),
                (3, "jordan"),
                (16, "jordan"),
                (17, "scalar"),
                (32, "diagonal"),
            ):
                source = {
                    "entries": [
                        [
                            {
                                "num": str(
                                    10**255
                                    if regime == "jordan" and j == i + 1
                                    else i
                                    if regime == "diagonal" and i == j
                                    else 7
                                    if regime == "scalar" and i == j
                                    else 0
                                ),
                                "den": "1",
                            }
                            for j in range(size)
                        ]
                        for i in range(size)
                    ]
                }
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "matrix.centralizer.compute",
                        "payload": {"matrix": source},
                    },
                )
                if (size, regime) in ((23, "scalar"), (17, "jordan")):
                    assert result.is_error
                    assert isinstance(result.content[0], TextContent)
                    text = result.content[0].text
                    error = json.loads(text[text.index("{") :])
                    assert error["code"] == "RESOURCE_ADMISSION_REJECTED"
                    assert error["errors"][0]["code"] == (
                        "matrix.centralizer.output"
                        if regime == "scalar"
                        else "matrix.centralizer.work"
                    )
                    continue
                assert not result.is_error
                assert result.structured_content is not None
                output = result.structured_content["output"]
                decoded = CentralizerResult.model_validate_json(json.dumps(output))
                assert decoded.dimension == (size**2 if regime == "scalar" else size)
                assert len(decoded.basis) == decoded.dimension

    asyncio.run(scenario())
