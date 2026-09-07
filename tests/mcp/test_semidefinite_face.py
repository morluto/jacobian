"""Live SDK result validation and recovery for a supplied exposing relation."""

import asyncio
import json
from copy import deepcopy

from jacobian.math.matrices.semidefinite._models import SemidefiniteFaceReductionRequest
from jacobian.math.matrices.semidefinite._tools import TOOLS, compute_face_reduction
from jacobian.mcp.server import create_server
from mcp import Client


def test_exposed_face_native_mcp_parity_and_invalid_relation_recovery() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        native = compute_face_reduction(
            SemidefiniteFaceReductionRequest.model_validate_json(json.dumps(payload))
        )
        async with Client(create_server(), raise_exceptions=False) as client:
            invalid = deepcopy(payload)
            invalid["multipliers"] = [{"num": "-1", "den": "1"}]
            error = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": invalid,
                },
            )
            assert error.is_error
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": payload,
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == native.model_dump(mode="json")

    asyncio.run(scenario())
