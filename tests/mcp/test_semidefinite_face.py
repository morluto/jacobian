"""Live SDK result validation and recovery for a supplied exposing relation."""

import asyncio
import json
from copy import deepcopy

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.semidefinite import RationalSemidefiniteSystem
from jacobian.math.matrices.semidefinite._models import SemidefiniteFaceReductionRequest
from jacobian.math.matrices.semidefinite._tools import TOOLS, compute_face_reduction
from jacobian.math.matrices.values import RationalMatrix
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
            large = SemidefiniteFaceReductionRequest(
                system=RationalSemidefiniteSystem(
                    order=1,
                    matrices=(
                        RationalMatrix(
                            entries=((CanonicalRational(num=1, den=10**20000 + 1),),)
                        ),
                    ),
                    rhs=(CanonicalRational(num=0, den=1),),
                ),
                multipliers=(CanonicalRational(num=1, den=1),),
            )
            large_result = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": large.model_dump(mode="json"),
                },
            )
            assert not large_result.is_error
            assert large_result.structured_content is not None
            assert large_result.structured_content["output"] == compute_face_reduction(
                large
            ).model_dump(mode="json")

    asyncio.run(scenario())
