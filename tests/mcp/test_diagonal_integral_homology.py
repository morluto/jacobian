"""Discovery, bounded diagonal homology, and non-conclusions through live MCP."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.math.topology.chain_complexes.values import (
    HomologyResult,
    IntegralHomologyGroupValue,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_diagonal_homology_discovery_execution_and_resource_refusal() -> None:
    async def scenario() -> None:
        operation_id = "chain_complex.homology.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            described = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            assert not described.is_error
            assert described.structured_content is not None
            assert (
                described.structured_content["operation"]["operation_id"]
                == operation_id
            )
            for p in (15, 10**31):
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": operation_id,
                        "payload": {
                            "complex": {
                                "coefficient_ring": "ZZ",
                                "degree_min": 0,
                                "degree_max": 1,
                                "basis_sizes": [2, 2],
                                "differential_matrices": [
                                    [[str(p), "0"], ["0", str(p + 1)]]
                                ],
                            }
                        },
                    },
                )
                assert not result.is_error
                assert result.structured_content is not None
                decoded = HomologyResult.model_validate_json(
                    json.dumps(result.structured_content["output"])
                )
                group = decoded.homology_groups[0]
                assert isinstance(group, IntegralHomologyGroupValue)
                assert group.free_rank == 0
                assert group.torsion_invariant_factors == (p * (p + 1),)
            coefficient = 10**32 - 1
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation_id,
                    "payload": {
                        "complex": {
                            "coefficient_ring": "ZZ",
                            "degree_min": 0,
                            "degree_max": 1,
                            "basis_sizes": [2, 2],
                            "differential_matrices": [
                                [
                                    [str(coefficient), str(coefficient - 1)],
                                    [str(coefficient - 2), str(coefficient - 3)],
                                ]
                            ],
                        }
                    },
                },
            )
            assert rejected.is_error
            content = rejected.content[0]
            assert isinstance(content, TextContent)
            failure = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert failure["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert failure["stage"] == "resource_admission"

    asyncio.run(scenario())
