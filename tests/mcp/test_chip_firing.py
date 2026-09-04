"""Live MCP projection of chip-firing defining regressions and rejection."""

import asyncio

from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from jacobian.mcp.server import create_server
from mcp import Client


def test_chip_firing_regressions_through_live_mcp() -> None:
    async def scenario() -> None:
        graph = {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
        }
        cases = [
            (
                "stabilize",
                {
                    "configuration": {
                        "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
                        "sink": "a",
                        "configuration": [0, 3],
                    }
                },
                "stable",
                [3, 0],
            ),
            (
                "q_reduced",
                {"graph": graph, "sink": "a", "divisor": [0, 1, 1]},
                "reduced_divisor",
                [2, 0, 0],
            ),
            (
                "q_reduced",
                {"graph": graph, "sink": "a", "divisor": [0, -1, 0]},
                "reduced_divisor",
                [-2, 0, 1],
            ),
            (
                "abel_jacobi",
                {"graph": graph, "sink": "a", "divisor": [-1, 2, -1]},
                "coordinates",
                [0],
            ),
        ]
        async with Client(create_server(), raise_exceptions=True) as client:
            for name, payload, field, expected in cases:
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": f"graph.chip_firing.{name}.compute",
                        "payload": payload,
                    },
                )
                assert not result.is_error
                assert result.structured_content["output"][field] == expected
            # The formerly rejected coupled HNF is a successful public
            # operation, not merely a fast private-backend diagnostic.
            coupled = {
                "vertices": [str(i) for i in range(7)],
                "edges": [
                    [str(i), str(j)]
                    for i, j in (
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (1, 2),
                        (1, 4),
                        (1, 5),
                        (1, 6),
                        (2, 3),
                        (2, 5),
                        (3, 4),
                        (3, 5),
                        (3, 6),
                        (4, 5),
                        (5, 6),
                    )
                ],
            }
            for divisor, principal in (
                ([-5, 1, 1, 1, 1, 1, 0], True),
                ([1, -1, 0, 0, 0, 0, 0], False),
            ):
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "graph.chip_firing.abel_jacobi.compute",
                        "payload": {"graph": coupled, "sink": "0", "divisor": divisor},
                    },
                )
                assert not result.is_error
                output = result.structured_content["output"]
                assert output["invariant_factors"] == [1, 1, 1, 1, 1, 2520]
                assert (output["coordinates"] == [0]) == principal
            disconnected = {"vertices": ["a", "b", "c"], "edges": [["b", "c"]]}
            # INVALID_PARAMS is a protocol exception, distinct from a
            # successful mathematical result or an execution tool error.
            for name in ("stabilize", "q_reduced"):
                payload = {"graph": disconnected, "sink": "a"}
                payload = (
                    {"configuration": {**payload, "configuration": [0, 1, 1]}}
                    if name == "stabilize"
                    else {**payload, "divisor": [0, 1, 1]}
                )
                try:
                    await client.call_tool(
                        "math.run",
                        {
                            "operation_id": f"graph.chip_firing.{name}.compute",
                            "payload": payload,
                        },
                    )
                except MCPError as exc:
                    assert exc.code == INVALID_PARAMS
                    assert (
                        exc.data["errors"][0]["code"]
                        == "chip_firing.requires_connected_graph"
                    )
                else:
                    raise AssertionError("MCP admitted a disconnected sink graph")

    asyncio.run(scenario())
