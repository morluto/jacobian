import asyncio

from tests.mcp.test_direct_operations import _server

from mcp import Client


def test_hypergraph_edge_count_moments_mcp_roundtrip() -> None:
    async def scenario() -> None:
        async with Client(
            _server("probability.hypergraph_edge_count_moments.compute"),
            raise_exceptions=True,
        ) as client:
            payload = {
                "hypergraph": {
                    "vertices": ["a", "b"],
                    "edges": [["e1", ["a"]], ["e2", ["b"]]],
                },
                "retention_probability": {"num": "1", "den": "2"},
            }
            result = await client.call_tool(
                "probability.hypergraph_edge_count_moments.compute", payload
            )
            assert result.structured_content is not None
            assert result.structured_content["edge_count_expectation"] == {
                "num": "1",
                "den": "1",
            }
            assert result.structured_content["overlap_profile"][0]["pair_count"] == "1"

    asyncio.run(scenario())
