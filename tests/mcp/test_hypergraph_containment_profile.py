import asyncio

from tests.mcp.test_direct_operations import _server

from mcp import Client


def test_hypergraph_containment_mcp_roundtrip() -> None:
    async def scenario() -> None:
        operation="incidence.containment_profiles.compute"
        async with Client(_server(operation), raise_exceptions=True) as client:
            payload={"incidence":{"vertices":["a","b","c"],"edges":[["e1",["a","b"]],["e2",["a","b"]],["e3",[]]]},"t":2}
            response=await client.call_tool(operation,payload)
            assert response.structured_content is not None
            assert response.structured_content["total_multiplicity"] == 2
    asyncio.run(scenario())
