from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server
from jacobian.registry import CheckerRegistry


def test_graph_resource_operations_do_not_assemble_the_portfolio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_portfolio_assembly(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "selected graph operations must not assemble the portfolio"
        )

    monkeypatch.setattr(CheckerRegistry, "authorize", reject_portfolio_assembly)

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            constructed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.construct.explicit",
                    "payload": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                },
            )
            assert constructed.structured_content is not None
            assert constructed.structured_content["execution"]["status"] == "COMPLETED"
            graph_uri = constructed.structured_content["output"]["graph_uri"]

            composed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.construct.compose",
                    "payload": {
                        "operation": "COMPLEMENT",
                        "left_graph_uri": graph_uri,
                    },
                },
            )
            assert composed.structured_content is not None
            assert composed.structured_content["execution"]["status"] == "COMPLETED"

            isomorphic = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.isomorphism.verify",
                    "payload": {
                        "left_graph_uri": graph_uri,
                        "right_graph_uri": graph_uri,
                        "mapping": {"a": "a", "b": "b", "c": "c"},
                    },
                },
            )
            assert isomorphic.structured_content is not None
            assert isomorphic.structured_content["output"]["conclusion"] == "TRUE"

            properties = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.compute.properties",
                    "payload": {
                        "graph_uri": graph_uri,
                        "properties": ["order", "size", "tree"],
                    },
                },
            )
            assert properties.structured_content is not None
            assert properties.structured_content["execution"]["status"] == "COMPLETED"
            assert properties.structured_content["output"]["properties"] == {
                "order": {
                    "value": 3,
                    "exactness": "EXACT",
                    "backend": "networkx",
                },
                "size": {
                    "value": 2,
                    "exactness": "EXACT",
                    "backend": "networkx",
                },
                "tree": {
                    "value": True,
                    "exactness": "EXACT",
                    "backend": "networkx",
                },
            }

            realized = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.realize.degree_sequence",
                    "payload": {"degree_sequence": [2, 2, 2]},
                },
            )
            assert realized.structured_content is not None
            certificate_uri = realized.structured_content["output"]["certificate_uri"]
            verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.degree_sequence.verify",
                    "payload": {"certificate_uri": certificate_uri},
                },
            )
            assert verified.structured_content is not None
            assert verified.structured_content["output"]["conclusion"] == "TRUE"
            assert verified.structured_content["verification_record_uri"] is not None

            neighborhood = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.compute.neighborhood_independence",
                    "payload": {"graph_uri": graph_uri},
                },
            )
            assert neighborhood.structured_content is not None
            neighborhood_certificate = neighborhood.structured_content["output"][
                "certificate_uri"
            ]
            neighborhood_verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.neighborhood_independence.verify",
                    "payload": {"certificate_uri": neighborhood_certificate},
                },
            )
            assert neighborhood_verified.structured_content is not None
            assert (
                neighborhood_verified.structured_content["output"]["conclusion"]
                == "TRUE"
            )

    asyncio.run(scenario())
