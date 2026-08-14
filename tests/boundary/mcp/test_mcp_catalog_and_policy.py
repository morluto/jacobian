"""MCP catalog projection and operator-policy boundary tests."""

from __future__ import annotations

import asyncio
import json

from jacobian.adapters.mcp.server import create_server
from jacobian.operation_visibility import OperationVisibilityPolicy


def test_mcp_no_retrieval_policy_is_operator_bound_and_fail_closed() -> None:
    async def scenario() -> None:
        from mcp import Client

        policy = OperationVisibilityPolicy(profile="COMPUTE_VERIFY_NO_RETRIEVAL")
        async with Client(
            create_server(operation_policy=policy),
            raise_exceptions=True,
        ) as client:
            resource = await client.read_resource("operation://catalog")
            catalog = json.loads(resource.contents[0].text)
            assert catalog["policy_profile"] == "COMPUTE_VERIFY_NO_RETRIEVAL"
            assert catalog["policy_digest"] == policy.digest
            assert all(
                "retrieval" not in descriptor["tags"]
                for descriptor in catalog["operations"]
            )

    asyncio.run(scenario())


def test_mcp_compact_operation_index_is_searchable_and_paginated() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(),
            raise_exceptions=True,
        ) as client:
            resource_result = await client.read_resource("operation://catalog")
            full_catalog = json.loads(resource_result.contents[0].text)
            discoverable_ids = {
                descriptor["operation_id"] for descriptor in full_catalog["operations"]
            }

            listed = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "exact", "limit": 20}},
            )
            assert isinstance(listed.structured_content, dict)
            index = listed.structured_content
            assert len(listed.content[0].text.encode("utf-8")) <= 16 * 1024
            assert json.loads(listed.content[0].text) == index
            assert index["response_byte_limit"] == 16 * 1024
            assert len(index["matches"]) <= 20
            indexed_ids = {
                descriptor["operation_id"] for descriptor in index["matches"]
            }
            assert all(
                "input_schema" not in descriptor for descriptor in index["matches"]
            )
            cursor = index["next_cursor"]
            while cursor is not None:
                next_page = await client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "search",
                            "query": "exact",
                            "cursor": cursor,
                            "limit": 20,
                        }
                    },
                )
                assert isinstance(next_page.structured_content, dict)
                page = next_page.structured_content
                assert len(next_page.content[0].text.encode("utf-8")) <= 16 * 1024
                indexed_ids.update(
                    descriptor["operation_id"] for descriptor in page["matches"]
                )
                cursor = page["next_cursor"]
            assert len(indexed_ids) == index["total_matches"]
            assert indexed_ids <= discoverable_ids

            first_page = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "exact", "limit": 20}},
            )
            assert isinstance(first_page.structured_content, dict)
            first = first_page.structured_content
            assert len(first["matches"]) <= 20
            assert first["next_cursor"] is not None
            second_page = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "exact",
                        "cursor": first["next_cursor"],
                        "limit": 20,
                    }
                },
            )
            assert isinstance(second_page.structured_content, dict)
            second = second_page.structured_content
            assert {
                descriptor["operation_id"] for descriptor in first["matches"]
            }.isdisjoint(descriptor["operation_id"] for descriptor in second["matches"])

            invalid_cursor = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "definitely-no-matching-operation",
                        "cursor": first["next_cursor"],
                        "limit": 20,
                    }
                },
            )
            invalid = json.loads(invalid_cursor.content[0].text)
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())
