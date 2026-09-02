"""MCP catalog projection and operator-policy boundary tests."""

from __future__ import annotations

import asyncio
import json

from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    TextContent,
    TextResourceContents,
)

from jacobian.mcp.server import create_server


def _resource_text(
    content: TextResourceContents | BlobResourceContents,
) -> str:
    assert isinstance(content, TextResourceContents)
    return content.text


def _content_text(content: ContentBlock) -> str:
    assert isinstance(content, TextContent)
    return content.text


def test_mcp_catalog_is_the_complete_static_operation_library() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(),
            raise_exceptions=True,
        ) as client:
            resource = await client.read_resource("operation://catalog")
            catalog = json.loads(_resource_text(resource.contents[0]))
            assert catalog["operations"]
            assert "policy_profile" not in catalog
            assert "policy_digest" not in catalog

    asyncio.run(scenario())


def test_mcp_compact_operation_matches_are_paginated() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(),
            raise_exceptions=True,
        ) as client:
            resource_result = await client.read_resource("operation://catalog")
            full_catalog = json.loads(_resource_text(resource_result.contents[0]))
            discoverable_ids = {
                descriptor["operation_id"] for descriptor in full_catalog["operations"]
            }

            listed = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "match",
                        "need": "exact mathematical computation",
                    }
                },
            )
            assert isinstance(listed.structured_content, dict)
            index = listed.structured_content
            listed_text = _content_text(listed.content[0])
            assert len(listed_text.encode("utf-8")) <= 16 * 1024
            assert json.loads(listed_text) == index
            assert len(index["matches"]) <= 10
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
                            "op": "match",
                            "need": "exact mathematical computation",
                            "cursor": cursor,
                            "limit": 10,
                        }
                    },
                )
                assert isinstance(next_page.structured_content, dict)
                page = next_page.structured_content
                assert (
                    len(_content_text(next_page.content[0]).encode("utf-8"))
                    <= 16 * 1024
                )
                indexed_ids.update(
                    descriptor["operation_id"] for descriptor in page["matches"]
                )
                cursor = page["next_cursor"]
            assert len(indexed_ids) == index["total_matches"]
            assert indexed_ids <= discoverable_ids

            first_page = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "match",
                        "need": "exact mathematical computation",
                        "limit": 10,
                    }
                },
            )
            assert isinstance(first_page.structured_content, dict)
            first = first_page.structured_content
            assert len(first["matches"]) <= 10
            assert first["next_cursor"] is not None
            second_page = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "match",
                        "need": "exact mathematical computation",
                        "cursor": first["next_cursor"],
                        "limit": 10,
                    }
                },
            )
            assert isinstance(second_page.structured_content, dict)
            second = second_page.structured_content
            assert {
                descriptor["operation_id"] for descriptor in first["matches"]
            }.isdisjoint(descriptor["operation_id"] for descriptor in second["matches"])

            wider_page = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "match",
                        "need": "exact mathematical computation",
                        "limit": 20,
                    }
                },
            )
            assert isinstance(wider_page.structured_content, dict)
            assert len(wider_page.structured_content["matches"]) == 20

            invalid_cursor = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "match",
                        # Keep the cursor invalid for this filtered result. The
                        # catalog contains many descriptions mentioning "operation",
                        # so that word is not a stable no-match fixture as the
                        # library grows.
                        "need": "zqx",
                        "cursor": "integer.compute.unknown",
                        "limit": 10,
                    }
                },
            )
            invalid = json.loads(_content_text(invalid_cursor.content[0]))
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())
