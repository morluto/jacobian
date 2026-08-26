"""MCP catalog projection and operator-policy boundary tests."""

from __future__ import annotations

import asyncio
import json

from mcp.types import (
    CallToolResult,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
)

from jacobian.mcp.server import create_server


def _tool_text(result: CallToolResult) -> str:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _resource_text(result: ReadResourceResult) -> str:
    content = result.contents[0]
    assert isinstance(content, TextResourceContents)
    return content.text


def test_mcp_catalog_is_the_complete_static_operation_library() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(),
            raise_exceptions=True,
        ) as client:
            resource = await client.read_resource("operation://catalog")
            catalog = json.loads(_resource_text(resource))
            assert catalog["operations"]
            assert "policy_profile" not in catalog
            assert "policy_digest" not in catalog
            assert "catalog_version" not in catalog
            assert all(
                "descriptor_version" not in operation and "version" not in operation
                for operation in catalog["operations"]
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
            full_catalog = json.loads(_resource_text(resource_result))
            discoverable_ids = {
                descriptor["operation_id"] for descriptor in full_catalog["operations"]
            }

            listed = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "exact", "limit": 20}},
            )
            assert isinstance(listed.structured_content, dict)
            index = listed.structured_content
            listed_text = _tool_text(listed)
            assert len(listed_text.encode("utf-8")) <= 16 * 1024
            assert json.loads(listed_text) == index
            assert index["response_byte_limit"] == 16 * 1024
            assert "discovery_version" not in index
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
                assert len(_tool_text(next_page).encode("utf-8")) <= 16 * 1024
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
            invalid = json.loads(_tool_text(invalid_cursor))
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())


def test_mcp_operation_browse_pages_the_complete_immutable_library() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(),
            raise_exceptions=True,
        ) as client:
            resource_result = await client.read_resource("operation://catalog")
            full_catalog = json.loads(_resource_text(resource_result))
            catalog_ids = [
                descriptor["operation_id"] for descriptor in full_catalog["operations"]
            ]

            request: dict[str, object] = {"op": "browse", "limit": 20}
            browsed_ids: list[str] = []
            while True:
                page_result = await client.call_tool("math.find", {"request": request})
                assert isinstance(page_result.structured_content, dict)
                page = page_result.structured_content
                assert page["kind"] == "browse"
                page_text = _tool_text(page_result)
                assert len(page_text.encode("utf-8")) <= 16 * 1024
                assert json.loads(page_text) == page
                assert page["response_byte_limit"] == 16 * 1024
                assert "discovery_version" not in page
                assert len(page["operations"]) <= 20
                assert all(
                    "input_schema" not in operation
                    and "output_schema" not in operation
                    and "examples" not in operation
                    for operation in page["operations"]
                )
                page_ids = [
                    operation["operation_id"] for operation in page["operations"]
                ]
                assert page_ids == sorted(page_ids)
                browsed_ids.extend(page_ids)
                cursor = page["next_cursor"]
                if cursor is None:
                    break
                request["cursor"] = cursor

            assert browsed_ids == catalog_ids
            assert len(set(browsed_ids)) == len(catalog_ids)

            invalid_cursor = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "browse",
                        "cursor": "integer.compute.unknown",
                    }
                },
            )
            invalid = json.loads(_tool_text(invalid_cursor))
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())


def test_mcp_search_and_browse_bound_unrestricted_card_metadata() -> None:
    from jacobian._models import StrictModel
    from jacobian.catalog.catalog import Catalog
    from jacobian.catalog.models import MathTool
    from jacobian.mcp.runtime import AppState
    from jacobian.mcp.server import _build_server

    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

    tools = tuple(
        MathTool(
            operation_id=f"test.large_metadata.card{index}",
            title=f"Large metadata card {index}",
            description="exact " + ("数学" * 12_000),
            request_type=Request,
            result_type=Result,
            run=lambda request: Result(value=request.value),
            tags=("标签" * 12_000,),
        )
        for index in range(4)
    )
    server = _build_server(state=AppState(operation_catalog=Catalog(tools)))

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            searched = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "exact", "limit": 4}},
            )
            assert isinstance(searched.structured_content, dict)
            search_page = searched.structured_content
            assert len(_tool_text(searched).encode("utf-8")) <= 16 * 1024
            assert search_page["response_byte_limit"] == 16 * 1024
            assert search_page["truncation_reason"] == "BYTE_LIMIT"
            assert search_page["match_metadata_truncated"] is True

            long_query = await client.call_tool(
                "math.find",
                {"request": {"op": "search", "query": "q" * 20_000}},
            )
            assert isinstance(long_query.structured_content, dict)
            long_query_page = long_query.structured_content
            assert len(_tool_text(long_query).encode("utf-8")) <= 16 * 1024
            assert long_query_page["query_metadata_truncated"] is True
            assert long_query_page["truncation_reason"] == "BYTE_LIMIT"

            request: dict[str, object] = {"op": "browse", "limit": 4}
            browsed_ids: list[str] = []
            while True:
                browsed = await client.call_tool("math.find", {"request": request})
                assert isinstance(browsed.structured_content, dict)
                browse_page = browsed.structured_content
                assert len(_tool_text(browsed).encode("utf-8")) <= 16 * 1024
                assert browse_page["response_byte_limit"] == 16 * 1024
                assert browse_page["truncation_reason"] == "BYTE_LIMIT"
                assert browse_page["operation_metadata_truncated"] is True
                browsed_ids.extend(
                    operation["operation_id"] for operation in browse_page["operations"]
                )
                cursor = browse_page["next_cursor"]
                if cursor is None:
                    break
                request["cursor"] = cursor

        assert browsed_ids == sorted(tool.operation_id for tool in tools)

    asyncio.run(scenario())
