"""Complete exact results survive bounded SDK and stdio observations."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolResult,
    ReadResourceResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.mcp.delivery import (
    DeliveredFindResult,
    DeliveredRunResult,
    ResultDelivery,
)
from jacobian.mcp.server import create_server
from mcp import Client, StdioServerParameters, stdio_client

PROFILE = "number_theory.integer_interval.greatest_prime_factor_profile.compute"
PAYLOAD = {"lower_bound": 1000, "upper_bound": 10000}


def _gpf(number: int) -> int:
    greatest = 1
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            greatest = divisor
            number //= divisor
        divisor += 1
    return max(greatest, number)


def _size(result: CallToolResult) -> int:
    return len(result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))


def _resource_text(resource: ReadResourceResult) -> str:
    assert len(resource.contents) == 1
    contents = resource.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert contents.mime_type == "application/json"
    return contents.text


async def _assert_gcd(client: Client) -> None:
    result = await client.call_tool(
        "math.run",
        {
            "operation_id": "integer.compute.gcd",
            "payload": {"left": "12", "right": "18"},
        },
    )
    assert not result.is_error
    assert result.structured_content["output"] == {"value": "6"}
    assert [block.type for block in result.content] == ["text"]


def test_large_sdk_result_is_retained_completely_with_bounded_observation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        delivery = ResultDelivery(8192)
        async with Client(
            create_server(result_delivery=delivery), raise_exceptions=True
        ) as client:
            await _assert_gcd(client)
            assert delivery.retained_count == 0
            result = await client.call_tool(
                "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
            )
            assert not result.is_error
            assert _size(result) <= 8192
            metadata = result.structured_content
            assert metadata["kind"] == "retained_result"
            assert "9001 entries" in metadata["preview"]
            links = [
                block for block in result.content if isinstance(block, ResourceLink)
            ]
            assert len(links) == 1
            assert str(links[0].uri) == metadata["uri"]
            resource = await client.read_resource(metadata["uri"])
            encoded = _resource_text(resource).encode("utf-8")
            assert len(encoded) == metadata["byte_count"] == links[0].size
            # A host retains bytes directly; its model-facing call stayed bounded.
            artifact = tmp_path / "complete-result.json"
            artifact.write_bytes(encoded)
            full = json.loads(artifact.read_bytes())
            assert full["operation_id"] == PROFILE
            rows = full["output"]["rows"]
            assert rows == [
                {"n": n, "greatest_prime_factor": _gpf(n)} for n in range(1000, 10001)
            ]
            assert delivery.retained_count == 1
            await _assert_gcd(client)
            assert delivery.retained_count == 1
        assert delivery.retained_count == delivery.retained_bytes == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation_id",
    [
        "topology.simplicial.fundamental_group.map.compose.compute",
        "modular_form.character.equal.check",
    ],
)
def test_large_inspection_preserves_the_full_typed_contract(operation_id: str) -> None:
    async def scenario() -> None:
        async with Client(
            create_server(result_delivery=ResultDelivery(8192)), raise_exceptions=True
        ) as client:
            response = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            assert not response.is_error
            assert _size(response) <= 8192
            metadata = response.structured_content
            assert metadata["kind"] == "retained_result"
            resource = await client.read_resource(metadata["uri"])
            full = json.loads(_resource_text(resource))
            assert full["kind"] == "operation"
            expected = Catalog.open().inspect(operation_id)
            assert expected is not None
            assert full["operation"] == expected.model_dump(mode="json")

    asyncio.run(scenario())


@pytest.mark.parametrize(("count", "byte_limit"), [(1, 64 * 1024 * 1024), (16, 100)])
def test_capacity_failure_keeps_existing_references_and_small_requests_usable(
    count: int, byte_limit: int
) -> None:
    async def scenario() -> None:
        delivery = ResultDelivery(
            4096, max_results=count, max_retained_bytes=byte_limit
        )
        async with Client(
            create_server(result_delivery=delivery), raise_exceptions=False
        ) as client:
            first = await client.call_tool(
                "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
            )
            if byte_limit > 100:
                assert not first.is_error
                second = await client.call_tool(
                    "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
                )
                resource = await client.read_resource(first.structured_content["uri"])
                assert (
                    len(json.loads(_resource_text(resource))["output"]["rows"]) == 9001
                )
                assert delivery.retained_count == 1
            else:
                second = first
                assert delivery.retained_count == delivery.retained_bytes == 0
            assert second.is_error
            message = "\n".join(
                block.text for block in second.content if block.type == "text"
            )
            diagnostic, _ = json.JSONDecoder().raw_decode(message[message.index("{") :])
            assert diagnostic["code"] == "RESULT_DELIVERY_FAILED"
            assert diagnostic["stage"] == "result_delivery"
            assert _size(second) <= 4096
            await _assert_gcd(client)

    asyncio.run(scenario())


def test_expiry_and_shutdown_release_retained_bytes() -> None:
    async def scenario() -> None:
        delivery = ResultDelivery(4096, lifetime_seconds=0.2)
        async with Client(
            create_server(result_delivery=delivery), raise_exceptions=True
        ) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
            )
            uri = result.structured_content["uri"]
            await asyncio.sleep(0.25)
            with pytest.raises(MCPError):
                await client.read_resource(uri)
            assert delivery.retained_count == delivery.retained_bytes == 0
            await _assert_gcd(client)
            result = await client.call_tool(
                "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
            )
            assert result.structured_content["kind"] == "retained_result"
        assert delivery.retained_count == delivery.retained_bytes == 0

    asyncio.run(scenario())


def test_cancellation_after_computation_does_not_publish_a_reference() -> None:
    async def scenario() -> None:
        delivery = ResultDelivery(4096)
        with anyio.CancelScope() as cancellation:
            result = invoke_operation(PROFILE, PAYLOAD, Catalog.open())
            cancellation.cancel()
            await delivery.project(
                CallToolResult(
                    content=[], structured_content=result.model_dump(mode="json")
                )
            )
            raise AssertionError("cancelled publication returned a result")
        assert delivery.retained_count == delivery.retained_bytes == 0

    asyncio.run(scenario())


def test_one_sdk_session_cannot_clear_another_live_sessions_references() -> None:
    async def scenario() -> None:
        delivery = ResultDelivery(4096)
        server = create_server(result_delivery=delivery)
        async with Client(server, raise_exceptions=True) as first:
            async with Client(server, raise_exceptions=True) as second:
                result = await second.call_tool(
                    "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
                )
                uri = result.structured_content["uri"]
                assert delivery.retained_count == 1
            assert delivery.retained_count == 1
            resource = await first.read_resource(uri)
            assert len(json.loads(_resource_text(resource))["output"]["rows"]) == 9001
            await _assert_gcd(first)
        assert delivery.retained_count == delivery.retained_bytes == 0

    asyncio.run(scenario())


def test_stdio_budget_flag_supports_full_resource_retrieval() -> None:
    async def scenario() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "jacobian.mcp", "--inline-result-bytes", "8192"],
            cwd=Path.cwd(),
            env=dict(os.environ),
        )
        async with Client(stdio_client(parameters), raise_exceptions=True) as client:
            await _assert_gcd(client)
            result = await client.call_tool(
                "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
            )
            assert _size(result) <= 8192
            resource = await client.read_resource(result.structured_content["uri"])
            assert len(json.loads(_resource_text(resource))["output"]["rows"]) == 9001

    asyncio.run(scenario())


def test_cancellation_after_publication_preserves_bounded_retention_until_expiry() -> (
    None
):
    async def scenario() -> None:
        delivery = ResultDelivery(4096, lifetime_seconds=0.2)
        native = invoke_operation(PROFILE, PAYLOAD, Catalog.open())
        with anyio.CancelScope() as cancellation:
            reference = await delivery.project(
                CallToolResult(
                    content=[], structured_content=native.model_dump(mode="json")
                )
            )
            uri = reference.structured_content["uri"]
            cancellation.cancel()
            await anyio.lowlevel.checkpoint()
            raise AssertionError("cancelled transport continued")
        assert delivery.retained_count == 1
        assert 0 < delivery.retained_bytes <= delivery.max_retained_bytes
        assert (
            json.loads(delivery.read(uri.removeprefix("jacobian-result://")))["output"]
            == native.model_dump(mode="json")["output"]
        )
        await anyio.sleep(0.25)
        assert delivery.retained_count == delivery.retained_bytes == 0

    asyncio.run(scenario())


def test_delivery_advertises_typed_forms_and_preserves_sdk_inputs() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=True) as client:
            original = {tool.name: tool for tool in (await client.list_tools()).tools}
        delivery = ResultDelivery(8192)
        async with Client(
            create_server(result_delivery=delivery), raise_exceptions=True
        ) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools.keys() == original.keys() == {"math.find", "math.run"}
            for name in tools:
                assert tools[name].input_schema == original[name].input_schema
            assert (
                tools["math.find"].output_schema
                == DeliveredFindResult.model_json_schema()
            )
            assert (
                tools["math.run"].output_schema
                == DeliveredRunResult.model_json_schema()
            )
            result = await client.call_tool(
                "math.find", {"query": "greatest common divisor", "limit": 1}
            )
            assert not result.is_error
            assert result.structured_content["kind"] == "matches"
            assert delivery.retained_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("budget", ["4095", "1048577"])
def test_invalid_stdio_budget_is_refused_without_starting_a_server(budget: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jacobian.mcp", "--inline-result-bytes", budget],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "inline_result_bytes must lie in" in result.stderr


@pytest.mark.parametrize("sdk_arguments", [False, True])
def test_large_errors_keep_the_sdk_failure_and_fit_the_budget(
    *, sdk_arguments: bool
) -> None:
    async def scenario() -> None:
        extra = {"x" * 120 + str(index): 1 for index in range(64)}
        request: dict[str, Any] = {
            "operation_id": "integer.compute.gcd",
            "payload": {"left": "12", "right": "18"},
        }
        tool_name = "math.run"
        if sdk_arguments:
            tool_name = "argument-probe"
            request = {"values": [None] * 64}
        else:
            request["payload"].update(extra)

        async def argument_probe(values: list[int]) -> dict[str, int]:
            raise AssertionError("invalid SDK arguments reached the handler")

        original_server = create_server()
        if sdk_arguments:
            original_server.add_tool(argument_probe, name=tool_name)
        async with Client(original_server, raise_exceptions=False) as client:
            original = await client.call_tool(tool_name, request)
            assert original.is_error
            assert _size(original) > 4096
        delivery = ResultDelivery(4096)
        bounded_server = create_server(result_delivery=delivery)
        if sdk_arguments:
            bounded_server.add_tool(argument_probe, name=tool_name)
        async with Client(bounded_server, raise_exceptions=False) as client:
            bounded = await client.call_tool(tool_name, request)
            assert bounded.is_error
            assert _size(bounded) <= 4096
            metadata = bounded.structured_content
            assert metadata["kind"] == "retained_error"
            assert metadata["representation"] == "tool_result"
            resource = await client.read_resource(metadata["uri"])
            full = json.loads(_resource_text(resource))
            assert full == original.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
            await _assert_gcd(client)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
def test_negotiated_wire_shape_and_complete_error_envelope_are_preserved(
    mode: Literal["legacy", "2026-07-28"],
) -> None:
    async def scenario() -> None:
        server = create_server(result_delivery=ResultDelivery(4096))
        original: list[dict[str, Any]] = []
        observed: list[dict[str, Any]] = []

        async def capture_original(
            ctx: ServerRequestContext[Any, Any], call_next: CallNext
        ) -> HandlerResult:
            result = await call_next(ctx)
            if ctx.method == "tools/call":
                assert isinstance(result, dict)
                original.append(result)
            return result

        async def capture_observed(
            ctx: ServerRequestContext[Any, Any], call_next: CallNext
        ) -> HandlerResult:
            result = await call_next(ctx)
            if ctx.method == "tools/call":
                assert isinstance(result, dict)
                observed.append(result)
            return result

        server.middleware.insert(1, capture_original)
        server.middleware.insert(0, capture_observed)
        async with Client(server, mode=mode, raise_exceptions=False) as client:
            error = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.gcd",
                    "payload": {"x" * 120 + str(i): 1 for i in range(64)},
                },
            )
            assert error.is_error
            expected_modern = mode != "legacy"
            assert ("resultType" in original[-1]) == expected_modern
            assert ("resultType" in observed[-1]) == expected_modern
            assert (
                len(
                    json.dumps(
                        observed[-1], ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                )
                <= 4096
            )
            resource = await client.read_resource(error.structured_content["uri"])
            assert json.loads(_resource_text(resource)) == original[-1]
            await _assert_gcd(client)
            small_error = await client.call_tool(
                "math.run", {"operation_id": "not-installed", "payload": {}}
            )
            assert small_error.is_error
            assert observed[-1] == original[-1]

    asyncio.run(scenario())


def test_utf8_and_escaped_json_at_the_exact_inline_boundary() -> None:
    async def scenario() -> None:
        vertices = [str(index) + '雪"\\\n' * 10 for index in range(12)]
        async with Client(create_server(), raise_exceptions=True) as client:
            original = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.cut.maximum.compute",
                    "payload": {"graph": {"vertices": vertices, "edges": []}},
                },
            )
        assert original.structured_content["output"]["cut_value"] == 0
        text = json.dumps(
            original.structured_content, ensure_ascii=False, separators=(",", ":")
        )
        compact = original.model_copy(
            update={"content": [TextContent(type="text", text=text)]}
        )
        boundary = _size(compact)
        assert 4096 < boundary < 1048576
        admitted = ResultDelivery(boundary)
        inline = await admitted.project(original)
        assert _size(inline) == boundary
        assert inline.structured_content == original.structured_content
        assert admitted.retained_count == 0
        retained = ResultDelivery(boundary - 1)
        reference = await retained.project(original)
        assert _size(reference) <= boundary - 1
        metadata = reference.structured_content
        full_text = retained.read(metadata["uri"].removeprefix("jacobian-result://"))
        assert full_text.encode("utf-8") == text.encode("utf-8")
        assert metadata["byte_count"] == len(full_text.encode("utf-8"))
        retained.clear()

    asyncio.run(scenario())


@pytest.mark.parametrize("byte_quota", [False, True])
def test_concurrent_publications_enforce_aggregate_quotas(*, byte_quota: bool) -> None:
    async def scenario() -> None:
        native = invoke_operation(PROFILE, PAYLOAD, Catalog.open())
        size = len(
            json.dumps(
                native.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        delivery = ResultDelivery(
            4096,
            max_results=16 if byte_quota else 2,
            max_retained_bytes=size + 1000 if byte_quota else 64 * 1024 * 1024,
        )
        async with Client(
            create_server(result_delivery=delivery), raise_exceptions=False
        ) as client:
            results = await asyncio.gather(
                *(
                    client.call_tool(
                        "math.run", {"operation_id": PROFILE, "payload": PAYLOAD}
                    )
                    for _ in range(4)
                )
            )
            expected = 1 if byte_quota else 2
            assert sum(not result.is_error for result in results) == expected
            assert delivery.retained_count == expected
            assert delivery.retained_bytes <= delivery.max_retained_bytes
            for result in results:
                assert _size(result) <= 4096
                if not result.is_error:
                    resource = await client.read_resource(
                        result.structured_content["uri"]
                    )
                    assert (
                        len(json.loads(_resource_text(resource))["output"]["rows"])
                        == 9001
                    )
            await _assert_gcd(client)
        assert delivery.retained_count == delivery.retained_bytes == 0

    asyncio.run(scenario())


def test_local_delivery_refuses_an_http_handshake() -> None:
    async def scenario() -> None:
        import httpx2

        delivery = ResultDelivery(4096)
        app = create_server(result_delivery=delivery).streamable_http_app(
            json_response=True, stateless_http=True
        )
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app), base_url="http://127.0.0.1:8000"
            ) as client,
        ):
            response = await client.post(
                "/mcp",
                headers={"accept": "application/json, text/event-stream"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2026-07-28",
                        "capabilities": {},
                        "clientInfo": {"name": "delivery-test", "version": "1"},
                    },
                },
            )
            assert (
                response.json()["error"]["message"]
                == "temporary result delivery is local-only"
            )
        assert delivery.retained_count == 0

    asyncio.run(scenario())
