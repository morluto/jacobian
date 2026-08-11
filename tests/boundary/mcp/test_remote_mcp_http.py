"""Authenticated remote MCP HTTP isolation and deployment smoke tests."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from importlib.metadata import version
from pathlib import Path

import pytest
from deploy.smoke_remote import inspect as inspect_remote_deployment
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from uvicorn import Config, Server

from jacobian.adapters.mcp.remote import (
    StaticTokenVerifier,
    load_static_token_file,
)
from jacobian.adapters.mcp.server import create_server


def test_authenticated_streamable_http_isolates_tenant_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {"tenant_id": "alpha", "token": "a" * 32},
                    {"tenant_id": "beta", "token": "b" * 32},
                    {
                        "tenant_id": "wrong-scope",
                        "token": "s" * 32,
                        "scopes": ["jacobian:other"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        public_base_url = f"http://127.0.0.1:{port}"
        mcp_server = create_server(
            tmp_path / "state",
            tenant_isolation=True,
            allow_anonymous=False,
            token_verifier=StaticTokenVerifier(load_static_token_file(token_file)),
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(public_base_url),
                resource_server_url=AnyHttpUrl(f"{public_base_url}/mcp"),
                required_scopes=["jacobian:use"],
            ),
            capability_adapter_entrypoints=(
                "tests.component.capabilities._fixture_capabilities:create_adapter",
            ),
        )
        http_server = Server(
            Config(
                mcp_server.streamable_http_app(),
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        )
        server_thread = threading.Thread(
            target=http_server.run,
            kwargs={"sockets": [listener]},
            daemon=True,
        )
        server_thread.start()
        try:
            _wait_for_server(http_server, server_thread)
            asyncio.run(_remote_auth_rejections(port))
            asyncio.run(_remote_tenant_scenario(port))
            monkeypatch.setenv("JACOBIAN_MCP_BEARER_TOKEN", "a" * 32)
            report = asyncio.run(
                inspect_remote_deployment(
                    url=f"http://127.0.0.1:{port}/mcp",
                    expected_version=version("jacobian"),
                    expected_policy_profile="DEFAULT",
                    required_capabilities={"fixture.increment"},
                    query="fixture increment",
                    timeout_seconds=60,
                )
            )
            assert report["server"]["version"] == version("jacobian")
            assert report["catalog"]["policy_profile"] == "DEFAULT"
            assert report["catalog"]["catalog_digest"].startswith("sha256:")
        finally:
            http_server.should_exit = True
            server_thread.join(timeout=10)
            assert not server_thread.is_alive()


async def _remote_auth_rejections(port: int) -> None:
    import httpx2

    url = f"http://127.0.0.1:{port}/mcp"
    async with httpx2.AsyncClient(trust_env=False) as client:
        unauthenticated = await client.post(url, json={})
        wrong_scope = await client.post(
            url,
            json={},
            headers={"Authorization": f"Bearer {'s' * 32}"},
        )

    assert unauthenticated.status_code == 401
    assert wrong_scope.status_code == 403


async def _remote_tenant_scenario(port: int) -> None:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    url = f"http://127.0.0.1:{port}/mcp"

    async def invoke(token: str) -> dict[str, object]:
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                trust_env=False,
                # The first request constructs the tenant's complete capability
                # runtime; keep transport tolerance separate from backend budgets.
                timeout=60,
            ) as http,
            Client(
                streamable_http_client(url, http_client=http),
                raise_exceptions=True,
            ) as client,
        ):
            catalog = await client.read_resource("capability://catalog")
            assert "fixture.increment" in catalog.contents[0].text
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "fixture.increment",
                    "payload": {"value": 4},
                },
            )
            assert isinstance(result.structured_content, dict)
            return result.structured_content["output"]

    alpha_output = await invoke("a" * 32)
    beta_output = await invoke("b" * 32)
    assert alpha_output == {"value": 5}
    assert beta_output == {"value": 5}


def _wait_for_server(
    http_server: Server,
    server_thread: threading.Thread,
) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            raise AssertionError("remote MCP server exited early")
        if http_server.started:
            return
        time.sleep(0.1)
    raise AssertionError("remote MCP server did not start")
