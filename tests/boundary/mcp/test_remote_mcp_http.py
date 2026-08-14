"""Authenticated remote MCP HTTP isolation and deployment smoke tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import pytest
from deploy.smoke_remote import inspect as inspect_remote_deployment
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from uvicorn import Config, Server

from jacobian.adapters.mcp.deployment_identity import DeploymentIdentity
from jacobian.adapters.mcp.remote import (
    StaticTokenVerifier,
    create_remote_server,
    load_static_token_file,
)
from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.runtime.model import JacobianRuntime
from tests.support.selected_runtime import create_selected_runtime


def test_authenticated_streamable_http_rejects_before_runtime_construction(
    tmp_path: Path,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> JacobianRuntime:
        raise AssertionError("auth rejection must not construct a tenant runtime")

    with _RunningRemoteServer(
        tmp_path,
        runtime_factory=fail_if_called,
    ) as port:
        asyncio.run(_remote_auth_rejections(port))


def test_authenticated_streamable_http_isolates_tenant_data(
    tmp_path: Path,
) -> None:
    def matrix_runtime(
        root: str | Path,
        **_kwargs: object,
    ) -> JacobianRuntime:
        return create_selected_runtime(
            root,
            (matrix_operations(),),
        )

    with _RunningRemoteServer(
        tmp_path,
        runtime_factory=matrix_runtime,
    ) as port:
        asyncio.run(_remote_tenant_scenario(port))
        tenant_roots = sorted((tmp_path / "state" / "tenants").iterdir())
        assert {path.name for path in tenant_roots} == {
            hashlib.sha256(b"alpha").hexdigest(),
            hashlib.sha256(b"beta").hexdigest(),
        }
        assert all((path / "metadata.sqlite3").is_file() for path in tenant_roots)


def test_default_authority_remote_mcp_matches_deployment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setattr(
        "jacobian.adapters.mcp.remote.load_deployment_identity",
        lambda: DeploymentIdentity(
            revision=revision,
            package_version=version("jacobian"),
        ),
    )
    with _RunningRemoteServer(tmp_path) as port:
        monkeypatch.setenv("JACOBIAN_MCP_BEARER_TOKEN", "a" * 32)
        report = asyncio.run(
            inspect_remote_deployment(
                url=f"http://127.0.0.1:{port}/mcp",
                expected_version=version("jacobian"),
                expected_revision=revision,
                expected_policy_profile="DEFAULT",
                required_operations={"matrix.determinant.compute"},
                query="exact matrix determinant",
                timeout_seconds=60,
            )
        )
        assert report["server"]["version"] == version("jacobian")
        assert report["deployment"]["revision"] == revision
        assert report["catalog"]["policy_profile"] == "DEFAULT"
        assert report["catalog"]["catalog_digest"].startswith("sha256:")


class _RunningRemoteServer:
    def __init__(
        self,
        tmp_path: Path,
        *,
        runtime_factory: Callable[..., JacobianRuntime] | None = None,
    ) -> None:
        self._tmp_path = tmp_path
        self._runtime_factory = runtime_factory
        self._http_server: Server | None = None
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self.port = 0

    def __enter__(self) -> int:
        token_file = self._tmp_path / "tokens.json"
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
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen()
        self.port = int(self._listener.getsockname()[1])
        public_base_url = f"http://127.0.0.1:{self.port}"
        mcp_server = create_remote_server(
            self._tmp_path / "state",
            allow_anonymous=False,
            token_verifier=StaticTokenVerifier(load_static_token_file(token_file)),
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(public_base_url),
                resource_server_url=AnyHttpUrl(f"{public_base_url}/mcp"),
                required_scopes=["jacobian:use"],
            ),
            runtime_factory=self._runtime_factory,
        )
        self._http_server = Server(
            Config(
                mcp_server.streamable_http_app(),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )
        self._thread = threading.Thread(
            target=self._http_server.run,
            kwargs={"sockets": [self._listener]},
            daemon=True,
        )
        self._thread.start()
        _wait_for_server(self._http_server, self._thread)
        return self.port

    def __exit__(self, *_exc: object) -> None:
        if self._http_server is not None:
            self._http_server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            assert not self._thread.is_alive()
        if self._listener is not None:
            self._listener.close()


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

    async def invoke(token: str, value: str) -> dict[str, object]:
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                trust_env=False,
                timeout=60,
            ) as http,
            Client(
                streamable_http_client(url, http_client=http),
                raise_exceptions=True,
            ) as client,
        ):
            catalog = await client.read_resource("operation://catalog")
            assert "matrix.determinant.compute" in catalog.contents[0].text
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.determinant.compute",
                    "payload": {
                        "matrix": {
                            "matrix_schema_version": "1",
                            "domain": "QQ",
                            "entries": [[{"num": value, "den": "1"}]],
                        }
                    },
                },
            )
            assert isinstance(result.structured_content, dict)
            return result.structured_content["output"]

    alpha_output = await invoke("a" * 32, "4")
    beta_output = await invoke("b" * 32, "9")
    assert alpha_output["result"]["determinant"] == {"num": "4", "den": "1"}
    assert beta_output["result"]["determinant"] == {"num": "9", "den": "1"}


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
