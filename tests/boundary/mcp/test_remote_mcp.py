from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import subprocess
import sys
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
    StaticTokenGrant,
    StaticTokenVerifier,
    TenantRuntimeLimitError,
    TenantRuntimeRouter,
    TenantRuntimeRouterClosedError,
    load_static_token_file,
)
from jacobian.adapters.mcp.server import create_server
from jacobian.runtime import CheckerAuthorityMode
from jacobian.storage.errors import ArtifactNotFoundError


def test_static_tokens_bind_distinct_authenticated_subjects() -> None:
    verifier = StaticTokenVerifier(
        (
            StaticTokenGrant(tenant_id="alpha", token="a" * 32),
            StaticTokenGrant(tenant_id="beta", token="b" * 32),
        )
    )

    alpha = asyncio.run(verifier.verify_token("a" * 32))
    beta = asyncio.run(verifier.verify_token("b" * 32))
    unknown = asyncio.run(verifier.verify_token("c" * 32))
    empty = asyncio.run(verifier.verify_token(""))
    oversized = asyncio.run(verifier.verify_token("c" * 4096))

    assert alpha is not None and alpha.subject == "alpha"
    assert beta is not None and beta.subject == "beta"
    assert unknown is None
    assert empty is None
    assert oversized is None


def test_remote_configuration_errors_name_the_rule_and_recovery(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"tenant_id must start with a letter or digit",
    ):
        StaticTokenGrant(tenant_id="bad subject", token="a" * 32)

    router = TenantRuntimeRouter(tmp_path, checker_authority=CheckerAuthorityMode.NONE)
    with pytest.raises(
        PermissionError,
        match="Authenticate with a configured bearer token and retry",
    ):
        router.runtime_for(None)
    with pytest.raises(
        PermissionError,
        match="Check the server token configuration",
    ):
        router.runtime_for("bad subject")

    missing = tmp_path / "missing-tokens.json"
    with pytest.raises(
        ValueError,
        match="Check that the file exists, is readable, and contains valid JSON",
    ):
        load_static_token_file(missing)

    token_file = tmp_path / "invalid-tokens.json"
    token_file.write_bytes(b"\xff\xfe")
    with pytest.raises(
        ValueError,
        match="Check that the file exists, is readable, and contains valid JSON",
    ):
        load_static_token_file(token_file)

    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {
                        "tenant_id": "alpha",
                        "token": "a" * 32,
                        "unexpected": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="unsupported field 'unexpected' in token grant 1",
    ):
        load_static_token_file(token_file)

    invalid_records = [
        ("not-an-object", "token grant 1 must be a JSON object"),
        (
            {"tenant_id": 1, "token": "a" * 32},
            "tenant_id in token grant 1 must be a string",
        ),
        (
            {"tenant_id": "alpha", "token": 1},
            "token in token grant 1 must be a string",
        ),
        (
            {"tenant_id": "alpha", "token": "a" * 32, "scopes": [1]},
            "scopes in token grant 1 must be an array of non-empty strings",
        ),
    ]
    for record, expected in invalid_records:
        token_file.write_text(
            json.dumps({"tokens": [record]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=expected):
            load_static_token_file(token_file)

    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {"tenant_id": "alpha", "token": "a" * 32},
                    {"tenant_id": "beta", "token": "short"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="token grant 2: remote bearer tokens must contain at least 32 characters",
    ):
        load_static_token_file(token_file)


def test_tenant_router_isolates_artifact_stores(tmp_path: Path) -> None:
    router = TenantRuntimeRouter(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        max_tenant_runtimes=2,
    )
    alpha = router.runtime_for("alpha")
    beta = router.runtime_for("beta")
    stored = alpha.core.store.register_descriptor(
        kind="semantics",
        name="alpha-only",
        version="1",
        definition={"value": 1},
    )

    assert alpha.core.store.root != beta.core.store.root
    assert router.runtime_for("alpha") is alpha
    with pytest.raises(ArtifactNotFoundError):
        beta.core.store.get(stored)


class _FakeRuntime:
    def __init__(self, identity: Path, *, fail_close: bool = False) -> None:
        self.identity = identity
        self.fail_close = fail_close
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("injected close failure")


def test_tenant_router_single_flights_one_tenant_and_parallelizes_distinct_tenants(
    tmp_path: Path,
) -> None:
    creation_barrier = threading.Barrier(2)
    created: list[_FakeRuntime] = []
    created_lock = threading.Lock()

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        creation_barrier.wait(timeout=2)
        runtime = _FakeRuntime(path)
        with created_lock:
            created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=3,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    leases = []

    def acquire(subject: str) -> None:
        leases.append(router.lease_for(subject))

    workers = [
        threading.Thread(target=acquire, args=(subject,))
        for subject in ("alpha", "alpha", "beta")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()

    alpha_runtimes = [
        lease.runtime
        for lease in leases
        if lease.runtime.identity.name == hashlib.sha256(b"alpha").hexdigest()
    ]
    assert len(created) == 2
    assert len(alpha_runtimes) == 2
    assert alpha_runtimes[0] is alpha_runtimes[1]
    for lease in leases:
        lease.release()


def test_tenant_router_uses_idle_ttl_lru_and_never_evicts_an_active_lease(
    tmp_path: Path,
) -> None:
    now = [0.0]
    created: list[_FakeRuntime] = []

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        runtime = _FakeRuntime(path)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=2,
        idle_timeout_seconds=10,
        clock=lambda: now[0],
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    alpha = router.lease_for("alpha")
    now[0] = 1
    beta_runtime = router.runtime_for("beta")
    now[0] = 2
    alpha.release()
    now[0] = 3
    alpha_runtime = router.runtime_for("alpha")
    now[0] = 4
    gamma_runtime = router.runtime_for("gamma")

    assert beta_runtime.close_calls == 1
    assert alpha_runtime.close_calls == 0
    assert gamma_runtime.close_calls == 0

    active = router.lease_for("alpha")
    now[0] = 20
    refreshed_gamma = router.runtime_for("gamma")
    assert refreshed_gamma is not gamma_runtime
    assert gamma_runtime.close_calls == 1
    active.release()


def test_tenant_router_restores_failed_eviction_and_shutdown_waits_for_leases(
    tmp_path: Path,
) -> None:
    first = _FakeRuntime(tmp_path / "first", fail_close=True)
    created = [first]

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        if len(created) == 1:
            return first
        runtime = _FakeRuntime(path)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=1,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    assert router.runtime_for("alpha") is first
    with pytest.raises(RuntimeError, match="injected close failure"):
        router.runtime_for("beta")
    assert router.runtime_for("alpha") is first

    lease = router.lease_for("alpha")
    with pytest.raises(TenantRuntimeLimitError, match="tenant limit"):
        router.lease_for("beta")
    first.fail_close = False
    close_entered = threading.Event()
    original_close = router.close

    def close_after_signaling() -> None:
        close_entered.set()
        original_close()

    router.close = close_after_signaling  # type: ignore[method-assign]
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (router.close(), closed.set()))
    closer.start()
    assert close_entered.wait(timeout=2)
    assert not closed.is_set()
    with pytest.raises(TenantRuntimeRouterClosedError, match="closing"):
        router.lease_for("beta")
    lease.release()
    closer.join(timeout=2)
    assert closed.is_set()


def test_tenant_router_retries_only_failed_runtime_shutdowns(tmp_path: Path) -> None:
    created: list[_FakeRuntime] = []

    def factory(path: Path, **_kwargs: object) -> _FakeRuntime:
        runtime = _FakeRuntime(path, fail_close=not created)
        created.append(runtime)
        return runtime

    router = TenantRuntimeRouter(
        tmp_path,
        max_tenant_runtimes=2,
        runtime_factory=factory,  # type: ignore[arg-type]
    )
    router.runtime_for("alpha")
    router.runtime_for("beta")

    with pytest.raises(ExceptionGroup, match="tenant runtimes failed to close"):
        router.close()
    assert [runtime.close_calls for runtime in created] == [1, 1]

    created[0].fail_close = False
    router.close()
    router.close()

    assert [runtime.close_calls for runtime in created] == [2, 1]


def test_anonymous_tenant_namespace_is_fixed_by_the_operator(tmp_path: Path) -> None:
    first = TenantRuntimeRouter(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        allow_anonymous=True,
        anonymous_tenant_id="test-endpoint-a",
    )
    second = TenantRuntimeRouter(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
        allow_anonymous=True,
        anonymous_tenant_id="test-endpoint-b",
    )

    first_runtime = first.runtime_for(None)
    second_runtime = second.runtime_for(None)

    assert first_runtime.core.store.root != second_runtime.core.store.root
    assert first.runtime_for(None) is first_runtime
    with pytest.raises(ValueError, match="anonymous_tenant_id must start"):
        TenantRuntimeRouter(
            tmp_path,
            checker_authority=CheckerAuthorityMode.NONE,
            allow_anonymous=True,
            anonymous_tenant_id="caller controlled",
        )


def test_token_file_is_strict_and_remote_cli_fails_closed(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {
                        "tenant_id": "alpha",
                        "token": "a" * 32,
                        "scopes": ["jacobian:use"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_static_token_file(token_file)[0].tenant_id == "alpha"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.adapters.mcp.server",
            "--transport",
            "streamable-http",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "require --auth-tokens-file" in completed.stderr

    named_without_anonymous = subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.adapters.mcp.server",
            "--transport",
            "streamable-http",
            "--anonymous-tenant-id",
            "test-endpoint-a",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert named_without_anonymous.returncode != 0
    assert "--anonymous-tenant-id requires --allow-anonymous" in (
        named_without_anonymous.stderr
    )

    conflicting_auth = subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.adapters.mcp.server",
            "--transport",
            "streamable-http",
            "--allow-anonymous",
            "--auth-tokens-file",
            str(token_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert conflicting_auth.returncode != 0
    assert "mutually exclusive" in conflicting_auth.stderr

    invalid_idle_timeout = subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.adapters.mcp.server",
            "--transport",
            "streamable-http",
            "--allow-anonymous",
            "--tenant-idle-timeout-seconds",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid_idle_timeout.returncode != 0
    assert "--tenant-idle-timeout-seconds must be positive" in (
        invalid_idle_timeout.stderr
    )


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
    from mcp.shared.exceptions import MCPError

    url = f"http://127.0.0.1:{port}/mcp"

    async def invoke(token: str, *, create: bool) -> tuple[int, str | None]:
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
            created_artifact_uri = None
            if create:
                await client.call_tool(
                    "capability.invoke",
                    {
                        "capability_id": "fixture.increment",
                        "mode": "EXPLORE",
                        "payload": {"value": 4},
                    },
                )
                created = await client.call_tool(
                    "capability.invoke",
                    {
                        "capability_id": "integer.compute.gcd",
                        "mode": "EXPLORE",
                        "payload": {"left": "84", "right": "30"},
                    },
                )
                created_artifact_uri = next(
                    block.uri
                    for block in created.content
                    if block.type == "resource_link"
                )
            searched = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "knowledge.search",
                    "mode": "EXPLORE",
                    "payload": {"query": "fixture.increment"},
                },
            )
            payload = json.loads(searched.content[0].text)
            assert payload["execution"]["status"] == "COMPLETED", payload["execution"]
            assert "hits" in payload["output"], payload
            return len(payload["output"]["hits"]), created_artifact_uri

    alpha_hits, alpha_artifact_uri = await invoke("a" * 32, create=True)
    beta_hits, _ = await invoke("b" * 32, create=False)
    assert alpha_hits == 1
    assert beta_hits == 0
    assert alpha_artifact_uri is not None

    async with (
        httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {'b' * 32}"},
            trust_env=False,
            timeout=60,
        ) as http,
        Client(
            streamable_http_client(url, http_client=http),
            raise_exceptions=True,
        ) as beta_client,
    ):
        with pytest.raises(MCPError):
            await beta_client.read_resource(alpha_artifact_uri)


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
