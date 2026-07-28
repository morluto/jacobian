from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jacobian.adapters.mcp.remote import (
    StaticTokenGrant,
    StaticTokenVerifier,
    TenantKernelLimitError,
    TenantKernelRouter,
    load_static_token_file,
)
from jacobian.contracts.workspaces import (
    WorkspaceOpenRequest,
    WorkspaceQueryRequest,
    WorkspaceQueryView,
)
from jacobian.store import ArtifactNotFoundError
from jacobian.workspaces import WorkspaceNotFoundError


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

    assert alpha is not None and alpha.subject == "alpha"
    assert beta is not None and beta.subject == "beta"
    assert unknown is None


def test_remote_configuration_errors_name_the_rule_and_recovery(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"tenant_id must start with a letter or digit",
    ):
        StaticTokenGrant(tenant_id="bad subject", token="a" * 32)

    router = TenantKernelRouter(tmp_path, install_references=False)
    with pytest.raises(
        PermissionError,
        match="Authenticate with a configured bearer token and retry",
    ):
        router.kernel_for(None)
    with pytest.raises(
        PermissionError,
        match="Check the server token configuration",
    ):
        router.kernel_for("bad subject")

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
    router = TenantKernelRouter(
        tmp_path,
        install_references=False,
        max_tenant_kernels=2,
    )
    alpha = router.kernel_for("alpha")
    beta = router.kernel_for("beta")
    stored = alpha.store.register_descriptor(
        kind="semantics",
        name="alpha-only",
        version="1",
        definition={"value": 1},
    )

    assert alpha.store.root != beta.store.root
    assert router.kernel_for("alpha") is alpha
    with pytest.raises(TenantKernelLimitError, match="tenant limit"):
        router.kernel_for("gamma")
    with pytest.raises(ArtifactNotFoundError):
        beta.store.get(stored)


def test_anonymous_tenant_namespace_is_fixed_by_the_operator(tmp_path: Path) -> None:
    first = TenantKernelRouter(
        tmp_path,
        install_references=False,
        allow_anonymous=True,
        anonymous_tenant_id="test-endpoint-a",
    )
    second = TenantKernelRouter(
        tmp_path,
        install_references=False,
        allow_anonymous=True,
        anonymous_tenant_id="test-endpoint-b",
    )

    first_kernel = first.kernel_for(None)
    second_kernel = second.kernel_for(None)

    assert first_kernel.store.root != second_kernel.store.root
    assert first.kernel_for(None) is first_kernel
    with pytest.raises(ValueError, match="anonymous_tenant_id must start"):
        TenantKernelRouter(
            tmp_path,
            install_references=False,
            allow_anonymous=True,
            anonymous_tenant_id="caller controlled",
        )


def test_tenant_router_isolates_epistemic_workspaces(tmp_path: Path) -> None:
    router = TenantKernelRouter(tmp_path, install_references=False)
    alpha = router.kernel_for("alpha")
    beta = router.kernel_for("beta")
    opened = alpha.workspaces.open(
        WorkspaceOpenRequest(
            idempotency_key="tenant-workspace-open-001",
            name="alpha workspace",
            problem="This working state belongs only to alpha.",
        )
    )

    with pytest.raises(WorkspaceNotFoundError):
        beta.workspaces.query(
            WorkspaceQueryRequest(
                workspace_id=opened.workspace_id,
                branch_id=opened.branch_id,
                view=WorkspaceQueryView.RESUME,
            )
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


@pytest.mark.subprocess
def test_authenticated_streamable_http_isolates_tenant_memory(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {"tenant_id": "alpha", "token": "a" * 32},
                    {"tenant_id": "beta", "token": "b" * 32},
                ]
            }
        ),
        encoding="utf-8",
    )
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = int(reserved.getsockname()[1])
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jacobian.adapters.mcp.server",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--state-dir",
            str(tmp_path / "state"),
            "--auth-tokens-file",
            str(token_file),
            "--public-base-url",
            f"http://127.0.0.1:{port}",
            "--capability-adapter",
            "tests.fixtures.capability_functions:create_adapter",
        ],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port, process)
        asyncio.run(_remote_tenant_scenario(port))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


async def _remote_tenant_scenario(port: int) -> None:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    url = f"http://127.0.0.1:{port}/mcp"

    async def invoke(token: str, *, create: bool) -> int:
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                trust_env=False,
                # The first request constructs the tenant's complete capability
                # kernel; keep transport tolerance separate from backend budgets.
                timeout=60,
            ) as http,
            Client(
                streamable_http_client(url, http_client=http),
                raise_exceptions=True,
            ) as client,
        ):
            catalog = await client.read_resource("capability://catalog")
            assert "fixture.increment" in catalog.contents[0].text
            if create:
                await client.call_tool(
                    "capability.invoke",
                    {
                        "capability_id": "fixture.increment",
                        "mode": "EXPLORE",
                        "payload": {"value": 4},
                    },
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
            return len(payload["output"]["hits"])

    assert await invoke("a" * 32, create=True) == 1
    assert await invoke("b" * 32, create=False) == 0


def _wait_for_port(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"remote MCP server exited early: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("remote MCP server did not start")
