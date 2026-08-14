"""Remote MCP token, scope, and CLI configuration boundary tests."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jacobian.adapters.mcp.remote import (
    StaticTokenGrant,
    StaticTokenVerifier,
    TenantRuntimeRouter,
    load_static_token_file,
)


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

    router = TenantRuntimeRouter(
        tmp_path,
        runtime_factory=lambda *_args, **_kwargs: pytest.fail(
            "authentication must precede runtime construction"
        ),
    )
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
            "jacobian.adapters.mcp.remote_cli",
            "--transport",
            "streamable-http",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "requires --auth-tokens-file" in completed.stderr

    named_without_anonymous = subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.adapters.mcp.remote_cli",
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
            "jacobian.adapters.mcp.remote_cli",
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
            "jacobian.adapters.mcp.remote_cli",
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
