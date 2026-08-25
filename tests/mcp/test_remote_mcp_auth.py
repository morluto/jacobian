"""Remote MCP token, scope, and CLI configuration boundary tests."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

import jacobian.mcp.remote as remote_module
import jacobian.mcp.remote_cli as remote_cli
from jacobian.mcp.remote import (
    StaticTokenGrant,
    StaticTokenVerifier,
    _resolve_tenant,
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

    with pytest.raises(
        PermissionError,
        match="Authenticate with a configured bearer token and retry",
    ):
        _resolve_tenant(None, allow_anonymous=False, anonymous_tenant_id="anonymous")
    with pytest.raises(
        PermissionError,
        match="Check the server token configuration",
    ):
        _resolve_tenant(
            "bad subject", allow_anonymous=False, anonymous_tenant_id="anonymous"
        )

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
            "jacobian.mcp.remote_cli",
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
            "jacobian.mcp.remote_cli",
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
            "jacobian.mcp.remote_cli",
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


@pytest.mark.parametrize(
    ("session_arguments", "expected_stateless", "expected_mode"),
    [
        ([], True, "stateless"),
        (["--stateless-http"], True, "stateless"),
        (["--stateful-http"], False, "stateful"),
    ],
)
def test_remote_streamable_http_session_mode_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    session_arguments: list[str],
    expected_stateless: bool,
    expected_mode: str,
) -> None:
    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeServer:
        def run(self, *args: object, **kwargs: object) -> None:
            run_calls.append((args, kwargs))

    monkeypatch.setattr(remote_module, "create_remote_server", lambda **_: FakeServer())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jacobian-remote-mcp",
            "--allow-anonymous",
            *session_arguments,
        ],
    )

    remote_cli.main()

    assert run_calls == [
        (
            ("streamable-http",),
            {
                "host": "127.0.0.1",
                "port": 8000,
                "streamable_http_path": "/mcp",
                "stateless_http": expected_stateless,
            },
        )
    ]
    assert f"session_mode={expected_mode}" in capsys.readouterr().err


def test_remote_session_mode_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        remote_cli._parser().parse_args(["--stateless-http", "--stateful-http"])
