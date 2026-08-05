from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
    ToolResolver,
    git_head_sha,
    operator_environment,
    run_operator_command,
)


@pytest.mark.parametrize("stdout", [b"x" * 40, b"\xff" * 40])
def test_git_head_sha_rejects_non_digest_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: bytes
) -> None:
    monkeypatch.setattr(
        "benchmarks.tooling.command_runner.run_operator_command",
        lambda *_args, **_kwargs: ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=stdout,
            stderr=b"",
        ),
    )

    assert git_head_sha(tmp_path) is None


def test_tooling_request_rejects_bare_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ToolCommandRequest(
            executable="git",
            cwd=str(tmp_path),
            timeout_seconds=1.0,
            stdout_limit_bytes=1,
            stderr_limit_bytes=1,
        )


def test_operator_environment_does_not_forward_sensitive_host_values() -> None:
    environment = operator_environment(
        source={
            "LANG": "C",
            "PATH": "/poisoned",
            "PYTHONPATH": "/poisoned-python",
            "HTTPS_PROXY": "https://credential.invalid",
            "AWS_SECRET_ACCESS_KEY": "secret",
        }
    )
    assert environment == {"LANG": "C"}


def test_resolver_returns_absolute_executable_once() -> None:
    resolver = ToolResolver(search_path=str(Path(sys.executable).parent))
    resolved = resolver.resolve(Path(sys.executable).name)
    assert resolved is not None
    assert Path(resolved).is_absolute()
    assert resolver.resolve(Path(sys.executable).name) == resolved


def test_missing_operator_command_is_classified(tmp_path: Path) -> None:
    result = run_operator_command(
        "jacobian-command-that-does-not-exist",
        cwd=tmp_path,
        timeout_seconds=1.0,
    )
    assert result.status is ToolCommandStatus.START_FAILED
    assert result.exit_code is None
    assert result.diagnostic is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_operator_command_has_bounded_output(tmp_path: Path) -> None:
    result = run_operator_command(
        Path(sys.executable).name,
        ("-c", "print('x' * 10000)"),
        cwd=tmp_path,
        timeout_seconds=5.0,
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert result.status is ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
    assert len(result.stdout) <= 32
