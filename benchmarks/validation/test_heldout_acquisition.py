from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.tooling import heldout_integrity
from benchmarks.tooling.command_runner import operator_environment
from benchmarks.tooling.heldout_integrity import _AWS_ENVIRONMENT_VARS
from benchmarks.validation.heldout_fixtures import _manifest


def test_aws_environment_vars_include_only_credentials_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("UNRELATED_SECRET", "should-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = operator_environment(include=_AWS_ENVIRONMENT_VARS)

    assert env["AWS_ACCESS_KEY_ID"] == "AKIATEST"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert env["AWS_SESSION_TOKEN"] == "token"
    assert env["AWS_REGION"] == "us-east-1"
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert "UNRELATED_SECRET" not in env


def test_aws_environment_vars_exclude_non_aws_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("DATABASE_URL", "postgres://should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")

    env = operator_environment(include=_AWS_ENVIRONMENT_VARS)

    assert env["AWS_ACCESS_KEY_ID"] == "AKIATEST"
    assert "DATABASE_URL" not in env
    assert "OPENAI_API_KEY" not in env


def test_fetch_bundle_passes_aws_environment_to_s3_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LEAKED_VAR", "should-not-appear")

    captured_envs: list[object] = []
    manifest = _manifest()

    def fake_run_command(
        command: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout_seconds: float = 600.0,
        environment: object | None = None,
    ) -> object:
        captured_envs.append(environment)
        dest = Path(arguments[-1])
        if dest.suffix == ".json":
            dest.write_text("{}", encoding="utf-8")
        else:
            dest.write_text("fake", encoding="utf-8")
        return type(
            "Result",
            (),
            {"exit_code": 0, "diagnostic": None, "stderr": b""},
        )()

    monkeypatch.setattr(heldout_integrity, "run_operator_command", fake_run_command)
    monkeypatch.setattr(heldout_integrity, "validate_manifest", lambda _p: manifest)
    monkeypatch.setattr(heldout_integrity, "verify_bundle", lambda _m, _r: None)
    monkeypatch.setattr(heldout_integrity, "_safe_extract", lambda _a, _o: None)

    def fake_digest(path: Path) -> str:
        name = Path(path).name
        if name == "snapshot-lock.json":
            return manifest["snapshot_lock"]["lock_digest"]
        if name == "bundle.tar.gz":
            return manifest["archive"]["sha256"]
        return "sha256:" + "0" * 64

    monkeypatch.setattr(heldout_integrity, "_digest", fake_digest)

    heldout_integrity.fetch_bundle("s3://bucket/manifest.json", tmp_path / "out")

    assert len(captured_envs) == 3
    for env in captured_envs:
        env_dict = dict(env) if env is not None else {}
        assert env_dict.get("AWS_ACCESS_KEY_ID") == "AKIATEST"
        assert env_dict.get("AWS_SECRET_ACCESS_KEY") == "secret"
        assert env_dict.get("AWS_REGION") == "us-east-1"
