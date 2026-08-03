"""Contracts for source bootstrap identity and environment pins."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_identity_uses_uv_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doctor = _load_tool("source_agent_doctor")

    def fake_run(command: list[str], **_kwargs: object) -> object:
        return type("Completed", (), {"stdout": "0.7.0a0\n"})()

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    assert doctor._repository_version(ROOT) == "0.7.0a0"


def test_active_uv_surfaces_share_the_repository_pin() -> None:
    pinned = (ROOT / ".uv-version").read_text().strip()
    assert f"ghcr.io/astral-sh/uv:{pinned}-" in (ROOT / "Dockerfile").read_text()
    setup_files = [
        ROOT / ".github" / "actions" / "setup-python-tests" / "action.yml",
        ROOT / ".github" / "actions" / "setup-lean" / "action.yml",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".github" / "workflows" / "release.yml",
        ROOT / ".github" / "workflows" / "release-please.yml",
    ]
    for path in setup_files:
        text = path.read_text()
        assert text.count("astral-sh/setup-uv@") == text.count(f'version: "{pinned}"')


def test_every_bootstrap_profile_audits_z3_and_networkx() -> None:
    doctor = _load_tool("source_agent_doctor")
    for providers in doctor._PROFILE_PROVIDERS.values():
        assert "z3" in providers
        assert "networkx" in providers


def test_bootstrap_dry_run_and_client_preflight_fail_closed() -> None:
    script = (ROOT / "scripts" / "setup-agent").read_text()
    assert "uv python find --no-python-downloads" in script
    assert "unknown MCP client" in script
    assert '--provider-path "$PATH"' in script
    assert '--project-environment "$UV_PROJECT_ENVIRONMENT"' in script
    assert '--elan-home "$ELAN_HOME"' in script
    assert '--lean-runtime "$JACOBIAN_LEAN_RUNTIME"' in script
    assert 'require_checkout_path_ignored "the Jacobian state directory"' in script
    assert 'require_checkout_path_ignored "a custom uv project environment"' in script
    assert '--python "$PYTHON_PATH"' in script
    assert 'Path(part or ".").resolve()' in script
    assert script.index("status --porcelain") < script.index("uv python find")


@pytest.mark.skipif(
    shutil.which("git") is None,
    reason="requires git to initialize a test checkout",
)
def test_nonexistent_checkout_directory_uses_git_directory_ignore_semantics(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".state/\n")
    state = tmp_path / ".state"
    assert not state.exists()
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "check-ignore", "-q", "--", f"{state}/"],
            check=False,
        ).returncode
        == 0
    )
    script = (ROOT / "scripts" / "setup-agent").read_text()
    assert 'check-ignore -q -- "$target/"' in script
