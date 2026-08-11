from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]


def _load_profiles() -> ModuleType:
    path = ROOT / "tools" / "development_profiles.py"
    spec = importlib.util.spec_from_file_location("development_profiles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profiles_share_locked_sync_contracts() -> None:
    profiles = _load_profiles()

    assert profiles.PROFILE_NAMES == (
        "core",
        "full-python",
        "lean",
        "external-proof",
    )
    assert profiles.sync_arguments("core") == ("sync", "--locked", "--dev")
    for name in ("full-python", "lean", "external-proof"):
        assert profiles.sync_arguments(name) == (
            "sync",
            "--locked",
            "--dev",
            "--all-extras",
        )


def test_invalid_profile_reports_all_supported_choices() -> None:
    profiles = _load_profiles()

    with pytest.raises(ValueError, match="core, full-python, lean, external-proof"):
        profiles.profile("everything")


def test_lean_setup_installs_pin_then_gets_cache_and_builds_without_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = _load_profiles()
    repo = tmp_path / "repo"
    (repo / "lean").mkdir(parents=True)
    (repo / "lean" / "lean-toolchain").write_text(
        "leanprover/lean4:v4.31.0\n", encoding="utf-8"
    )
    commands: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(profiles, "_tool_path", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        profiles,
        "_uv_diagnostic",
        lambda _repo: profiles.Diagnostic(
            "uv", "available", "0.11.28", "0.11.28", "", ""
        ),
    )

    def record(arguments: object, cwd: Path) -> int:
        commands.append((tuple(arguments), cwd))
        return 0

    profiles.setup_profile(repo, "lean", run=record)

    assert commands[:4] == [
        (("uv", "sync", "--locked", "--dev", "--all-extras"), repo),
        (
            ("elan", "toolchain", "install", "leanprover/lean4:v4.31.0"),
            repo,
        ),
        (("lake", "exe", "cache", "get"), repo / "lean"),
        (
            ("lake", "build", "repl", "jacobian_lean_proof_state"),
            repo / "lean",
        ),
    ]
    assert all("update" not in command for command, _cwd in commands)
    assert commands[-1][0][:6] == (
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "python",
        str(repo / "tools" / "development_profiles.py"),
    )


def test_external_proof_setup_never_downloads_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = _load_profiles()
    repo = tmp_path / "repo"
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        profiles,
        "_uv_diagnostic",
        lambda _repo: profiles.Diagnostic(
            "uv", "available", "0.11.28", "0.11.28", "", ""
        ),
    )

    def record(arguments: object, _cwd: Path) -> int:
        commands.append(tuple(arguments))
        return 0

    profiles.setup_profile(repo, "external-proof", run=record)
    first_run = list(commands)
    profiles.setup_profile(repo, "external-proof", run=record)

    assert first_run[0] == (
        "uv",
        "sync",
        "--locked",
        "--dev",
        "--all-extras",
    )
    assert commands == [*first_run, *first_run]
    assert all(command[0] == "uv" for command in commands)


def test_doctor_distinguishes_available_unavailable_and_incompatible_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = _load_profiles()

    monkeypatch.setattr(profiles.importlib.metadata, "version", lambda _name: "1.3.4")
    available = profiles._distribution_diagnostic(
        requirement="cvc5",
        distribution="cvc5",
        expected="==1.3.4",
        profile_name="full-python",
    )
    monkeypatch.setattr(profiles.importlib.metadata, "version", lambda _name: "1.2.0")
    incompatible = profiles._distribution_diagnostic(
        requirement="cvc5",
        distribution="cvc5",
        expected="==1.3.4",
        profile_name="full-python",
    )

    def missing(_name: str) -> str:
        raise profiles.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(profiles.importlib.metadata, "version", missing)
    unavailable = profiles._distribution_diagnostic(
        requirement="cvc5",
        distribution="cvc5",
        expected="==1.3.4",
        profile_name="full-python",
    )

    assert [available.status, incompatible.status, unavailable.status] == [
        "available",
        "incompatible",
        "unavailable",
    ]
    assert unavailable.recovery == "Run `make setup PROFILE=full-python`"
    assert unavailable.documentation == profiles.OPTIONAL_BACKEND_DOCUMENTATION
