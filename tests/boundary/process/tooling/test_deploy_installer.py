from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
INSTALLER = REPOSITORY_ROOT / "deploy" / "install.sh"
SERVICE_STATE = REPOSITORY_ROOT / "deploy" / "lib" / "service_state.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="deploy installer tests require bash",
)


def _run(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _run_with_resolved_ancestor(
    tmp_path: Path,
    *,
    resolved_ancestor: str,
) -> subprocess.CompletedProcess[str]:
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    readlink = tool_directory / "readlink"
    readlink.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$JACOBIAN_TEST_READLINK_TARGET\"\n",
        encoding="utf-8",
    )
    readlink.chmod(0o755)
    environment = dict(os.environ)
    environment["JACOBIAN_TEST_READLINK_TARGET"] = resolved_ancestor
    environment["PATH"] = f"{tool_directory}:{environment['PATH']}"
    return _run(
        "--install-root",
        "/srv/jacobian-test-alias/release-root/jacobian",
        "--dry-run",
        environment=environment,
    )


def test_installer_is_valid_bash() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(INSTALLER)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_installer_help_exposes_three_deployment_modes() -> None:
    completed = _run("--help")

    assert completed.returncode == 0
    assert "--mode local" in completed.stdout
    assert "--mode domain" in completed.stdout
    assert "--mode tailscale" in completed.stdout
    assert "--install-root" in completed.stdout
    assert "--with-lean" in completed.stdout


def test_lean_dry_run_uses_a_distinct_release_profile() -> None:
    completed = _run("--with-lean", "--dry-run")

    assert completed.returncode == 0, completed.stderr
    release_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.strip().startswith("release:")
    )
    assert release_line.endswith("-lean")
    assert "lean:        pinned CORE + MATHLIB runtime" in completed.stdout


def test_domain_dry_run_reports_connector_without_requiring_root() -> None:
    completed = _run(
        "--mode",
        "domain",
        "--domain",
        "math.example.org",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "mode:        domain" in completed.stdout
    assert "connector:   https://math.example.org/mcp" in completed.stdout
    assert "python:      /opt/jacobian/python" in completed.stdout
    assert "caddy:       enabled" in completed.stdout
    assert "funnel:      disabled" in completed.stdout


def test_dry_run_derives_every_runtime_path_from_custom_install_root() -> None:
    completed = _run(
        "--install-root",
        "/srv/math/jacobian",
        "--with-lean",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "install:     /srv/math/jacobian" in completed.stdout
    assert "release:     /srv/math/jacobian/releases/" in completed.stdout
    assert "python:      /srv/math/jacobian/python" in completed.stdout


@pytest.mark.parametrize(
    "root",
    (
        "relative/path",
        "/",
        "/srv/path with spaces",
        "/srv/../jacobian",
        "/home/apps/jacobian",
        "/root",
        "/run/jacobian",
        "/run/user/1000/jacobian",
        "/dev/shm/jacobian",
        "/tmp/jacobian",
        "/var/tmp/jacobian",
    ),
)
def test_install_root_rejects_unsafe_or_ambiguous_paths(root: str) -> None:
    completed = _run("--install-root", root, "--dry-run")

    assert completed.returncode != 0
    assert "--install-root" in completed.stderr


def test_install_root_rejects_an_allowed_symlink_into_a_hidden_path(
    tmp_path: Path,
) -> None:
    completed = _run_with_resolved_ancestor(
        tmp_path,
        resolved_ancestor=str(tmp_path),
    )

    assert completed.returncode != 0
    assert "resolves below a path hidden by the systemd sandbox" in completed.stderr


@pytest.mark.parametrize("resolved_ancestor", ("/run", "/dev/shm"))
def test_install_root_rejects_an_allowed_symlink_into_volatile_runtime(
    tmp_path: Path,
    resolved_ancestor: str,
) -> None:
    completed = _run_with_resolved_ancestor(
        tmp_path,
        resolved_ancestor=resolved_ancestor,
    )

    assert completed.returncode != 0
    assert "resolves below a volatile runtime hierarchy" in completed.stderr


def test_install_root_canonicalizes_an_allowed_symlink_ancestor(
    tmp_path: Path,
) -> None:
    completed = _run_with_resolved_ancestor(
        tmp_path,
        resolved_ancestor="/opt",
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "install:     /opt/jacobian-test-alias/release-root/jacobian"
        in completed.stdout
    )


@pytest.mark.parametrize("target_name", ("actual root", "actual|root"))
def test_install_root_rejects_unsafe_resolved_symlink_targets(
    tmp_path: Path,
    target_name: str,
) -> None:
    completed = _run_with_resolved_ancestor(
        tmp_path,
        resolved_ancestor=f"/srv/{target_name}",
    )

    assert completed.returncode != 0
    assert "resolves to a non-root path with unsupported characters" in (
        completed.stderr
    )


def test_dry_run_never_echoes_supplied_credentials(tmp_path: Path) -> None:
    sentinel = "sentinel-secret-that-must-not-be-logged"
    credentials = tmp_path / "tokens.json"
    credentials.write_text(
        '{"tokens":{"' + sentinel + '":{"tenant_id":"tenant-a"}}}',
        encoding="utf-8",
    )

    completed = _run("--auth-tokens-file", str(credentials), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr


def test_domain_mode_requires_a_valid_fqdn() -> None:
    missing = _run("--mode", "domain", "--dry-run")
    invalid = _run(
        "--mode",
        "domain",
        "--domain",
        "https://math.example.org/path",
        "--dry-run",
    )

    assert missing.returncode != 0
    assert "--mode domain requires --domain" in missing.stderr
    assert invalid.returncode != 0
    assert "fully qualified DNS name" in invalid.stderr


def test_public_anonymous_mode_requires_double_confirmation() -> None:
    rejected = _run(
        "--mode",
        "domain",
        "--domain",
        "math.example.org",
        "--allow-anonymous",
        "--dry-run",
    )
    accepted = _run(
        "--mode",
        "domain",
        "--domain",
        "math.example.org",
        "--allow-anonymous",
        "--confirm-public-anonymous",
        "--dry-run",
    )

    assert rejected.returncode != 0
    assert "--confirm-public-anonymous" in rejected.stderr
    assert accepted.returncode == 0, accepted.stderr
    assert "auth:        anonymous shared tenant jacobian-test" in accepted.stdout


def test_local_mode_rejects_an_unusable_domain_option() -> None:
    completed = _run(
        "--mode",
        "local",
        "--domain",
        "math.example.org",
        "--dry-run",
    )

    assert completed.returncode != 0
    assert "--domain is only valid with --mode domain" in completed.stderr


def test_release_environment_is_built_at_its_final_path() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    release_block = source[
        source.index('log "installing immutable release') : source.index(
            'log "installing authentication configuration"'
        )
    ]

    assert 'cd "${RELEASE_DIR}"' in release_block
    assert 'UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_ROOT}"' in release_block
    assert "--managed-python" in release_block
    assert "--link-mode copy" in release_block
    assert 'mv "${RELEASE_CANDIDATE}" "${RELEASE_DIR}"' not in release_block


def test_deployment_lock_is_host_global_and_precedes_shared_host_mutations() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert 'DEPLOY_LOCK_PATH="/run/lock/jacobian-mcp-install.lock"' in source
    assert 'exec 9>"${DEPLOY_LOCK_PATH}"' in source
    assert '$(dirname -- "${RELEASE_ROOT}")/.install.lock' not in source
    lock = source.index('"${FLOCK_BIN}" --nonblock 9')
    first_shared_mutation = source.index("groupadd --system jacobian")

    assert lock < first_shared_mutation


def test_release_runtime_is_checked_before_current_symlink_is_changed() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    ownership = source.index(
        'chown -R root:root "${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}"'
    )
    runtime_permissions = source.index(
        'chmod -R a+rX "${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}"'
    )
    validation = source.index('validate_release_runtime "${RELEASE_DIR}"')
    revision_marker = source.index(
        'printf \'%s\\n\' "${REVISION}" >"${RELEASE_DIR}/.git-revision"'
    )
    marker_permissions = source.index(
        '"${RELEASE_DIR}/.release-profile"', revision_marker
    )
    current_link = source.index('ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"')

    assert ownership < runtime_permissions < validation
    assert validation < revision_marker < marker_permissions < current_link
    assert '"${RUNUSER_BIN}" --user jacobian -- "${entrypoint}" --version' in source


def test_runtime_inputs_remain_service_readable_after_root_probes() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "export PYTHONDONTWRITEBYTECODE=1" in source
    assert 'find "${runtime_root}"' in source
    assert "-type d \\( ! -readable -o ! -executable \\)" in source
    assert "-type f ! -readable" in source
    assert 'RUNTIME_READ_ROOTS=("${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}")' in source
    assert 'RUNTIME_READ_ROOTS+=("${LEAN_ELAN_HOME}")' in source

    first_audit = source.index(
        'validate_service_readability "${RUNTIME_READ_ROOTS[@]}"'
    )
    current_link = source.index('ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"')
    smoke = source.index('log "running the read-only deployment smoke"')
    final_audit = source.rindex(
        'validate_service_readability "${RUNTIME_READ_ROOTS[@]}"'
    )
    accepted = source.index("DEPLOYMENT_ACCEPTED=1")

    assert first_audit < current_link < smoke < final_audit < accepted


def test_lean_profile_is_built_and_validated_before_activation() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    install_toolchain = source.index(
        '"${LEAN_ELAN_HOME}/bin/elan" toolchain install "${LEAN_TOOLCHAIN}"'
    )
    inspect_toolchains = source.index('"${LEAN_ELAN_HOME}/bin/elan" toolchain list')
    fetch_cache = source.index("lake exe cache get")
    build_runtime = source.index(
        "lake build repl JacobianLeanRuntime jacobian_lean_proof_state"
    )
    validate = source.index('validate_lean_release_runtime "${RELEASE_DIR}"')
    shared_toolchain_permissions = source.index('chmod -R a+rX "${LEAN_ELAN_HOME}"')
    revision_marker = source.index(
        'printf \'%s\\n\' "${REVISION}" >"${RELEASE_DIR}/.git-revision"'
    )
    current_link = source.index('ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"')

    assert inspect_toolchains < install_toolchain < fetch_cache < build_runtime
    assert build_runtime < shared_toolchain_permissions < validate
    assert validate < revision_marker < current_link
    assert '"ELAN_HOME=${LEAN_ELAN_HOME}"' in source
    assert '"PATH=${LEAN_SERVICE_PATH}"' in source
    assert 'chmod -R a+rX "${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}"' in source
    assert "lean_provider_runtime(" in source
    assert "ProviderAvailability.AVAILABLE" in source


def test_lean_profile_finds_the_invoking_users_elan_under_sudo() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    resolve_home = source.index(
        'INVOKING_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6 || true)"'
    )
    elan_fallback = source.index('ELAN_FALLBACKS+=("${INVOKING_HOME}/.elan/bin/elan")')
    resolve_elan = source.index('find_executable elan "${ELAN_FALLBACKS[@]}"')

    assert resolve_home < elan_fallback < resolve_elan


def test_systemd_service_can_read_the_operator_managed_lean_toolchain() -> None:
    service = (REPOSITORY_ROOT / "deploy/systemd/jacobian-mcp.service").read_text(
        encoding="utf-8"
    )

    assert "Environment=ELAN_HOME=/opt/jacobian/lean/elan" in service
    assert "Environment=PATH=/opt/jacobian/lean/elan/bin:" in service
    assert (
        "Environment=JACOBIAN_DEPLOYMENT_REVISION_FILE="
        "/opt/jacobian/current/.git-revision" in service
    )
    assert "ProtectHome=true" in service


def test_installer_renders_custom_runtime_paths_into_service_and_override() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert '-e "s|/opt/jacobian/current|${CURRENT_LINK}|g"' in source
    assert '-e "s|/opt/jacobian/lean/elan|${LEAN_ELAN_HOME}|g"' in source
    assert source.count('-e "s|/opt/jacobian/current|${CURRENT_LINK}|g"') == 2


def test_lean_profile_requires_catalog_and_behavior_smokes() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    smoke_block = source[source.index('log "running the read-only deployment smoke"') :]

    for operation_id in (
        "lean.check",
        "lean.proof_state.apply_tactic",
        "lean.term.apply",
        "lean.retrieve.premises",
    ):
        assert f"--require-operation {operation_id}" in smoke_block
    assert '--expect-revision "${REVISION}"' in smoke_block
    assert '"${RELEASE_DIR}/deploy/smoke_lean.py"' in smoke_block


def test_activation_arms_rollback_before_switching_current() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    armed = source.index("ROLLBACK_ARMED=1")
    current_link = source.index('ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"')
    smoke = source.index('log "running the read-only deployment smoke"')
    accepted = source.index("DEPLOYMENT_ACCEPTED=1")

    assert armed < current_link < smoke < accepted
    assert 'return "${original_status}"' in source
    assert 'exit "${status}"' in source
    assert "rollback encountered additional failures" in source


def test_installer_compiles_state_before_service_activation() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    snapshot = source.index('snapshot_file state "${STATE_DIR}"')
    lifecycle = source.index('log "running jacobian ${STATE_COMMAND}')
    restart = source.index('"${SYSTEMCTL_BIN}" restart jacobian-mcp.service', lifecycle)
    assert snapshot < lifecycle < restart
    assert source.index('"${SYSTEMCTL_BIN}" stop jacobian-mcp.service') < snapshot
    assert 'restore_file state "${STATE_DIR}"' in source
    assert '"${RUNUSER_BIN}" -u jacobian -- env' in source
    assert '--state-dir "${STATE_DIR}"' in source
    assert 'STATE_COMMAND="init"' in source
    assert 'STATE_COMMAND="update"' in source


def test_generated_token_is_written_only_to_the_restricted_file() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600" in source
    assert "print(token)" not in source
    assert "print(next(grant.token" not in source
    assert "JACOBIAN_MCP_AUTH_TOKENS_FILE" in source
    assert "generated bearer token: ${" not in source
    assert "retrieve it explicitly with privileged access" in source


def test_rollback_restores_prior_service_activity_and_enablement(
    tmp_path: Path,
) -> None:
    state = tmp_path / "systemd-state"
    snapshots = tmp_path / "snapshots"
    state.mkdir()
    snapshots.mkdir()
    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text(
        """#!/usr/bin/env bash
set -eu
action="$1"
case "$action" in
  is-enabled) test -f "$FAKE_SYSTEMD_STATE/$3.enabled" ;;
  is-active) test -f "$FAKE_SYSTEMD_STATE/$3.active" ;;
  enable) : >"$FAKE_SYSTEMD_STATE/$2.enabled" ;;
  disable) rm -f -- "$FAKE_SYSTEMD_STATE/$2.enabled" ;;
  restart) : >"$FAKE_SYSTEMD_STATE/$2.active" ;;
  stop) rm -f -- "$FAKE_SYSTEMD_STATE/$2.active" ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_systemctl.chmod(0o755)
    (state / "jacobian-mcp.service.enabled").touch()
    (state / "jacobian-mcp.service.active").touch()
    (state / "jacobian-caddy.service.enabled").touch()
    environment = os.environ | {"FAKE_SYSTEMD_STATE": str(state)}
    units = (
        "jacobian-mcp.service",
        "jacobian-caddy.service",
        "jacobian-funnel.service",
    )

    for unit in units:
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; snapshot_systemd_service_state "$2" "$3" "$4"',
                "service-state-test",
                str(SERVICE_STATE),
                str(fake_systemctl),
                str(snapshots),
                unit,
            ],
            check=True,
            env=environment,
        )
        (state / f"{unit}.enabled").touch()
        (state / f"{unit}.active").touch()

    for unit in units:
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; restore_systemd_service_state "$2" "$3" "$4"',
                "service-state-test",
                str(SERVICE_STATE),
                str(fake_systemctl),
                str(snapshots),
                unit,
            ],
            check=True,
            env=environment,
        )

    assert (state / "jacobian-mcp.service.enabled").is_file()
    assert (state / "jacobian-mcp.service.active").is_file()
    assert (state / "jacobian-caddy.service.enabled").is_file()
    assert not (state / "jacobian-caddy.service.active").exists()
    assert not (state / "jacobian-funnel.service.enabled").exists()
    assert not (state / "jacobian-funnel.service.active").exists()
