from __future__ import annotations

import os
import py_compile
import time
from pathlib import Path

import pytest

from jacobian.plugin_execution import PluginExecutor

_HAS_LINUX_PROCESS_IDENTITIES = Path("/proc/self/stat").exists()


def _wait_until_process_exits(
    identity: str,
    *,
    timeout_seconds: float = 2,
) -> None:
    pid_text, expected_start_time = identity.split(":", 1)
    stat_path = Path(f"/proc/{pid_text}/stat")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            current_start_time = (
                stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
            )
        except (FileNotFoundError, ProcessLookupError):
            return
        if current_start_time != expected_start_time:
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"descendant process {pid_text} remained alive")
        time.sleep(0.01)


def test_plugin_executor_returns_only_canonical_result() -> None:
    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:echo",
        request={"candidate": {"value": 3}},
        timeout_seconds=30,
    )

    assert result.status.value == "COMPLETED"
    assert result.output == {"seen": {"candidate": {"value": 3}}}
    assert "untrusted plugin diagnostic" in result.diagnostics


def test_plugin_executor_rejects_changed_implementation_digest() -> None:
    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:echo",
        implementation_digest="sha256:" + "0" * 64,
        request={"candidate": {"value": 3}},
        timeout_seconds=30,
    )

    assert result.status.value == "ERROR"
    assert result.output is None
    assert result.detail == (
        "The plugin changed after it was registered. "
        "Reload Jacobian to register the current plugin version, then retry."
    )


def test_plugin_executor_does_not_expose_untrusted_exception_text() -> None:
    result = PluginExecutor().run(
        entrypoint="tests.support.search_entrypoints:propose_declared_failure",
        request={},
        timeout_seconds=30,
    )

    assert result.status.value == "ERROR"
    assert result.output is None
    assert result.detail == (
        "The plugin stopped before returning a result. Retry once; "
        "if it happens again, inspect the local plugin log."
    )
    assert "fixture" not in result.detail


def test_module_import_diagnostics_do_not_corrupt_worker_protocol() -> None:
    result = PluginExecutor().run(
        entrypoint="tests.support.plugin_noisy_module:echo",
        request={"value": 7},
        timeout_seconds=30,
    )

    assert result.status.value == "COMPLETED"
    assert result.output == {"seen": {"value": 7}}
    assert "module import diagnostic" in result.diagnostics


def test_plugin_timeout_has_no_mathematical_output() -> None:
    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:wait_forever",
        request={},
        timeout_seconds=1,
    )

    assert result.status.value == "TIMEOUT"
    assert result.output is None
    assert result.detail == (
        "The plugin did not finish within the allowed time. "
        "Retry with a larger time budget or a smaller request."
    )


def test_plugin_unreadable_response_explains_recovery() -> None:
    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:exit_without_response",
        request={},
        timeout_seconds=30,
    )

    assert result.status.value == "ERROR"
    assert result.output is None
    assert result.detail == (
        "The plugin returned an unreadable response. Retry once; "
        "if it happens again, inspect the local plugin log."
    )


def test_plugin_diagnostic_limit_fails_closed() -> None:
    start = time.monotonic()
    result = PluginExecutor(max_diagnostic_bytes=32).run(
        entrypoint="tests.support.process_entrypoints:emit_large_diagnostic",
        request={},
        timeout_seconds=30,
    )

    # Bound kill/fail-closed latency under load; not a performance SLO.
    assert time.monotonic() - start < 30
    assert result.status.value == "ERROR"
    assert result.output is None
    assert result.detail == (
        "The plugin produced too many diagnostics. Retry with a smaller request "
        "and inspect the local plugin log if the limit is reached again."
    )


@pytest.mark.skipif(
    not _HAS_LINUX_PROCESS_IDENTITIES,
    reason="requires non-signaling Linux process identities",
)
def test_plugin_timeout_kills_descendant_processes(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"
    started_marker = tmp_path / "descendant-started"
    pid_marker = tmp_path / "descendant-pid"

    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:spawn_delayed_child",
        request={
            "marker": str(marker),
            "started_marker": str(started_marker),
            "pid_marker": str(pid_marker),
            "delay_seconds": 3,
        },
        timeout_seconds=2,
    )

    assert started_marker.read_text(encoding="utf-8") == "started"
    assert result.status.value == "TIMEOUT"
    _wait_until_process_exits(pid_marker.read_text(encoding="utf-8"))
    assert not marker.exists()


@pytest.mark.skipif(
    not _HAS_LINUX_PROCESS_IDENTITIES,
    reason="requires non-signaling Linux process identities",
)
def test_plugin_success_kills_descendant_holding_output_pipes(tmp_path: Path) -> None:
    marker = tmp_path / "pipe-holder-survived"
    pid_marker = tmp_path / "pipe-holder-pid"
    start = time.monotonic()

    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:spawn_child_then_return",
        request={"marker": str(marker), "pid_marker": str(pid_marker)},
        timeout_seconds=2,
    )
    elapsed = time.monotonic() - start

    # Bound kill latency under load; not a performance SLO.
    assert elapsed < 30
    assert result.status.value == "COMPLETED"
    assert result.output == {"worker": "returned"}
    _wait_until_process_exits(pid_marker.read_text(encoding="utf-8"))
    assert not marker.exists()


@pytest.mark.skipif(
    not _HAS_LINUX_PROCESS_IDENTITIES,
    reason="requires non-signaling Linux process identities",
)
def test_plugin_success_still_kills_detached_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "detached-descendant-survived"
    pid_marker = tmp_path / "detached-descendant-pid"

    result = PluginExecutor().run(
        entrypoint=(
            "tests.support.process_entrypoints:spawn_detached_child_then_return"
        ),
        request={"marker": str(marker), "pid_marker": str(pid_marker)},
        timeout_seconds=5,
    )

    assert result.status.value == "COMPLETED"
    _wait_until_process_exits(pid_marker.read_text(encoding="utf-8"))
    assert not marker.exists()


def test_plugin_worker_does_not_inherit_parent_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JACOBIAN_TEST_SECRET", "do-not-forward")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret.invalid")

    result = PluginExecutor().run(
        entrypoint="tests.support.process_entrypoints:report_environment",
        request={},
        timeout_seconds=5,
    )

    assert result.status.value == "COMPLETED"
    assert result.output == {"secret": None, "https_proxy": None}


def test_plugin_worker_rejects_bytecode_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "bytecode_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "entry.py").write_text(
        "from .helper import VALUE\ndef run(_request):\n    return {'value': VALUE}\n",
        encoding="utf-8",
    )
    helper = package / "helper.py"
    helper.write_text("VALUE = 7\n", encoding="utf-8")
    py_compile.compile(
        str(helper),
        cfile=str(package / "helper.pyc"),
        doraise=True,
    )
    helper.unlink()
    monkeypatch.syspath_prepend(str(tmp_path))
    existing_path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path) if not existing_path else f"{tmp_path}:{existing_path}",
    )

    result = PluginExecutor().run(
        entrypoint="bytecode_plugin.entry:run",
        request={},
        timeout_seconds=5,
    )

    assert result.status.value == "ERROR"
    assert result.detail == (
        "The plugin stopped before returning a result. Retry once; "
        "if it happens again, inspect the local plugin log."
    )
