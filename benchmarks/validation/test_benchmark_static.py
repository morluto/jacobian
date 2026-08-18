from __future__ import annotations

from benchmarks.tooling.command_runner import ToolCommandResult, ToolCommandStatus
from tools import check_benchmark_static


def test_static_commands_scan_benchmarks_without_execution_commands() -> None:
    commands = check_benchmark_static._commands()

    assert [label for label, _ in commands] == ["Ruff lint", "Ruff format", "mypy"]
    assert any(
        "benchmarks" in argument for _, command in commands[:2] for argument in command
    )
    assert all(
        argument not in {"pytest", "harbor", "oracle", "model"}
        for _, command in commands
        for argument in command
    )


def test_static_gate_stops_and_fails_closed_on_a_failed_check(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(request):
        calls.append((request.executable, *request.arguments))
        assert request.cwd == str(check_benchmark_static.ROOT)
        assert request.timeout_seconds == 300.0
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=9,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(check_benchmark_static, "run_tool_command", fake_run)

    assert check_benchmark_static.main() == 9
    assert len(calls) == 1


def test_mypy_covers_package_not_individual_files() -> None:
    """Mypy targets should be package-level, not a per-file allowlist."""
    targets = check_benchmark_static.MYPY_TARGETS
    # At least one package-level target
    package_targets = [t for t in targets if not t.endswith(".py")]
    assert package_targets, "MYPY_TARGETS should include package-level targets"
    # Must not use --follow-imports=skip
    commands = check_benchmark_static._commands()
    mypy_command = next(cmd for label, cmd in commands if label == "mypy")
    assert "--follow-imports=skip" not in mypy_command


