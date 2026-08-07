from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tools.test_topology import lane_environment, load_topology, main, pytest_command

ROOT = Path(__file__).resolve().parents[4]


def test_domain_lane_dry_run_is_explicit_and_topology_owned() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/test_topology.py",
            "domain",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "tests/domain" in result.stdout
    assert "--timeout 120" in result.stdout


def test_harbor_execution_check_stays_out_of_the_full_verifier_corpus() -> None:
    result = subprocess.run(
        ["make", "--dry-run", "harbor-execution-check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "tools/check_harbor_dataset.py --check" in result.stdout
    assert "tools/check_benchmark_contracts.py" in result.stdout
    assert "tests/unit/tooling/test_harbor*.py" in result.stdout
    assert "benchmarks/validation" not in result.stdout
    assert "harbor-oracle" not in result.stdout


def test_make_semantic_lane_forwards_pytest_arguments() -> None:
    result = subprocess.run(
        [
            "make",
            "--dry-run",
            "test-process",
            "TESTS=tests/boundary/process/tooling/test_topology_runner.py",
            "PYTEST_ARGS=-k target_test --junitxml=pytest.xml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert '--pytest-args "--durations=10 -k target_test --junitxml=pytest.xml"' in (
        result.stdout
    )


def test_topology_runner_executes_pytest_via_command_runner(monkeypatch) -> None:
    from types import SimpleNamespace

    from tools import test_topology

    topology = test_topology.load_topology()
    observed: dict[str, object] = {}

    def fake_run_pytest(arguments: object, **kwargs: object) -> object:
        observed.update(arguments=arguments, **kwargs)
        return SimpleNamespace(exit_code=0)

    monkeypatch.setattr(
        test_topology,
        "run_pytest",
        fake_run_pytest,
    )

    rc = test_topology.run_lane(
        topology,
        "unit",
        ["tests/unit/tooling/test_fixture_architecture.py"],
        ["-q"],
    )

    assert rc == 0
    assert observed["arguments"] == [
        "tests/unit/tooling/test_fixture_architecture.py",
        "-q",
        "--timeout",
        "10",
    ]
    environment = observed["environment"]
    assert environment is not None
    assert environment["JACOBIAN_TEST_LANE"] == "unit"
    assert observed["timeout_seconds"] == 3600.0


def test_topology_runner_returns_pytest_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    from tools import test_topology

    topology = test_topology.load_topology()

    monkeypatch.setattr(
        test_topology,
        "run_pytest",
        lambda *args, **kwargs: SimpleNamespace(exit_code=1),
    )

    rc = test_topology.run_lane(topology, "unit", ["tests/unit/test_bad.py"], [])

    assert rc == 1


def test_focused_selector_does_not_start_configured_workers() -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")

    command = pytest_command(
        topology,
        "process",
        ["tests/boundary/process/tooling/test_topology_runner.py::test_x"],
    )

    assert "-n" not in command
    assert "--dist" not in command
    assert "tests/boundary/process/tooling/test_topology_runner.py::test_x" in command


def test_full_parallel_lane_retains_configured_workers() -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")

    command = pytest_command(topology, "process")

    assert command[command.index("-n") + 1] == "2"
    assert command[command.index("--dist") + 1] == "worksteal"


def test_explicit_xdist_zero_suppresses_lane_worker_pool() -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")

    command = pytest_command(topology, "process", extra_args=["-n", "0"])

    # The user's serial request is honored; lane defaults do not override it.
    assert command[command.index("-n") + 1] == "0"
    assert "--dist" not in command


def test_explicit_xdist_zero_equals_form_suppresses_lane_worker_pool() -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")

    command = pytest_command(topology, "process", extra_args=["-n=0"])

    assert "-n=0" in command
    assert "--dist" not in command


def test_explicit_numprocesses_suppresses_lane_worker_pool() -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")

    command = pytest_command(topology, "process", extra_args=["--numprocesses", "0"])

    assert command[command.index("--numprocesses") + 1] == "0"
    assert "--dist" not in command
    assert "-n" not in command


def test_dry_run_explicit_xdist_zero_suppresses_lane_workers(capsys) -> None:
    rc = main(["process", "--pytest-args=-n 0", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    command = out.splitlines()[-1]
    # The lane's configured ``-n 2 --dist worksteal`` is not appended.
    assert "-n 0" in command
    assert "-n 2" not in command
    assert "--dist" not in command


def test_focused_serial_lane_remains_serial_and_preserves_extra_args() -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")

    command = pytest_command(
        topology,
        "lean",
        ["tests/boundary/providers/lean/test_lean_repl_runtime.py::test_x"],
        ["-k", "test_x"],
    )

    assert "-n" not in command
    assert command[command.index("-k") + 1] == "test_x"
    assert command[command.index("--timeout") + 1] == "300"


def test_dry_run_full_lane_reports_configured_metadata_and_command(capsys) -> None:
    rc = main(["composition", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "# lane: composition" in out
    assert "# tier: composition" in out
    assert "# workers: 2" in out
    assert "# distribution: worksteal" in out
    assert "# timeout_seconds: 120" in out
    assert "# timing_sharding: true" in out
    assert "# selectors:" not in out
    # The command remains the final un-prefixed line and stays consumable.
    assert "tests/composition" in out
    assert "--timeout 120" in out


def test_dry_run_focused_selector_reports_zero_workers(capsys) -> None:
    rc = main(
        [
            "composition",
            "tests/composition/test_x.py::test_x",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "# workers: 0" in out
    assert "# distribution: none" in out
    assert "# selectors: 1" in out
    assert "-n" not in out.splitlines()[-1]


def test_dry_run_forwards_explicit_pytest_arguments(capsys) -> None:
    rc = main(
        [
            "process",
            "--pytest-args=-k target_test --junitxml=pytest.xml",
            "tests/boundary/process/tooling/test_topology_runner.py",
            "--dry-run",
        ]
    )
    command = capsys.readouterr().out.splitlines()[-1]

    assert rc == 0
    assert "--pytest-args=" not in command
    assert "-k target_test" in command
    assert "--junitxml=pytest.xml" in command


def test_dry_run_serial_lane_reports_no_workers(capsys) -> None:
    rc = main(["storage", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "# lane: storage" in out
    assert "# workers: 0" in out
    assert "# distribution: none" in out
    assert "# timing_sharding: false" in out
    assert "tests/boundary/storage" in out


def test_lane_environment_forwards_only_allowlisted_lean_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")
    fake_home = str(tmp_path / "home")
    fake_elan = str(tmp_path / "elan")
    monkeypatch.setenv("HOME", fake_home)
    monkeypatch.setenv("ELAN_HOME", fake_elan)
    monkeypatch.setenv("JACOBIAN_TOPOLOGY_LEAK", "secret")

    lean = lane_environment(topology.lane("lean"))
    assert lean["JACOBIAN_TEST_LANE"] == "lean"
    assert lean["PATH"] == os.environ["PATH"]
    assert lean["HOME"] == fake_home
    assert lean["ELAN_HOME"] == fake_elan
    assert "JACOBIAN_TOPOLOGY_LEAK" not in lean

    provider = lane_environment(topology.lane("provider"))
    assert provider["JACOBIAN_TEST_LANE"] == "provider"
    assert provider["PATH"] == os.environ["PATH"]
    assert provider["HOME"] == fake_home
    assert provider["ELAN_HOME"] == fake_elan
    assert "JACOBIAN_TOPOLOGY_LEAK" not in provider

    unit = lane_environment(topology.lane("unit"))
    assert unit["JACOBIAN_TEST_LANE"] == "unit"
    assert unit["PATH"] == os.environ["PATH"]
    assert "HOME" not in unit
    assert "ELAN_HOME" not in unit
    assert "JACOBIAN_TOPOLOGY_LEAK" not in unit


def test_lane_environment_never_forwards_unauthorized_host_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = load_topology(ROOT / "tests" / "topology.toml")
    monkeypatch.setenv("JACOBIAN_TOPOLOGY_LEAK", "secret")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    for lane_name in ("lean", "provider", "unit", "process", "storage"):
        environment = lane_environment(topology.lane(lane_name))
        assert "JACOBIAN_TOPOLOGY_LEAK" not in environment, lane_name
        assert environment["JACOBIAN_TEST_LANE"] == lane_name
