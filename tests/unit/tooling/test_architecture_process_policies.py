"""Architecture checker contracts for process and environment confinement."""

from __future__ import annotations

from pathlib import Path

from tools.check_architecture import check_architecture


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


# Fragments for constructing subprocess tokens without self-triggering.
_sub = "sub"
_proc = "proc" + "ess"
_sp = _sub + _proc


def test_subprocess_in_product_source_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/verification/service.py",
        "import subprocess\n\nsubprocess.run(['echo', 'hi'])\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "src/jacobian/verification/service.py"
    assert sub[0].line == 1


def test_subprocess_in_bounded_process_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/bounded_process.py",
        "import subprocess\n\nsubprocess.Popen(['true'])\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_command_runner_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "benchmarks/tooling/command_runner.py",
        "import subprocess\n\nsubprocess.Popen(['true'])\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_clean_room_lean_replay_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "benchmarks/datasets/provider-feasibility-v1/lean-repl/tests/replay.py",
        "import subprocess\n\nsubprocess.Popen(['repl'])\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_checkers_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/sat.py",
        "import subprocess\n\nsubprocess.run(['cadical'])\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "src/jacobian_checkers/sat.py"


def test_subprocess_in_unlisted_tool_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/unlisted_runner.py",
        "import subprocess\n\nsubprocess.run(['echo'])\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "tools/unlisted_runner.py"


def test_subprocess_in_development_profiles_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/development_profiles.py",
        "import subprocess\n\nsubprocess.run(['uv', 'sync'])\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_deleted_e2e_fixture_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/e2e/verified_results/test_reference_runtime.py",
        "import subprocess\n\nsubprocess.run(['echo'])\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1


def test_subprocess_in_explicit_boundary_fixture_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/boundary/process/test_bounded_process.py",
        "import subprocess\n\nsubprocess.run(['echo'])\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_unlisted_boundary_test_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "tests/boundary/process/test_random_new.py", "import subprocess\n")
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "tests/boundary/process/test_random_new.py"


def test_subprocess_in_unlisted_component_test_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "tests/component/test_random.py", "import subprocess\n")
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1


def test_os_execvpe_in_product_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/runner.py",
        "import os\n\nos.execvpe('python', ['python'], os.environ.copy())\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert "execvpe" in sub[0].message


def test_os_execvpe_in_unlisted_tool_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/unlisted_runner.py",
        "import os\n\nos.execvpe('python', ['python'], {})\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1


def test_subprocess_import_from_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/lean_frontend/repl.py",
        "from subprocess import Popen\n\nPopen(['lean'])\n",
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1


def test_embedded_subprocess_run_in_string_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/provider_measurements.py",
        f'_PROBE = r"""\nimport {_sp}\nprocess = {_sp}.run(\n    sys.argv[1:],\n    check=True,\n)\n"""\n',
    )
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert "embeds" in sub[0].message
    assert sub[0].line == 1


def test_embedded_subprocess_popen_in_string_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/worker.py", f"code = \"{_sp}.Popen(['echo'])\"\n")
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert "Popen" in sub[0].message


def test_embedded_subprocess_import_in_string_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/runner.py", f'_code = "import {_sp}\\n"\n')
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1


def test_embedded_subprocess_in_allowed_file_is_not_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/bounded_process.py",
        f'_DOC = "uses {_sp}.run internally"\n',
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_benign_string_without_subprocess_pattern_is_not_flagged(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "src/jacobian/config.py", 'message = "the process is running"\n')
    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_run_bounded_process_in_product_caller_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/verification/service.py",
        "from jacobian.bounded_process import run_bounded_process\nrun_bounded_process(['echo'])\n",
    )
    report = check_architecture(tmp_path)
    gateway = [v for v in report.violations if v.code == "bounded-process-gateway"]
    assert len(gateway) >= 1
    assert gateway[0].path == "src/jacobian/verification/service.py"


def test_run_bounded_process_in_process_policy_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/process_policy.py",
        "from jacobian.bounded_process import run_bounded_process\nrun_bounded_process(['echo'])\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "bounded-process-gateway" for v in report.violations)


def test_run_bounded_process_in_bounded_process_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/bounded_process.py",
        "from jacobian.bounded_process import run_bounded_process\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "bounded-process-gateway" for v in report.violations)


def test_run_bounded_process_in_checker_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/sat.py",
        "from jacobian.bounded_process import run_bounded_process\n",
    )
    report = check_architecture(tmp_path)
    gateway = [v for v in report.violations if v.code == "bounded-process-gateway"]
    assert len(gateway) == 1


def test_run_bounded_process_in_test_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/boundary/process/test_foo.py",
        "from jacobian.bounded_process import run_bounded_process\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "bounded-process-gateway" for v in report.violations)


def test_shutil_which_in_product_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/capability_service.py",
        "import shutil\n\nshutil.which('lean')\n",
    )
    report = check_architecture(tmp_path)
    which = [v for v in report.violations if v.code == "shutil-which-resolver"]
    assert len(which) == 1
    assert which[0].path == "src/jacobian/capability_service.py"


def test_shutil_which_in_process_policy_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/process_policy.py",
        "import shutil\n\nshutil.which('prlimit')\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "shutil-which-resolver" for v in report.violations)


def test_shutil_which_in_command_runner_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "benchmarks/tooling/command_runner.py",
        "import shutil\n\nshutil.which('git')\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "shutil-which-resolver" for v in report.violations)


def test_shutil_which_in_checker_lean4_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian_checkers/lean4.py",
        "import shutil\n\nshutil.which('elan')\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "shutil-which-resolver" for v in report.violations)


def test_shutil_import_without_which_is_not_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path, "src/jacobian/utils.py", "import shutil\n\nshutil.copy('a', 'b')\n"
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "shutil-which-resolver" for v in report.violations)


def test_dict_os_environ_in_product_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/lean_frontend/exploration.py",
        "import os\n\nenv = dict(os.environ)\n",
    )
    report = check_architecture(tmp_path)
    env = [v for v in report.violations if v.code == "environ-spreading"]
    assert len(env) == 1
    assert env[0].path == "src/jacobian/lean_frontend/exploration.py"


def test_os_environ_copy_in_product_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/runtime/services.py",
        "import os\n\nenv = os.environ.copy()\n",
    )
    report = check_architecture(tmp_path)
    env = [v for v in report.violations if v.code == "environ-spreading"]
    assert len(env) == 1


def test_starstar_os_environ_in_product_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/worker.py",
        "import os\n\nenv = {**os.environ, 'FOO': '1'}\n",
    )
    report = check_architecture(tmp_path)
    env = [v for v in report.violations if v.code == "environ-spreading"]
    assert len(env) == 1


def test_os_environ_get_in_product_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/config.py",
        "import os\n\nhome = os.environ.get('HOME', '/tmp')\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "environ-spreading" for v in report.violations)


def test_environ_spreading_in_tests_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/boundary/process/test_env.py",
        "import os\n\nenv = dict(os.environ)\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "environ-spreading" for v in report.violations)
