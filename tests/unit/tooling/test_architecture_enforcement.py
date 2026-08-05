"""Focused tests for the product architecture enforcement checker."""

from __future__ import annotations

import shutil
from pathlib import Path

from tools.check_architecture import (
    ArchitecturePolicyError,
    assert_architecture,
    check_architecture,
)

_ROOT = Path(__file__).resolve().parents[3]
_REAL_TASK = (
    _ROOT / "benchmarks/datasets/agent-workflow-v1/finite-field-irreducibility-repair"
)

# Fragments for constructing subprocess tokens without self-triggering.
_sub = "sub"
_proc = "proc" + "ess"
_sp = _sub + _proc


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Check 1: subprocess + os.exec* confinement
# ---------------------------------------------------------------------------


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


def test_subprocess_in_checkers_is_flagged(tmp_path: Path) -> None:
    """Mathematical checkers must use the product gateway, not direct subprocess."""
    _write(
        tmp_path,
        "src/jacobian_checkers/sat.py",
        "import subprocess\n\nsubprocess.run(['cadical'])\n",
    )

    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "src/jacobian_checkers/sat.py"


def test_subprocess_in_test_topology_is_flagged(tmp_path: Path) -> None:
    """Repository validation tooling is not an allowed subprocess fixture."""
    _write(
        tmp_path,
        "tools/test_topology.py",
        "import subprocess\n\nsubprocess.run(['echo'])\n",
    )

    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "tools/test_topology.py"


def test_subprocess_in_explicit_e2e_fixture_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/e2e/verified_results/test_reference_runtime.py",
        "import subprocess\n\nsubprocess.run(['echo'])\n",
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_explicit_boundary_fixture_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/boundary/process/test_bounded_process.py",
        "import subprocess\n\nsubprocess.run(['echo'])\n",
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


def test_subprocess_in_unlisted_boundary_test_is_flagged(tmp_path: Path) -> None:
    """Broad boundary directories are not allowlisted — only exact fixture files."""
    _write(
        tmp_path,
        "tests/boundary/process/test_random_new.py",
        "import subprocess\n",
    )

    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert sub[0].path == "tests/boundary/process/test_random_new.py"


def test_subprocess_in_unlisted_component_test_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/component/test_random.py",
        "import subprocess\n",
    )

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


def test_os_execvpe_in_test_topology_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/test_topology.py",
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
    """String constants embedding subprocess API calls bypass the import gate."""
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
    _write(
        tmp_path,
        "src/jacobian/worker.py",
        f"code = \"{_sp}.Popen(['echo'])\"\n",
    )

    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1
    assert "Popen" in sub[0].message


def test_embedded_subprocess_import_in_string_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/runner.py",
        f'_code = "import {_sp}\\n"\n',
    )

    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert len(sub) == 1


def test_embedded_subprocess_in_allowed_file_is_not_flagged(tmp_path: Path) -> None:
    """Allowed files may contain subprocess strings (e.g. the engine itself)."""
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
    """Strings that don't contain subprocess API patterns must not trigger."""
    _write(
        tmp_path,
        "src/jacobian/config.py",
        'message = "the process is running"\n',
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "subprocess-confined" for v in report.violations)


# ---------------------------------------------------------------------------
# Check 2: run_bounded_process gateway confinement
# ---------------------------------------------------------------------------


def test_run_bounded_process_in_product_caller_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/verification/service.py",
        "from jacobian.bounded_process import run_bounded_process\n"
        "run_bounded_process(['echo'])\n",
    )

    report = check_architecture(tmp_path)
    gateway = [v for v in report.violations if v.code == "bounded-process-gateway"]
    assert len(gateway) >= 1
    assert gateway[0].path == "src/jacobian/verification/service.py"


def test_run_bounded_process_in_process_policy_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/process_policy.py",
        "from jacobian.bounded_process import run_bounded_process\n"
        "run_bounded_process(['echo'])\n",
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
    """Tests may monkeypatch run_bounded_process; only product src is gated."""
    _write(
        tmp_path,
        "tests/boundary/process/test_foo.py",
        "from jacobian.bounded_process import run_bounded_process\n",
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "bounded-process-gateway" for v in report.violations)


# ---------------------------------------------------------------------------
# Check 3: shutil.which resolver confinement
# ---------------------------------------------------------------------------


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
        tmp_path,
        "src/jacobian/utils.py",
        "import shutil\n\nshutil.copy('a', 'b')\n",
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "shutil-which-resolver" for v in report.violations)


# ---------------------------------------------------------------------------
# Check 4: os.environ spreading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Check 5: public-contract drift (missing contract is a violation)
# ---------------------------------------------------------------------------


def _copy_real_task(root: Path, task_name: str) -> None:
    """Copy a real agent-workflow-v1 task into the test tree."""
    dest = root / "benchmarks/datasets/agent-workflow-v1" / task_name
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_REAL_TASK, dest)


def test_public_contract_drift_is_flagged(tmp_path: Path) -> None:
    _copy_real_task(tmp_path, "test-task")
    # Corrupt the submission schema to introduce drift.
    schema_path = (
        tmp_path
        / "benchmarks/datasets/agent-workflow-v1/test-task/environment/submission_schema.json"
    )
    import json

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["required"] = ["task_id"]  # Remove a required field.
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert len(drift) >= 1
    assert "test-task" in drift[0].path


def test_public_contract_no_drift_passes(tmp_path: Path) -> None:
    _copy_real_task(tmp_path, "test-task")

    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert drift == []


def test_missing_public_contract_is_a_violation(tmp_path: Path) -> None:
    task_dir = (
        tmp_path / "benchmarks/datasets/agent-workflow-v1" / "missing-contract-task"
    )
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir(parents=True)
    # Mark as a canonical task directory.
    (task_dir / "task.toml").write_text("", encoding="utf-8")

    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert len(drift) == 1
    assert "missing" in drift[0].message.lower()
    assert "missing-contract-task" in drift[0].path


def test_non_task_directory_does_not_require_contract(tmp_path: Path) -> None:
    """Directories without task.toml (e.g. jobs/, members/) are not tasks."""
    meta_dir = tmp_path / "benchmarks/datasets/agent-workflow-v1" / "metadata"
    (meta_dir / "tests").mkdir(parents=True)

    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert drift == []


# ---------------------------------------------------------------------------
# Check 6: unsupported surfaces (Python AST + text scan)
# ---------------------------------------------------------------------------

# Tokens are built from fragments to avoid self-triggering.
_K = "knowledge"
_S = "search"
_RM = "Research" + "Memory"
_RE = "Research" + "Episode"
_KS_DOT = _K + "." + _S
_KS_US = _K + "_" + _S


def test_research_memory_import_in_src_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/capability_service.py",
        f"from jacobian.memory import {_RM}\n",
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1


def test_research_memory_import_in_tests_is_flagged(tmp_path: Path) -> None:
    """Unsupported surfaces are scanned in tests too, not just src."""
    _write(
        tmp_path,
        "tests/composition/test_memory.py",
        f"from jacobian.memory import {_RM}\n",
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1
    assert surf[0].path == "tests/composition/test_memory.py"


def test_knowledge_search_string_in_src_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/builtin_capabilities.py",
        f'capability_id = "{_KS_DOT}"\n',
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) == 1
    assert _KS_DOT in surf[0].message


def test_knowledge_search_in_docs_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/search.md",
        f"# Search\n\nUse {_KS_DOT} for retrieval.\n",
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) == 1
    assert surf[0].path == "docs/explanation/search.md"


def test_knowledge_search_in_schema_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "benchmarks/schemas/test-schema.json",
        f'{{"capability_id": "{_KS_DOT}"}}\n',
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) == 1
    assert surf[0].path == "benchmarks/schemas/test-schema.json"


def test_research_episode_in_docs_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/episodes.md",
        f"# Episodes\n\nThe {_RE} store records past work.\n",
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1
    assert surf[0].path == "docs/explanation/episodes.md"


def test_research_memory_prose_in_docs_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/memory.md",
        "# Memory\n\nThe Research memory store is deprecated.\n",
    )

    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1


def test_generic_memory_word_is_not_flagged(tmp_path: Path) -> None:
    """Legitimate generic words like 'memory limits' must not trigger."""
    _write(
        tmp_path,
        "src/jacobian/process_policy.py",
        "import os\n\n# Configure memory limits for child processes.\n",
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "unsupported-surface" for v in report.violations)


def test_changelog_is_excluded_from_surface_scan(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "CHANGELOG.md",
        f"# Changelog\n\nAdded {_KS_DOT} capability.\n",
    )

    report = check_architecture(tmp_path)
    assert all(v.code != "unsupported-surface" for v in report.violations)


# ---------------------------------------------------------------------------
# Exclusion and orchestration
# ---------------------------------------------------------------------------


def test_wt438_directory_is_excluded(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "wt-438/src/jacobian/bad.py",
        f"import subprocess\nfrom jacobian.memory import {_RM}\n"
        "import shutil\nshutil.which('git')\n"
        "import os\nenv = dict(os.environ)\n",
    )

    report = check_architecture(tmp_path)
    assert report.ok
    assert report.violations == ()


def test_clean_tree_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/bounded_process.py",
        "import subprocess\n",
    )
    _write(
        tmp_path,
        "src/jacobian/process_policy.py",
        "from jacobian.bounded_process import run_bounded_process\n"
        "import shutil\nshutil.which('prlimit')\n",
    )
    _write(
        tmp_path,
        "src/jacobian/capability_service.py",
        "import os\nhome = os.environ.get('HOME')\n",
    )
    _write(
        tmp_path,
        "tests/boundary/process/test_bounded_process.py",
        "import subprocess\nsubprocess.run(['echo'])\n",
    )

    report = check_architecture(tmp_path)
    assert report.ok, report.render()


def test_report_render_shows_violations(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/bad.py", "import subprocess\n")

    report = check_architecture(tmp_path)
    rendered = report.render()
    assert "architecture:" in rendered
    assert "subprocess-confined" in rendered
    assert "src/jacobian/bad.py" in rendered


def test_assert_raises_on_failure(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/bad.py", "import subprocess\n")

    try:
        assert_architecture(tmp_path)
    except ArchitecturePolicyError as exc:
        assert "subprocess-confined" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected architecture policy failure")


def test_multiple_violations_sorted_by_path(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/zzz.py", "import subprocess\n")
    _write(tmp_path, "src/jacobian/aaa.py", "import subprocess\n")

    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert sub[0].path < sub[1].path
