from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[4]
SCRIPTS = ROOT / ".github" / "scripts"


def _load(name: str, filename: str) -> ModuleType:
    loader = SourceFileLoader(name, str(SCRIPTS / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_combines_commit_worktree_staged_and_untracked_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _load("plan_local_tests", "plan-local-tests")
    responses = {
        ("diff", "--name-status", "base..HEAD"): [
            planner.Change("M", "committed.py"),
            planner.Change("M", "same.py"),
        ],
        ("diff", "--name-status"): [
            planner.Change("M", "unstaged.py"),
            planner.Change("M", "same.py"),
        ],
        ("diff", "--cached", "--name-status"): [planner.Change("A", "staged.py")],
    }
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    monkeypatch.setattr(planner, "git_changes", lambda *args: responses[args])
    monkeypatch.setattr(
        planner,
        "git_paths",
        lambda *args: ["untracked.py"],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    assert [
        (change.status, change.path) for change in planner.changed_entries("base")
    ] == [
        ("M", "committed.py"),
        ("M", "same.py"),
        ("A", "staged.py"),
        ("M", "unstaged.py"),
        ("?", "untracked.py"),
    ]


def test_clean_test_plan_selects_no_lanes() -> None:
    planner = _load("plan_local_tests_clean", "plan-local-tests")

    assert planner.classify([]) == {"classification": "clean"}


def test_known_ci_tooling_change_uses_owned_process_tests() -> None:
    planner = _load("plan_local_tests_ci_override", "plan-local-tests")

    selected, fallback = planner.exact_tests(
        [planner.Change("M", ".github/scripts/emit-plan-receipt")]
    )

    assert fallback is None
    assert selected == ["tests/boundary/process/tooling/test_plan_receipt.py"]


def test_deleted_ci_tooling_change_cannot_use_owned_process_override() -> None:
    planner = _load("plan_local_tests_deleted_ci_override", "plan-local-tests")

    selected, fallback = planner.exact_tests(
        [planner.Change("D", ".github/scripts/emit-plan-receipt")]
    )

    assert selected == []
    assert fallback == ".github/scripts/emit-plan-receipt: D changes are not exact"


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


def test_posix_topology_runner_replaces_itself_with_pytest(monkeypatch) -> None:
    if os.name == "nt":
        pytest.skip("POSIX exec is not used on Windows")
    from tools import test_topology

    topology = test_topology.load_topology()
    observed: dict[str, object] = {}

    def stop_after_exec(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        observed.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(os, "execvpe", stop_after_exec)
    monkeypatch.delenv("JACOBIAN_TEST_LANE", raising=False)

    with pytest.raises(RuntimeError, match="exec intercepted"):
        test_topology.run_lane(
            topology,
            "unit",
            ["tests/unit/tooling/test_fixture_architecture.py"],
            ["-q"],
        )

    assert observed["executable"] == sys.executable
    assert observed["arguments"] == [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/tooling/test_fixture_architecture.py",
        "-q",
        "--timeout",
        "10",
    ]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["JACOBIAN_TEST_LANE"] == "unit"


@pytest.mark.parametrize("surface", ["help", "version"])
def test_cheap_cli_surfaces_do_not_import_runtime_or_math_backends(
    surface: str,
) -> None:
    probe = """
import json
import sys

if sys.argv[1] == "help":
    from jacobian.cli import app
    app(args=["--help"], prog_name="jacobian", standalone_mode=False)
else:
    import jacobian
    assert jacobian.__version__

forbidden = (
    "jacobian.adapters.mcp.server",
    "jacobian.domains",
    "jacobian.lean_frontend.service",
    "sympy",
    "cvc5",
    "flint",
    "z3",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print("JACOBIAN_IMPORT_SURFACE=" + json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, surface],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("JACOBIAN_IMPORT_SURFACE=")
    )

    assert json.loads(marker.partition("=")[2]) == []


def test_plan_preserves_both_sides_of_rename(monkeypatch) -> None:
    planner = _load("plan_local_tests_rename", "plan-local-tests")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="R100\tsrc/jacobian/old.py\tsrc/jacobian/new.py\n",
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    assert planner.git_changes("diff", "--name-status") == [
        planner.Change("R", "src/jacobian/old.py"),
        planner.Change("R", "src/jacobian/new.py"),
    ]


def _planner_tree(tmp_path: Path, monkeypatch) -> ModuleType:
    planner = _load(f"plan_local_tests_{tmp_path.name}", "plan-local-tests")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    monkeypatch.setattr(
        planner,
        "OWNERSHIP",
        tmp_path / ".github" / "local-test-ownership.json",
    )
    return planner


def test_plan_selects_tests_that_directly_import_changed_leaf_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    source = tmp_path / "src/jacobian/leaf.py"
    direct = tmp_path / "tests/unit/test_leaf.py"
    unrelated = tmp_path / "tests/unit/test_other.py"
    source.parent.mkdir(parents=True)
    direct.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    direct.write_text("from jacobian.leaf import VALUE\n", encoding="utf-8")
    unrelated.write_text("import jacobian.other\n", encoding="utf-8")

    tests, fallback = planner.exact_tests([planner.Change("M", "src/jacobian/leaf.py")])

    assert tests == ["tests/unit/test_leaf.py"]
    assert fallback is None


def test_plan_falls_back_when_source_dependency_makes_impact_transitive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    leaf = tmp_path / "src/jacobian/leaf.py"
    consumer = tmp_path / "src/jacobian/consumer.py"
    test = tmp_path / "tests/unit/test_consumer.py"
    leaf.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    leaf.write_text("VALUE = 1\n", encoding="utf-8")
    consumer.write_text("from jacobian.leaf import VALUE\n", encoding="utf-8")
    test.write_text("import jacobian.consumer\n", encoding="utf-8")

    tests, fallback = planner.exact_tests([planner.Change("M", "src/jacobian/leaf.py")])

    assert tests == []
    assert fallback is not None
    assert "transitive impact is ambiguous" in fallback


def test_plan_treats_package_prefix_import_as_transitive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    package = tmp_path / "src/jacobian/domain/__init__.py"
    consumer = tmp_path / "src/jacobian/consumer.py"
    test = tmp_path / "tests/unit/test_consumer.py"
    package.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    package.write_text("VALUE = 1\n", encoding="utf-8")
    consumer.write_text("import jacobian.domain.child\n", encoding="utf-8")
    test.write_text("import jacobian.consumer\n", encoding="utf-8")

    tests, fallback = planner.exact_tests(
        [planner.Change("M", "src/jacobian/domain/__init__.py")]
    )

    assert tests == []
    assert fallback is not None
    assert "transitive impact is ambiguous" in fallback


def test_plan_resolves_relative_source_imports_as_transitive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    leaf = tmp_path / "src/jacobian/domain/leaf.py"
    consumer = tmp_path / "src/jacobian/domain/consumer.py"
    test = tmp_path / "tests/unit/test_consumer.py"
    leaf.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    leaf.write_text("VALUE = 1\n", encoding="utf-8")
    consumer.write_text("from .leaf import VALUE\n", encoding="utf-8")
    test.write_text("import jacobian.domain.consumer\n", encoding="utf-8")

    tests, fallback = planner.exact_tests(
        [planner.Change("M", "src/jacobian/domain/leaf.py")]
    )

    assert tests == []
    assert fallback is not None
    assert "transitive impact is ambiguous" in fallback


def test_plan_falls_back_for_delete_rename_and_untracked_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)

    for status in ("D", "R", "?"):
        tests, fallback = planner.exact_tests(
            [planner.Change(status, "src/jacobian/leaf.py")]
        )
        assert tests == []
        assert fallback is not None


def test_plan_falls_back_for_changed_test_support_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    helper = tmp_path / "tests/support/domains.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    tests, fallback = planner.exact_tests(
        [planner.Change("M", "tests/support/domains.py")]
    )

    assert tests == []
    assert fallback == ("tests/support/domains.py: test support code has broad impact")


def test_plan_uses_explicit_ownership_override_for_exact_nodes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    manifest = tmp_path / ".github/local-test-ownership.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"version": 1, "overrides": {'
        '"generated/schema.json": ["tests/unit/contracts/test_schema.py::test_public_schema"]'
        "}}",
        encoding="utf-8",
    )
    test = tmp_path / "tests/unit/contracts/test_schema.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_public_schema(): pass\n", encoding="utf-8")

    tests, fallback = planner.exact_tests(
        [planner.Change("M", "generated/schema.json")]
    )

    assert tests == ["tests/unit/contracts/test_schema.py::test_public_schema"]
    assert fallback is None


def test_plan_routes_lean_selectors_to_serial_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    ordinary = tmp_path / "tests/unit/test_leaf.py"
    lean = tmp_path / "tests/boundary/providers/lean/test_lean_leaf.py"
    ordinary.parent.mkdir(parents=True)
    lean.parent.mkdir(parents=True)
    (tmp_path / "tests" / "topology.toml").write_text(
        "version = 1\n\n"
        "[[lanes]]\nname = 'unit'\npaths = ['tests/unit']\n\n"
        "[[lanes]]\nname = 'lean'\npaths = ['tests/boundary/providers/lean']\n",
        encoding="utf-8",
    )
    ordinary.write_text("def test_leaf(): pass\n", encoding="utf-8")
    lean.write_text(
        "def test_lean_leaf(): pass\n",
        encoding="utf-8",
    )

    assert planner.focused_commands(
        [
            "tests/unit/test_leaf.py",
            "tests/boundary/providers/lean/test_lean_leaf.py::test_lean_leaf",
        ]
    ) == [
        "make test-unit TESTS=tests/unit/test_leaf.py",
        "make test-lean "
        "TESTS=tests/boundary/providers/lean/test_lean_leaf.py::test_lean_leaf",
    ]


def test_plan_routes_provider_selectors_to_the_provider_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    provider = tmp_path / "tests/boundary/providers/cvc5/test_provider.py"
    provider.parent.mkdir(parents=True)
    (tmp_path / "tests" / "topology.toml").write_text(
        "version = 1\n\n"
        "[[lanes]]\nname = 'provider'\n"
        "paths = ['tests/boundary/providers/cvc5/**']\n",
        encoding="utf-8",
    )
    provider.write_text("def test_provider(): pass\n", encoding="utf-8")

    assert planner.focused_commands(
        ["tests/boundary/providers/cvc5/test_provider.py::test_provider"]
    ) == [
        "make test-provider "
        "TESTS=tests/boundary/providers/cvc5/test_provider.py::test_provider"
    ]


def test_plan_keeps_a_changed_test_as_an_exact_node_selector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    test = tmp_path / "tests/unit/test_leaf.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_leaf(): pass\n", encoding="utf-8")

    selected, fallback = planner.exact_tests(
        [planner.Change("M", "tests/unit/test_leaf.py")]
    )

    assert selected == ["tests/unit/test_leaf.py"]
    assert fallback is None


def test_plan_resolves_relative_imports_from_package_initializers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    package = tmp_path / "src/jacobian/pkg"
    leaf = package / "leaf.py"
    initializer = package / "__init__.py"
    direct = tmp_path / "tests/unit/test_leaf.py"
    package_user = tmp_path / "tests/unit/test_package.py"
    package.mkdir(parents=True)
    direct.parent.mkdir(parents=True)
    leaf.write_text("VALUE = 1\n", encoding="utf-8")
    initializer.write_text("from .leaf import VALUE\n", encoding="utf-8")
    direct.write_text("from jacobian.pkg.leaf import VALUE\n", encoding="utf-8")
    package_user.write_text("from jacobian.pkg import VALUE\n", encoding="utf-8")

    selected, fallback = planner.exact_tests(
        [planner.Change("M", "src/jacobian/pkg/leaf.py")]
    )

    assert selected == []
    assert fallback == (
        "src/jacobian/pkg/leaf.py: imported by src/jacobian/pkg/__init__.py; "
        "transitive impact is ambiguous"
    )


def test_plan_fails_closed_for_unknown_path_and_invalid_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    manifest = tmp_path / ".github/local-test-ownership.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{broken", encoding="utf-8")

    tests, fallback = planner.exact_tests([planner.Change("M", "mystery.asset")])

    assert tests == []
    assert fallback == ".github/local-test-ownership.json: invalid manifest"
