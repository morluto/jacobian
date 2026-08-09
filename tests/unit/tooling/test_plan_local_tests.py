from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / ".github" / "scripts"


def _load_script_module(name: str, filename: str) -> ModuleType:
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
    planner = _load_script_module("plan_local_tests", "plan-local-tests")
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
    monkeypatch.setattr(planner.subprocess, "run", lambda *args, **kwargs: None)

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
    planner = _load_script_module("plan_local_tests_clean", "plan-local-tests")

    assert planner.classify([]) == {"classification": "clean"}


def test_known_ci_tooling_change_uses_owned_process_tests() -> None:
    planner = _load_script_module("plan_local_tests_ci_override", "plan-local-tests")

    selected, fallback = planner.exact_tests(
        [planner.Change("M", ".github/scripts/emit-plan-receipt")]
    )

    assert fallback is None
    assert selected == ["tests/boundary/process/tooling/test_plan_receipt.py"]


def test_explicit_paths_report_missing_files_as_deletes(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    planner = _load_script_module("plan_local_tests_explicit_paths", "plan-local-tests")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    monkeypatch.setattr(
        planner,
        "classify",
        lambda paths: {"classification": "selective", "run-unit": "true"},
    )
    present = tmp_path / "tests" / "unit" / "test_present.py"
    present.parent.mkdir(parents=True)
    present.write_text("def test_present(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["plan-local-tests", "--paths", "tests/unit/test_present.py", "removed.py"],
    )

    planner.main()
    output = capsys.readouterr().out

    assert "present.py" in output
    assert "removed.py" in output
    assert "removed.py: D changes are not exact" in output


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            ".github/scripts/plan-local-tests",
            {
                "tests/boundary/process/tooling/test_cli_import_surface.py",
                "tests/boundary/process/tooling/test_topology_runner.py",
                "tests/unit/tooling/test_plan_local_tests.py",
                "tests/unit/tooling/test_ci_planner_catalog.py",
            },
        ),
        (
            "tools/test_topology.py",
            {
                "tests/boundary/process/tooling/test_topology_runner.py",
                "tests/unit/tooling/test_topology_manifest.py",
            },
        ),
        (
            "tools/check_doc_commands.py",
            {"tests/unit/tooling/test_doc_commands.py"},
        ),
        (
            "tools/development_profiles.py",
            {
                "tests/unit/tooling/test_development_profiles.py",
                "tests/boundary/process/tooling/test_source_agent_bootstrap.py",
            },
        ),
    ],
)
def test_new_tooling_changes_use_narrow_owned_tests(
    path: str,
    expected: set[str],
) -> None:
    planner = _load_script_module("plan_local_tests_owned_tools", "plan-local-tests")

    selected, fallback = planner.exact_tests([planner.Change("M", path)])

    assert fallback is None
    assert set(selected) == expected


def test_deleted_ci_tooling_change_cannot_use_owned_process_override() -> None:
    planner = _load_script_module(
        "plan_local_tests_deleted_ci_override", "plan-local-tests"
    )

    selected, fallback = planner.exact_tests(
        [planner.Change("D", ".github/scripts/emit-plan-receipt")]
    )

    assert selected == []
    assert fallback == ".github/scripts/emit-plan-receipt: D changes are not exact"


def test_documentation_does_not_erase_exact_tooling_ownership() -> None:
    planner = _load_script_module("plan_local_tests_docs_mixed", "plan-local-tests")

    selected, fallback = planner.exact_tests(
        [
            planner.Change("M", "CONTRIBUTING.md"),
            planner.Change("M", "tools/development_profiles.py"),
        ]
    )

    assert fallback is None
    assert selected == [
        "tests/boundary/process/tooling/test_source_agent_bootstrap.py",
        "tests/unit/tooling/test_development_profiles.py",
    ]


def test_plan_preserves_both_sides_of_rename(monkeypatch) -> None:
    planner = _load_script_module("plan_local_tests_rename", "plan-local-tests")
    completed = planner.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="R100\tsrc/jacobian/old.py\tsrc/jacobian/new.py\n",
        stderr="",
    )
    monkeypatch.setattr(planner.subprocess, "run", lambda *args, **kwargs: completed)

    assert planner.git_changes("diff", "--name-status") == [
        planner.Change("R", "src/jacobian/old.py"),
        planner.Change("R", "src/jacobian/new.py"),
    ]


def _planner_tree(tmp_path: Path, monkeypatch) -> ModuleType:
    planner = _load_script_module(
        f"plan_local_tests_{tmp_path.name}", "plan-local-tests"
    )
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


def test_plan_rejects_non_exact_ownership_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner = _planner_tree(tmp_path, monkeypatch)
    manifest = tmp_path / ".github/local-test-ownership.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"version": 1, "overrides": {'
        '"generated/schema.json": ["tests/unit/contracts/test_schema.py"]'
        "}}",
        encoding="utf-8",
    )
    test = tmp_path / "tests/unit/contracts/test_schema.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_public_schema(): pass\n", encoding="utf-8")

    tests, fallback = planner.exact_tests(
        [planner.Change("D", "generated/schema.json")]
    )

    assert tests == []
    assert fallback == "generated/schema.json: D changes are not exact"


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


def test_plan_focused_tests_keep_independent_gates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script_module("plan_local_tests_focused_output", "plan-local-tests")
    monkeypatch.setenv("PATHS", '["tools/check_doc_commands.py"]')
    monkeypatch.setattr(sys, "argv", ["plan-local-tests"])

    planner.main()
    output = capsys.readouterr().out

    # Focused selectors are still present...
    assert "make test-unit TESTS=tests/unit/tooling/test_doc_commands.py" in output
    # ...but focused selection no longer suppresses the independent gates that
    # the classification selected for this infrastructure tool path.
    assert "make check-static" in output
    assert "make docs-linkcheck" in output
    # Broad pytest lane fallbacks are not printed for a focused selection.
    assert "make test-process" not in output
    assert "make test-component" not in output


def _plan(**selected: bool) -> dict[str, str]:
    plan = {
        "classification": "selective",
        "run-python": "false",
        "run-deploy": "false",
    }
    plan.update(
        {
            f"run-{name.replace('_', '-')}": str(value).lower()
            for name, value in selected.items()
        }
    )
    return plan


def test_planned_commands_selects_docs_linkcheck_for_docs_paths() -> None:
    planner = _load_script_module("plan_local_tests_docs_gate", "plan-local-tests")

    commands = planner.planned_commands(
        [planner.Change("M", "docs/explanation/goals.md")],
        _plan(docs=True),
        [],
        None,
        False,
    )

    assert commands == ["make docs-linkcheck"]


def test_planned_commands_selects_npm_test_for_npm_paths() -> None:
    planner = _load_script_module("plan_local_tests_npm_gate", "plan-local-tests")

    commands = planner.planned_commands(
        [planner.Change("M", "npm/index.js")],
        _plan(npm=True),
        [],
        None,
        False,
    )

    assert commands == ["make npm-test"]


def test_planned_commands_uses_check_static_for_makefile() -> None:
    planner = _load_script_module("plan_local_tests_makefile_gate", "plan-local-tests")

    commands = planner.planned_commands(
        [planner.Change("M", "Makefile")],
        _plan(static=True),
        [],
        None,
        False,
    )

    assert commands == ["make check-static"]
    assert "make lint typecheck" not in commands


def test_planned_commands_uses_lint_typecheck_once_for_ordinary_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script_module("plan_local_tests_lint_gate", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "focused_commands",
        lambda tests: ["make test-unit TESTS=tests/unit/test_leaf.py"],
    )

    commands = planner.planned_commands(
        [planner.Change("M", "src/jacobian/leaf.py")],
        _plan(unit=True, component=True, static=True, python=True),
        ["tests/unit/test_leaf.py"],
        None,
        False,
    )

    # Ordinary Python routes to the light lint/typecheck handoff once, even
    # though the classifier also selected the heavier static lane.
    assert commands[0] == "make lint typecheck"
    assert commands.count("make lint typecheck") == 1
    assert "make check-static" not in commands
    # Focused selectors are preserved alongside the gate.
    assert "make test-unit TESTS=tests/unit/test_leaf.py" in commands


def test_planned_commands_uses_check_static_for_infrastructure_with_focused_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script_module("plan_local_tests_ci_tool_gate", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "focused_commands",
        lambda tests: ["make test-unit TESTS=tests/unit/test_plan_local_tests.py"],
    )

    commands = planner.planned_commands(
        [planner.Change("M", ".github/scripts/plan-local-tests")],
        _plan(
            unit=True,
            process=True,
            static=True,
            npm=True,
            security=True,
            duplicate=True,
            python=True,
        ),
        ["tests/unit/tooling/test_plan_local_tests.py"],
        None,
        False,
    )

    # Infrastructure routes to check-static (which already covers lint and
    # typecheck), so the separate lint/typecheck handoff is not added.
    assert commands[0] == "make check-static"
    assert "make lint typecheck" not in commands
    assert "make test-unit TESTS=tests/unit/test_plan_local_tests.py" in commands
    assert all("security" not in command for command in commands)
    assert all("duplicate" not in command for command in commands)
    assert all("npm-test" not in command for command in commands)


def test_planned_commands_mixed_infra_and_python_uses_check_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script_module("plan_local_tests_mixed", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "focused_commands",
        lambda tests: ["make test-unit TESTS=tests/unit/test_leaf.py"],
    )

    commands = planner.planned_commands(
        [
            planner.Change("M", "src/jacobian/leaf.py"),
            planner.Change("M", ".github/workflows/ci.yml"),
        ],
        _plan(unit=True, static=True, python=True),
        ["tests/unit/test_leaf.py"],
        None,
        False,
    )

    assert commands[0] == "make check-static"
    assert "make lint typecheck" not in commands
    assert "make test-unit TESTS=tests/unit/test_leaf.py" in commands


def test_planned_commands_adds_docs_gate_alongside_ordinary_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script_module("plan_local_tests_docs_py", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "focused_commands",
        lambda tests: ["make test-unit TESTS=tests/unit/test_leaf.py"],
    )

    commands = planner.planned_commands(
        [
            planner.Change("M", "docs/explanation/goals.md"),
            planner.Change("M", "src/jacobian/leaf.py"),
        ],
        _plan(unit=True, static=True, docs=True, python=True),
        ["tests/unit/test_leaf.py"],
        None,
        False,
    )

    assert "make lint typecheck" in commands
    assert "make docs-linkcheck" in commands
    assert "make check-static" not in commands
    assert "make test-unit TESTS=tests/unit/test_leaf.py" in commands


def test_planned_commands_appends_deploy_gate_last() -> None:
    planner = _load_script_module("plan_local_tests_deploy_gate", "plan-local-tests")

    commands = planner.planned_commands(
        [planner.Change("M", "deploy/install.sh")],
        _plan(process=True),
        [],
        "deploy/install.sh: no exact Python ownership evidence",
        True,
    )

    assert commands[-1] == "make deploy-check"
    assert "make test-process" in commands


def test_planned_commands_falls_back_to_pytest_lanes_with_lint_gate() -> None:
    planner = _load_script_module("plan_local_tests_fallback_lanes", "plan-local-tests")

    commands = planner.planned_commands(
        [planner.Change("D", "src/jacobian/leaf.py")],
        _plan(unit=True, component=True, static=True, python=True),
        [],
        "src/jacobian/leaf.py: D changes are not exact",
        False,
    )

    # Delete -> fallback. Ordinary Python with no focused tests still gets the
    # lint/typecheck handoff once, plus the selected pytest lanes (static is not
    # re-run as a separate gate because ordinary Python routes to lint/typecheck).
    assert commands[0] == "make lint typecheck"
    assert "make test-unit" in commands
    assert "make test-component" in commands
    assert "make check-static" not in commands


def test_planned_commands_clean_tree_selects_no_commands() -> None:
    planner = _load_script_module("plan_local_tests_clean_cmds", "plan-local-tests")

    commands = planner.planned_commands(
        [],
        {"classification": "clean", "run-python": "false", "run-deploy": "false"},
        [],
        None,
        False,
    )

    assert commands == []


def test_execute_commands_runs_all_and_summarizes_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script_module("plan_local_tests_exec_fail", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "_git_revision",
        lambda revision: "deadbeef" if revision == "HEAD" else "basebeef",
    )
    monkeypatch.setattr(planner, "_git_dirty", lambda: True)
    calls: list[list[str]] = []

    def fake_run(argv, cwd, check):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        returncode = 1 if "boom" in argv[-1] else 0
        return planner.subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(planner.subprocess, "run", fake_run)

    returncode = planner.execute_commands(
        ["make lint typecheck", "make boom"], base="origin/main"
    )
    output = capsys.readouterr().out

    assert returncode != 0
    # Every command runs even after a failure so the summary is complete.
    assert len(calls) == 2
    assert "summary:" in output
    assert "base: basebeef" in output
    assert "head: deadbeef" in output
    assert "dirty: true" in output
    assert "total: 2" in output
    assert "passed: 1" in output
    assert "failed: 1" in output
    assert "result: fail" in output
    assert "[pass]" in output
    assert "[fail]" in output
    assert "make lint typecheck" in output
    assert "make boom" in output


def test_execute_commands_returns_zero_when_all_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script_module("plan_local_tests_exec_pass", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "_git_revision",
        lambda revision: "abc123" if revision == "HEAD" else "base123",
    )
    monkeypatch.setattr(planner, "_git_dirty", lambda: False)
    monkeypatch.setattr(
        planner.subprocess,
        "run",
        lambda argv, cwd, check: planner.subprocess.CompletedProcess(argv, 0),
    )

    returncode = planner.execute_commands(
        ["make lint typecheck", "make docs-linkcheck"], base="origin/main"
    )
    output = capsys.readouterr().out

    assert returncode == 0
    assert "result: pass" in output
    assert "total: 2" in output
    assert "failed: 0" in output
    assert "passed: 2" in output


def test_execute_commands_handles_empty_command_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script_module("plan_local_tests_exec_empty", "plan-local-tests")
    monkeypatch.setattr(
        planner,
        "_git_revision",
        lambda revision: "abc123" if revision == "HEAD" else "base123",
    )
    monkeypatch.setattr(planner, "_git_dirty", lambda: False)

    returncode = planner.execute_commands([], base="origin/main")
    output = capsys.readouterr().out

    assert returncode == 0
    assert "total: 0" in output
    assert "result: pass" in output
    assert "passed: 0" in output
    assert "failed: 0" in output


def test_plan_discovery_preserves_deleted_renamed_and_untracked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script_module("plan_local_tests_discovery_all", "plan-local-tests")
    responses = {
        ("diff", "--name-status", "base..HEAD"): [planner.Change("D", "src/gone.py")],
        ("diff", "--name-status"): [
            planner.Change("R", "src/old.py"),
            planner.Change("R", "src/new.py"),
        ],
        ("diff", "--cached", "--name-status"): [],
    }
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    monkeypatch.setattr(planner, "git_changes", lambda *args: responses[args])
    monkeypatch.setattr(planner, "git_paths", lambda *args: ["untracked.py"])
    monkeypatch.setattr(planner.subprocess, "run", lambda *args, **kwargs: None)

    entries = [
        (change.status, change.path) for change in planner.changed_entries("base")
    ]

    assert ("D", "src/gone.py") in entries
    assert ("R", "src/old.py") in entries
    assert ("R", "src/new.py") in entries
    assert ("?", "untracked.py") in entries
