from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str) -> ModuleType:
    path = ROOT / ".github" / "scripts" / name
    module_name = name.replace("-", "_")
    loader = SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


def _ci_plan(*args: str) -> dict[str, str]:
    classifier = _load_script("classify-ci-paths")
    output = io.StringIO()
    with pytest.MonkeyPatch.context() as process_state:
        process_state.setattr(sys, "argv", ["classify-ci-paths", *args])
        with contextlib.redirect_stdout(output):
            classifier.main()
    return dict(line.split("=", 1) for line in output.getvalue().splitlines())


def test_explicit_paths_route_deployment_without_all_suite_fallback() -> None:
    plan = _ci_plan("--paths", "deploy/install.sh")

    assert plan["run-process"] == "true"
    assert plan["run-unit"] == "false"
    assert plan["run-provider"] == "false"


def test_provider_selection_is_additive_but_default_provider_stays_deferred() -> None:
    path = "tests/boundary/providers/sympy/test_sympy.py"
    default = _ci_plan("--paths", path)
    opted_in = _ci_plan("--paths", path, "--provider", "sympy")

    assert default["run-provider"] == "false"
    assert json.loads(default["provider-selection"]) == []
    assert opted_in["run-provider"] == "true"
    assert json.loads(opted_in["provider-selection"]) == ["sympy"]


def test_paths_environment_supports_local_planning_without_a_git_base(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script("plan-local-tests")
    monkeypatch.setenv("PATHS", '["deploy/install.sh"]')
    monkeypatch.setattr(sys, "argv", ["plan-local-tests"])

    planner.main()
    output = capsys.readouterr().out

    assert "base: (explicit paths)" in output
    assert "make deploy-check" in output
    assert "make test-process" in output


def test_planners_accept_a_file_backed_path_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("tools/check_doc_commands.py\n", encoding="utf-8")

    classifier = _load_script("classify-ci-paths")
    output = io.StringIO()
    monkeypatch.setattr(
        sys,
        "argv",
        ["classify-ci-paths", "--paths-file", str(paths_file)],
    )
    with contextlib.redirect_stdout(output):
        classifier.main()

    plan = dict(line.split("=", 1) for line in output.getvalue().splitlines())
    assert plan["run-docs"] == "true"
    assert plan["run-component"] == "false"


def test_benchmark_path_input_is_forwarded_without_running_git(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script("plan-benchmarks")
    calls: dict[str, object] = {}

    def fake_plan(paths: list[str], **kwargs: object) -> dict[str, str]:
        calls["paths"] = paths
        calls.update(kwargs)
        return {"paths": json.dumps(paths)}

    monkeypatch.setattr(planner, "plan", fake_plan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan-benchmarks",
            "--event",
            "pull_request",
            "--paths",
            "benchmarks/README.md",
        ],
    )

    assert planner.main() == 0
    assert calls["paths"] == ["benchmarks/README.md"]
    assert "benchmarks/README.md" in capsys.readouterr().out


def test_benchmark_documentation_does_not_select_oracle_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script("plan-benchmarks")
    monkeypatch.setattr(planner, "_topology_digest", lambda _suites: "digest")

    plan = planner.plan(
        ["benchmarks/README.md", "benchmarks/validation/README.md"],
        event="pull_request",
    )

    assert plan["run-benchmark-oracle"] == "false"
    assert json.loads(plan["benchmark-oracle-matrix"]) == []


def test_ci_validator_accepts_catalog_bound_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _ci_plan("--paths", "deploy/install.sh")
    validator = _load_script("validate-ci-plan")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO("\n".join(f"{key}={value}" for key, value in plan.items()) + "\n"),
    )

    assert validator.main() is None


def _run_validator(plan: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> int:
    validator = _load_script("validate-ci-plan")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO("\n".join(f"{key}={value}" for key, value in plan.items()) + "\n"),
    )
    try:
        validator.main()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    return 0


@pytest.mark.parametrize(
    ("base_paths", "mutation"),
    [
        # run-deploy must stay false for isolated classifications; the new
        # deploy key is otherwise masked by the missing-key rejection in the
        # older malformed-plan suite.
        (("lean/JacobianLeanRuntime.lean",), {"run-deploy": "true"}),
        (("README.md",), {"run-deploy": "true"}),
        # provider-selection is authorization: an unknown name or a selection
        # without the provider lane must be rejected.
        (
            ("tests/boundary/providers/sympy/test_sympy.py", "--provider", "sympy"),
            {"provider-selection": '["bogus"]'},
        ),
        (("README.md",), {"provider-selection": '["sympy"]'}),
    ],
)
def test_ci_validator_rejects_new_deploy_and_provider_incoherence(
    base_paths: tuple[str, ...],
    mutation: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {**_ci_plan("--paths", *base_paths), **mutation}

    assert _run_validator(plan, monkeypatch) != 0


def test_ci_validator_rejects_plan_missing_run_deploy_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = dict(_ci_plan("--paths", "deploy/install.sh"))
    del plan["run-deploy"]

    assert _run_validator(plan, monkeypatch) != 0


def test_shared_path_policy_rejects_non_repository_relative_paths() -> None:
    helper = _load_script("_ci_paths.py")

    for bad in ("", "/etc/passwd", "../escape", "ok/../escape"):
        with pytest.raises(ValueError):
            helper.normalize_paths([bad])


def test_shared_path_policy_deduplicates_normalized_paths() -> None:
    helper = _load_script("_ci_paths.py")

    assert helper.normalize_paths(["./README.md", "README.md", "./docs/README.md"]) == [
        "README.md",
        "docs/README.md",
    ]


def test_classifier_and_planner_share_one_path_normalization_policy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Absolute and parent-traversal paths must be rejected at the CLI boundary
    # of both control planes, proving they share _ci_paths rather than drift.
    for bad in ("/etc/passwd", "../escape"):
        for script_name in ("classify-ci-paths", "plan-local-tests"):
            script = _load_script(script_name)
            monkeypatch.setattr(sys, "argv", [script_name, "--paths", bad])
            with pytest.raises(SystemExit) as exc_info:
                script.main()
            assert exc_info.value.code != 0
            assert "repository-relative" in capsys.readouterr().err


def test_plan_local_tests_omits_deploy_gate_without_deployment_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script("plan-local-tests")
    monkeypatch.setenv("PATHS", '["README.md"]')
    monkeypatch.setattr(sys, "argv", ["plan-local-tests"])

    planner.main()
    output = capsys.readouterr().out

    assert "make deploy-check" not in output
    assert "make docs-linkcheck" in output


def test_plan_local_tests_labels_documentation_as_non_pytest_work(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script("plan-local-tests")
    monkeypatch.setenv("PATHS", '["README.md"]')
    monkeypatch.setattr(sys, "argv", ["plan-local-tests"])

    planner.main()
    output = capsys.readouterr().out

    assert "not applicable (no pytest lane selected)" in output
    assert "suite fallback" not in output


def test_plan_local_tests_omits_commands_subsumed_by_another_local_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planner = _load_script("plan-local-tests")
    monkeypatch.setenv("PATHS", '["Makefile"]')
    monkeypatch.setattr(sys, "argv", ["plan-local-tests"])

    planner.main()
    output = capsys.readouterr().out

    assert "make check-static" in output
    assert "    make build\n" not in output


def _write_validator_tree(tmp_path: Path) -> None:
    """Materialize the minimal tree validate-ci-plan reads from ROOT."""

    (tmp_path / ".github").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    manifest = json.loads((ROOT / ".github" / "ci-impact.json").read_text("utf-8"))
    (tmp_path / ".github" / "ci-impact.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (tmp_path / "tests" / "topology.toml").write_text(
        (ROOT / "tests" / "topology.toml").read_text("utf-8"), encoding="utf-8"
    )
    (tmp_path / "Makefile").write_text(
        (ROOT / "Makefile").read_text("utf-8"), encoding="utf-8"
    )


def test_ci_validator_enforces_local_only_deploy_catalog_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_validator_tree(tmp_path)
    manifest_path = tmp_path / ".github" / "ci-impact.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    del manifest["catalog"]["deploy"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validator = _load_script("validate-ci-plan")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    with pytest.raises(SystemExit):
        validator.main()


def test_ci_validator_rejects_cyclic_local_command_subsumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_validator_tree(tmp_path)
    manifest_path = tmp_path / ".github" / "ci-impact.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["catalog"]["build"]["local_subsumes"] = ["static"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validator = _load_script("validate-ci-plan")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    with pytest.raises(SystemExit, match="local_subsumes must be acyclic"):
        validator.main()
