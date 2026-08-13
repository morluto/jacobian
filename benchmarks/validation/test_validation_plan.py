"""Tests for the importable benchmark host-validation planner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from benchmarks.tooling.validation_plan import host_validation_plan


def test_task_change_selects_only_leaf_and_filtered_generic(tmp_path: Path) -> None:
    task = "task-a"
    leaf = tmp_path / "benchmarks/validation/mathematical_benchmarks_v1/test_task_a.py"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("", encoding="utf-8")
    suite = SimpleNamespace(tasks=(SimpleNamespace(path=Path(task)),))

    plan = host_validation_plan(
        tmp_path,
        [f"benchmarks/datasets/mathematical-benchmarks-v1/{task}/tests/verifier.py"],
        {"mathematical-benchmarks-v1": suite},
    )

    assert [(entry.selector, entry.keyword) for entry in plan.entries] == [
        (
            "benchmarks/validation/mathematical_benchmarks_v1/"
            "test_generic_verifier_contracts.py",
            task,
        ),
        (
            "benchmarks/validation/mathematical_benchmarks_v1/test_task_a.py",
            "",
        ),
    ]
    assert all(reason.startswith("focused host validation:") for reason in plan.reasons)


def test_shared_harness_change_explains_full_validation(tmp_path: Path) -> None:
    support = tmp_path / "benchmarks/validation/mathematical_benchmarks_v1/support.py"
    support.parent.mkdir(parents=True)
    support.write_text("", encoding="utf-8")

    plan = host_validation_plan(
        tmp_path, [support.relative_to(tmp_path).as_posix()], {}
    )

    assert len(plan.entries) == 4
    assert plan.reasons == (
        "shared verifier harness requires full host validation: "
        "benchmarks/validation/mathematical_benchmarks_v1/support.py",
    )


def test_control_plane_change_selects_its_exact_contract_tests(tmp_path: Path) -> None:
    plan = host_validation_plan(
        tmp_path, ["benchmarks/tooling/benchmark_validation.py"], {}
    )

    assert [entry.selector for entry in plan.entries] == [
        "benchmarks/validation/test_benchmark_validation.py"
    ]
    assert plan.reasons == (
        "focused host validation: control-test_benchmark_validation",
    )


def test_workflow_change_selects_each_owned_contract_without_full_fallback(
    tmp_path: Path,
) -> None:
    plan = host_validation_plan(tmp_path, [".github/workflows/benchmarks.yml"], {})

    assert {entry.selector for entry in plan.entries} == {
        "benchmarks/validation/test_benchmark_planner.py",
        "benchmarks/validation/test_benchmark_validation.py",
    }
    assert len(plan.entries) == 2


def test_non_host_control_utility_has_an_explicit_empty_host_selection(
    tmp_path: Path,
) -> None:
    plan = host_validation_plan(tmp_path, [".github/scripts/manage-test-timings"], {})

    assert plan.entries == ()
    assert plan.reasons == ()


def test_shared_path_policy_remains_conservative(tmp_path: Path) -> None:
    plan = host_validation_plan(tmp_path, [".github/scripts/_ci_paths.py"], {})

    assert len(plan.entries) == 4
    assert "full host validation" in plan.reasons[0]


def test_unknown_tooling_still_falls_back_to_full_validation(tmp_path: Path) -> None:
    path = "benchmarks/tooling/new_shared_harness.py"
    plan = host_validation_plan(tmp_path, [path], {})

    assert len(plan.entries) == 4
    assert plan.reasons == (
        f"shared benchmark tooling requires full host validation: {path}",
    )


def test_root_makefile_change_relies_on_contract_gate_not_host_verifiers(
    tmp_path: Path,
) -> None:
    plan = host_validation_plan(tmp_path, ["Makefile"], {})

    assert plan.entries == ()
    assert plan.reasons == ()


def test_harbor_makefile_change_remains_fail_closed(tmp_path: Path) -> None:
    plan = host_validation_plan(tmp_path, ["make/harbor.mk"], {})

    assert len(plan.entries) == 4
    assert plan.reasons == (
        "shared verifier execution harness requires full host validation: "
        "make/harbor.mk",
    )


def test_focused_host_matrix_over_limit_escalates_to_full_suite(
    tmp_path: Path,
) -> None:
    """Portfolio-wide task support migrations must not exceed the matrix cap."""
    from benchmarks.tooling.validation_plan import HOST_VALIDATION_MAX_JOBS

    tasks = tuple(f"task-{index:03d}" for index in range(HOST_VALIDATION_MAX_JOBS + 1))
    suite = SimpleNamespace(
        tasks=tuple(SimpleNamespace(path=Path(task)) for task in tasks)
    )
    paths = [
        f"benchmarks/datasets/mathematical-benchmarks-v1/{task}/tests/verifier_support.py"
        for task in tasks
    ]

    plan = host_validation_plan(
        tmp_path,
        paths,
        {"mathematical-benchmarks-v1": suite},
    )

    assert len(plan.entries) == 4
    assert plan.reasons == (
        "focused host validation exceeded matrix job limit; using full suite",
    )


def test_conjecture_probes_v1_task_change_selects_only_leaf(tmp_path: Path) -> None:
    """A conjecture-probes-v1 task change must select its dedicated leaf test
    when one exists, rather than escalating to full host validation."""
    from types import SimpleNamespace

    from benchmarks.tooling.validation_plan import host_validation_plan

    task = "hadwiger-triangle-free-minor-certificate"
    leaf = (
        tmp_path
        / "benchmarks/validation/conjecture_probes_v1"
        / "test_hadwiger_triangle_free_minor_certificate.py"
    )
    leaf.parent.mkdir(parents=True)
    leaf.write_text("", encoding="utf-8")
    suite = SimpleNamespace(tasks=(SimpleNamespace(path=Path(task)),))

    plan = host_validation_plan(
        tmp_path,
        [f"benchmarks/datasets/conjecture-probes-v1/{task}/tests/verifier.py"],
        {"conjecture-probes-v1": suite},
    )

    assert [entry.selector for entry in plan.entries] == [
        "benchmarks/validation/conjecture_probes_v1/"
        "test_hadwiger_triangle_free_minor_certificate.py",
    ]
    assert all(reason.startswith("focused host validation:") for reason in plan.reasons)
