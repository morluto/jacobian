"""Stable host-side validation planning for Harbor benchmark changes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HOST_VALIDATION_ROOT = "benchmarks/validation"
HOST_VALIDATION_FULL_SHARDS = 4
# GitHub Actions matrix jobs are capped at 256; stay at or below that limit.
HOST_VALIDATION_MAX_JOBS = 256
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST_VALIDATION_DATASET_FILES = {
    "multi-tool-coordination-v1": (
        "benchmarks/validation/multi_tool_coordination_v1/test_pilot_contract.py",
    ),
    "symbolic-coordination-v1": (
        "benchmarks/validation/symbolic_coordination_v1/test_pilot_contract.py",
    ),
    "research-diagnostics-v1": (
        "benchmarks/validation/research_diagnostics_v1/test_structured_verifiers.py",
    ),
    "provider-feasibility-v1": (
        "benchmarks/validation/test_provider_download_integrity.py",
    ),
}


def _discover_conjecture_host_tests() -> tuple[str, ...]:
    """Discover dedicated conjecture-probes-v1 host test files in sorted order."""

    root = _PROJECT_ROOT / "benchmarks" / "validation" / "conjecture_probes_v1"
    return tuple(
        str(path.relative_to(_PROJECT_ROOT).as_posix())
        for path in sorted(root.glob("test_*.py"))
    )


CONTROL_PLANE_HOST_TESTS = {
    ".github/scripts/emit-plan-receipt": (),
    ".github/scripts/manage-test-timings": (),
    ".github/scripts/plan-benchmarks": (
        "benchmarks/validation/test_benchmark_planner.py",
        "benchmarks/validation/test_benchmark_plan_validation.py",
    ),
    ".github/scripts/validate-benchmark-plan": (
        "benchmarks/validation/test_benchmark_plan_validation.py",
    ),
    ".github/workflows/benchmarks.yml": (
        "benchmarks/validation/test_benchmark_planner.py",
        "benchmarks/validation/test_benchmark_validation.py",
    ),
    ".github/workflows/heldout-benchmarks.yml": (
        "benchmarks/validation/test_heldout_bundle.py",
        "benchmarks/validation/test_heldout_runner.py",
    ),
    "benchmarks/tooling/benchmark_timings.py": (
        "benchmarks/validation/test_benchmark_timings.py",
    ),
    "benchmarks/tooling/benchmark_validation.py": (
        "benchmarks/validation/test_benchmark_validation.py",
    ),
    "benchmarks/tooling/host_validation.py": (
        "benchmarks/validation/test_host_validation.py",
    ),
    "benchmarks/tooling/validation_plan.py": (
        "benchmarks/validation/test_validation_plan.py",
        "benchmarks/validation/test_benchmark_planner.py",
    ),
    "tools/benchmark_pr_status.py": (
        "benchmarks/validation/test_benchmark_pr_status.py",
    ),
    "tools/check_benchmark_adapters.py": (
        "benchmarks/validation/test_benchmark_adapters.py",
    ),
    "tools/check_benchmark_contracts.py": (
        "benchmarks/validation/test_benchmark_contracts.py",
    ),
    "tools/check_benchmark_static.py": (
        "benchmarks/validation/test_benchmark_static.py",
    ),
    "tools/harbor_task_workflow.py": (
        "benchmarks/validation/test_harbor_task_workflow.py",
    ),
    "tools/check_harbor_dataset.py": (
        "benchmarks/validation/test_mathematical_benchmarks_v1.py",
    ),
    "tools/sync_harbor_verifier_support.py": (
        "benchmarks/validation/test_public_contract.py",
    ),
    "benchmarks/environment-profiles.toml": (
        "benchmarks/validation/test_benchmark_environment_profiles.py",
    ),
}
SHARED_HOST_HARNESS_PATHS = {
    "Makefile",
    "tools/pytest_lifecycle.py",
}


@dataclass(frozen=True, slots=True)
class HostValidation:
    """One bounded pytest selector in the host-validation matrix."""

    name: str
    selector: str
    keyword: str = ""
    splits: int = 0
    group: int = 0
    predicted_seconds: float = 1.0

    def as_matrix_entry(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HostValidationPlan:
    """Selected host checks plus human-readable escalation reasons."""

    entries: tuple[HostValidation, ...]
    reasons: tuple[str, ...]


def _prediction(name: str, timings: Mapping[str, float] | None) -> float:
    value = (timings or {}).get(f"host-validation/{name}", 1.0)
    return round(float(value), 6)


def _entry(
    *,
    name: str,
    selector: str,
    timings: Mapping[str, float] | None,
    keyword: str = "",
    splits: int = 0,
    group: int = 0,
) -> HostValidation:
    return HostValidation(
        name=name,
        selector=selector,
        keyword=keyword,
        splits=splits,
        group=group,
        predicted_seconds=_prediction(name, timings),
    )


def full_host_validation(
    *, timings: Mapping[str, float] | None = None
) -> tuple[HostValidation, ...]:
    return tuple(
        _entry(
            name=f"full-{group}-of-{HOST_VALIDATION_FULL_SHARDS}",
            selector=HOST_VALIDATION_ROOT,
            splits=HOST_VALIDATION_FULL_SHARDS,
            group=group,
            timings=timings,
        )
        for group in range(1, HOST_VALIDATION_FULL_SHARDS + 1)
    )


def dataset_host_validation(
    dataset: str, *, timings: Mapping[str, float] | None = None
) -> tuple[HostValidation, ...]:
    if dataset == "conjecture-probes-v1":
        selectors = _discover_conjecture_host_tests()
    else:
        selectors = HOST_VALIDATION_DATASET_FILES.get(dataset, ())
    return tuple(
        _entry(name=f"{dataset}-{index}", selector=selector, timings=timings)
        for index, selector in enumerate(selectors, start=1)
    )


def task_host_validation(
    root: Path,
    dataset: str,
    task: str,
    *,
    timings: Mapping[str, float] | None = None,
) -> tuple[HostValidation, ...]:
    """Select the leaf and task-filtered generic contracts for one task."""
    if dataset == "mathematical-benchmarks-v1":
        dedicated = (
            root
            / "benchmarks"
            / "validation"
            / "mathematical_benchmarks_v1"
            / f"test_{task.replace('-', '_')}.py"
        )
        entries: list[HostValidation] = []
        if dedicated.is_file():
            entries.append(
                _entry(
                    name=f"{task}-specific",
                    selector=dedicated.relative_to(root).as_posix(),
                    timings=timings,
                )
            )
        entries.append(
            _entry(
                name=f"{task}-generic",
                selector=(
                    "benchmarks/validation/mathematical_benchmarks_v1/"
                    "test_generic_verifier_contracts.py"
                ),
                keyword=task,
                timings=timings,
            )
        )
        return tuple(entries)
    if dataset == "conjecture-probes-v1":
        dedicated = (
            root
            / "benchmarks"
            / "validation"
            / "conjecture_probes_v1"
            / f"test_{task.replace('-', '_')}.py"
        )
        if dedicated.is_file():
            return (
                _entry(
                    name=task,
                    selector=dedicated.relative_to(root).as_posix(),
                    timings=timings,
                ),
            )
        return dataset_host_validation(dataset, timings=timings)
    if dataset == "public-reproductions-v1" and task == "sat-erdos-schur-f4":
        return (
            _entry(
                name=task,
                selector="benchmarks/validation/test_sat_erdos_schur_f4.py",
                timings=timings,
            ),
        )
    if dataset == "conjecture-probes-v1":
        dedicated = (
            root
            / "benchmarks"
            / "validation"
            / "conjecture_probes_v1"
            / f"test_{task.replace('-', '_')}.py"
        )
        conjecture_entries: list[HostValidation] = []
        if dedicated.is_file():
            conjecture_entries.append(
                _entry(
                    name=f"{task}-specific",
                    selector=dedicated.relative_to(root).as_posix(),
                    timings=timings,
                )
            )
        return tuple(conjecture_entries) or dataset_host_validation(
            dataset, timings=timings
        )
    return dataset_host_validation(dataset, timings=timings)


def _validation_path_plan(
    root: Path, path: str, timings: Mapping[str, float] | None
) -> tuple[tuple[HostValidation, ...], str | None]:
    candidate = root / path
    if candidate.is_file() and candidate.name.startswith("test_"):
        return (_entry(name=candidate.stem, selector=path, timings=timings),), None
    if candidate.suffix == ".py":
        return (), f"shared verifier harness requires full host validation: {path}"
    return (), f"unclassified validation path requires full host validation: {path}"


def _dataset_path_plan(
    root: Path,
    path: str,
    suites_by_id: Mapping[str, Any],
    timings: Mapping[str, float] | None,
) -> tuple[tuple[HostValidation, ...], str | None]:
    parts = Path(path).parts
    dataset = parts[2]
    suite = suites_by_id.get(dataset)
    task_ids = {ref.path.name for ref in suite.tasks} if suite else set()
    relative = parts[3:]
    task = relative[0] if len(relative) >= 2 else None
    if task is not None and task in task_ids:
        entries = (
            ()
            if relative[1:] == ("README.md",)
            else task_host_validation(root, dataset, task, timings=timings)
        )
        return entries, None
    if relative[:1] == ("members",) and len(relative) == 2:
        member = Path(relative[1]).stem
        if member in task_ids:
            return task_host_validation(root, dataset, member, timings=timings), None
    if relative[:1] in {("README.md",), ("dataset.toml",)}:
        return (), None
    entries = dataset_host_validation(dataset, timings=timings)
    if entries:
        return entries, None
    return (), f"dataset-wide change requires full host validation: {path}"


def _shared_path_plan(
    path: str, timings: Mapping[str, float] | None
) -> tuple[tuple[HostValidation, ...], str | None]:
    owned = CONTROL_PLANE_HOST_TESTS.get(path)
    if owned is not None:
        return tuple(
            _entry(
                name=f"control-{Path(selector).stem}",
                selector=selector,
                timings=timings,
            )
            for selector in owned
        ), None
    if path in SHARED_HOST_HARNESS_PATHS:
        return (
            (),
            f"shared verifier execution harness requires full host validation: {path}",
        )
    if path in {"benchmarks/README.md", "benchmarks/__init__.py"} or path.startswith(
        "benchmarks/templates/"
    ):
        return (), None
    if path == "benchmarks/adapters/README.md":
        return (), None
    if path.startswith("benchmarks/adapters/"):
        return (
            _entry(
                name="benchmark-adapters",
                selector="benchmarks/validation/test_benchmark_adapters.py",
                timings=timings,
            ),
        ), None
    return (), f"shared benchmark tooling requires full host validation: {path}"


def host_validation_plan(
    root: Path,
    benchmark_paths: Sequence[str],
    suites_by_id: Mapping[str, Any],
    *,
    timings: Mapping[str, float] | None = None,
) -> HostValidationPlan:
    """Plan host regressions, escalating only paths with portfolio-wide reach."""
    entries: list[HostValidation] = []
    full_reasons: list[str] = []
    for path in benchmark_paths:
        if path.startswith("benchmarks/validation/"):
            selected, reason = _validation_path_plan(root, path, timings)
        elif len(Path(path).parts) >= 4 and Path(path).parts[:2] == (
            "benchmarks",
            "datasets",
        ):
            selected, reason = _dataset_path_plan(root, path, suites_by_id, timings)
        else:
            selected, reason = _shared_path_plan(path, timings)
        entries.extend(selected)
        if reason is not None:
            full_reasons.append(reason)
    if full_reasons:
        return HostValidationPlan(
            full_host_validation(timings=timings), tuple(dict.fromkeys(full_reasons))
        )
    unique = {(entry.selector, entry.keyword): entry for entry in entries}
    ordered = tuple(
        sorted(unique.values(), key=lambda entry: (entry.selector, entry.keyword))
    )
    if len(ordered) > HOST_VALIDATION_MAX_JOBS:
        return HostValidationPlan(
            full_host_validation(timings=timings),
            ("focused host validation exceeded matrix job limit; using full suite",),
        )
    reasons = tuple(f"focused host validation: {entry.name}" for entry in ordered)
    return HostValidationPlan(ordered, reasons)


__all__ = [
    "CONTROL_PLANE_HOST_TESTS",
    "HOST_VALIDATION_MAX_JOBS",
    "HostValidation",
    "HostValidationPlan",
    "dataset_host_validation",
    "full_host_validation",
    "host_validation_plan",
    "task_host_validation",
]
