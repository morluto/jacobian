"""Reject direct Harbor runs whose selected agent tasks cannot install Codex."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _task_manifests(
    *,
    root: Path,
    config_path: Path,
    dataset: str,
    task_names: str | None,
) -> list[Path]:
    """Resolve the manifests selected by Make overrides or the Harbor job."""

    manifests = (
        _task_manifests_from_names(root, dataset, task_names)
        if task_names
        else _task_manifests_from_config(root, config_path)
    )

    for manifest in manifests:
        if not manifest.is_file():
            raise ValueError(f"selected task manifest is missing: {manifest}")
    return list(dict.fromkeys(manifests))


def _task_manifests_from_names(
    root: Path,
    dataset: str,
    task_names: str,
) -> list[Path]:
    """Resolve Make's DATASET and TASKS overrides beneath the repository."""

    dataset_base = _resolve_path(root, "benchmarks/datasets")
    dataset_root = _resolve_path(dataset_base, dataset)
    try:
        dataset_root.relative_to(dataset_base)
    except ValueError as error:
        raise ValueError(f"DATASET escapes benchmarks/datasets: {dataset!r}") from error
    names = task_names.split()
    if not names:
        raise ValueError("TASKS must contain at least one task name")
    manifests = []
    for name in names:
        task_root = (dataset_root / name).resolve()
        try:
            task_root.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(f"TASKS entry escapes DATASET: {name!r}") from error
        manifests.append(task_root / "task.toml")
    return manifests


def _task_manifests_from_config(root: Path, config_path: Path) -> list[Path]:
    """Resolve task manifests named by the selected Harbor job configuration."""

    path = _resolve_path(root, str(config_path))
    job: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(job, dict):
        raise ValueError("EVAL_CONFIG must contain a JSON object")
    datasets = job.get("datasets")
    tasks = job.get("tasks")
    if datasets is not None and tasks is not None:
        raise ValueError("EVAL_CONFIG must select via datasets or tasks, not both")
    manifests = _dataset_manifests(root, datasets) if isinstance(datasets, list) else []
    if isinstance(tasks, list):
        manifests.extend(_explicit_task_manifests(root, tasks))
    if not manifests:
        raise ValueError("EVAL_CONFIG does not identify any selected tasks")
    return manifests


def _dataset_manifests(root: Path, datasets: list[Any]) -> list[Path]:
    """Resolve the task manifests selected through Harbor dataset entries."""

    manifests: list[Path] = []
    for entry in datasets:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("EVAL_CONFIG has an invalid dataset selection")
        dataset_root = _resolve_path(root, entry["path"])
        names = entry.get("task_names")
        if names is None:
            manifests.extend(sorted(dataset_root.rglob("task.toml")))
        elif isinstance(names, list) and all(isinstance(name, str) for name in names):
            manifests.extend(
                (dataset_root / name / "task.toml").resolve() for name in names
            )
        else:
            raise ValueError("EVAL_CONFIG task_names must be an array of strings")
    return manifests


def _explicit_task_manifests(root: Path, tasks: list[Any]) -> list[Path]:
    """Resolve the task manifests selected by explicit Harbor task paths."""

    manifests: list[Path] = []
    for entry in tasks:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("EVAL_CONFIG has an invalid explicit task selection")
        manifests.append((_resolve_path(root, entry["path"]) / "task.toml").resolve())
    return manifests


def no_network_tasks(
    *,
    root: Path,
    config_path: Path,
    dataset: str,
    task_names: str | None,
) -> list[str]:
    """Return selected task identities whose agent container has no network."""

    blocked: list[str] = []
    for manifest in _task_manifests(
        root=root,
        config_path=config_path,
        dataset=dataset,
        task_names=task_names,
    ):
        config = tomllib.loads(manifest.read_text(encoding="utf-8"))
        environment = config.get("environment", {})
        if (
            isinstance(environment, dict)
            and environment.get("network_mode") == "no-network"
        ):
            try:
                name = manifest.parent.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                name = str(manifest.parent)
            blocked.append(name)
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks", help="Space-separated TASKS Make variable")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        blocked = no_network_tasks(
            root=args.root,
            config_path=args.config,
            dataset=args.dataset,
            task_names=args.tasks,
        )
    except (
        OSError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        ValueError,
    ) as error:
        print(
            f"Cannot preflight the selected Harbor task network mode: {error}",
            file=sys.stderr,
        )
        return 2

    if blocked:
        print(
            "Direct agent-eval cannot install the Codex runtime in selected no-network task(s): "
            f"{', '.join(blocked)}. No trials were scheduled. Set JACOBIAN_EVAL_PROXY=1 "
            "and configure a reachable upstream proxy, then rerun.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
