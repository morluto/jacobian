"""Strict repository-owned contracts for the Harbor benchmark control plane."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from benchmarks.tooling.benchmark_snapshots import validate_lock
from benchmarks.tooling.harbor_suite import (
    BENCHMARKS,
    ROOT,
    HarborSuiteError,
    Suite,
    load_registry,
)
from benchmarks.tooling.strict_boundaries import (
    HarborJobSelection,
    strict_model_failures,
)
from benchmarks.tooling.verifier_audits import (
    canonical_string_rational_schema_failures,
    formula_string_schema_failures,
    fraction_coprimality_failures,
    hidden_expected_scoring_failures,
    mirror_witness_failures,
    prose_witness_failures,
    unread_hash_witness_failures,
)

SCHEMAS = BENCHMARKS / "schemas"
SNAPSHOTS = BENCHMARKS / "snapshots"


@dataclass(frozen=True, slots=True)
class BenchmarkContractInventory:
    """Deterministic repository-owned inputs outside suite membership."""

    schemas: tuple[Path, ...]
    proxy_jobs: tuple[Path, ...]
    snapshot_locks: tuple[Path, ...]


def benchmark_contract_inventory() -> BenchmarkContractInventory:
    """Discover the fixed repository contracts used by the aggregate gate."""

    return BenchmarkContractInventory(
        schemas=tuple(sorted(SCHEMAS.glob("*.schema.json"))),
        proxy_jobs=(
            BENCHMARKS / "config" / "mathematical-benchmarks-v1-control-proxy.json",
            BENCHMARKS
            / "datasets"
            / "mathematical-benchmarks-v1"
            / "jobs"
            / "jacobian-observation-proxy.json",
        ),
        snapshot_locks=tuple(sorted(SNAPSHOTS.rglob("*.lock.json"))),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(
            f"{path.relative_to(ROOT)}: invalid JSON: {exc}"
        ) from exc


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarborSuiteError(
            f"{path.relative_to(ROOT)}: invalid TOML: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HarborSuiteError(f"{path.relative_to(ROOT)}: TOML root must be a table")
    return value


@lru_cache(maxsize=16)
def _validator(schema_name: str) -> Draft202012Validator:
    path = SCHEMAS / schema_name
    schema = _read_json(path)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise HarborSuiteError(
            f"{path.relative_to(ROOT)}: invalid JSON Schema: {exc.message}"
        ) from exc
    return Draft202012Validator(schema)


def _validate(instance: Any, schema_name: str, path: Path) -> list[str]:
    failures: list[str] = []
    for error in sorted(
        _validator(schema_name).iter_errors(instance),
        key=lambda item: list(item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path)
        suffix = f" at {location}" if location else ""
        failures.append(f"{path.relative_to(ROOT)}{suffix}: {error.message}")
    return failures


def _dataset_selection_failures(
    datasets: Any, *, path: Path, suite: Suite | None
) -> list[str]:
    failures: list[str] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        dataset_path = dataset.get("path")
        if not isinstance(dataset_path, str):
            continue
        resolved = (ROOT / dataset_path).resolve()
        try:
            resolved.relative_to(BENCHMARKS / "datasets")
        except ValueError:
            failures.append(
                f"{path.relative_to(ROOT)}: dataset path escapes benchmarks/datasets"
            )
            continue
        if not resolved.is_dir():
            failures.append(
                f"{path.relative_to(ROOT)}: dataset path does not exist: {dataset_path}"
            )
        if suite is not None and resolved != suite.path:
            failures.append(
                f"{path.relative_to(ROOT)}: job selects a different dataset"
            )
        known = {ref.path.name for ref in suite.tasks} if suite is not None else set()
        unknown = sorted(set(dataset.get("task_names", [])) - known) if known else []
        if unknown:
            failures.append(
                f"{path.relative_to(ROOT)}: unknown task_names: {', '.join(unknown)}"
            )
    return failures


def _task_selection_failures(
    tasks: Any, *, path: Path, suite: Suite | None
) -> list[str]:
    failures: list[str] = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("path"), str):
            continue
        declared = str(task["path"])
        resolved = (ROOT / declared).resolve()
        try:
            resolved.relative_to(BENCHMARKS / "datasets")
        except ValueError:
            failures.append(
                f"{path.relative_to(ROOT)}: task path escapes benchmarks/datasets"
            )
            continue
        if not resolved.is_dir() or not (resolved / "task.toml").is_file():
            failures.append(
                f"{path.relative_to(ROOT)}: task path is not a Harbor task: {declared}"
            )
            continue
        if suite is not None and resolved.parent != suite.path:
            failures.append(
                f"{path.relative_to(ROOT)}: task path selects a different dataset"
            )
    return failures


def validate_job_contract(
    raw: object,
    *,
    path: Path,
    suite: Suite | None = None,
) -> list[str]:
    """Validate one parsed Harbor job through the aggregate gate's contract."""

    failures = _validate(raw, "harbor-job.schema.json", path)
    if not isinstance(raw, dict):
        return failures
    if ("datasets" in raw) == ("tasks" in raw):
        failures.append(
            f"{path.relative_to(ROOT)}: select exactly one of datasets or tasks"
        )
    # Strict typed validation of the dataset/task selection entries runs before
    # the semantic path-resolution checks so a malformed selection fails closed
    # with a field-path diagnostic instead of reaching artifact side effects.
    selection_payload: dict[str, Any] = {}
    if "datasets" in raw:
        selection_payload["datasets"] = raw["datasets"]
    if "tasks" in raw:
        selection_payload["tasks"] = raw["tasks"]
    structural_failures = strict_model_failures(
        HarborJobSelection,
        selection_payload,
        label=str(path.relative_to(ROOT)),
    )
    if structural_failures:
        return [*failures, *structural_failures]
    failures.extend(
        _dataset_selection_failures(raw.get("datasets", []), path=path, suite=suite)
    )
    failures.extend(
        _task_selection_failures(raw.get("tasks", []), path=path, suite=suite)
    )
    return failures


def _validate_job(path: Path, suite: Suite | None = None) -> list[str]:
    return validate_job_contract(_read_json(path), path=path, suite=suite)


def _validate_task(suite: Suite, task_dir: Path) -> list[str]:
    failures: list[str] = []
    task_toml = task_dir / "task.toml"
    raw = _read_toml(task_toml)
    metadata = raw.get("metadata")
    failures.extend(_validate(metadata, "task-metadata.schema.json", task_toml))

    expected_id = f"jacobian/{task_dir.name}"
    input_path = task_dir / "environment" / "input.json"
    input_payload = _read_json(input_path)
    if (
        not isinstance(input_payload, dict)
        or input_payload.get("task_id") != expected_id
    ):
        failures.append(
            f"{input_path.relative_to(ROOT)}: task_id must be {expected_id!r}"
        )

    schema_path = task_dir / "environment" / "submission_schema.json"
    submission_schema = _read_json(schema_path)
    try:
        Draft202012Validator.check_schema(submission_schema)
    except SchemaError as exc:
        failures.append(
            f"{schema_path.relative_to(ROOT)}: invalid JSON Schema: {exc.message}"
        )
        return failures
    contract_path = task_dir / "tests" / "public_contract.json"
    if contract_path.is_file():
        failures.extend(
            _validate(
                _read_json(contract_path),
                "public-contract.schema.json",
                contract_path,
            )
        )
    failures.extend(fraction_coprimality_failures(task_dir / "tests" / "verifier.py"))
    failures.extend(canonical_string_rational_schema_failures(schema_path))
    failures.extend(mirror_witness_failures(task_dir / "tests" / "verifier.py"))
    failures.extend(unread_hash_witness_failures(task_dir / "tests" / "verifier.py"))
    failures.extend(formula_string_schema_failures(schema_path))
    failures.extend(
        hidden_expected_scoring_failures(task_dir / "tests" / "verifier.py")
    )
    failures.extend(prose_witness_failures(task_dir / "tests" / "verifier.py"))
    solution_path = task_dir / "solution" / "submission.json"
    if not solution_path.is_file():
        # Measurement and provider tasks construct their Oracle submission at
        # runtime.  Their generated value is schema-checked by the verifier and
        # full-reward Oracle lane; the static gate still checks the schema itself.
        return failures
    solution = _read_json(solution_path)
    try:
        Draft202012Validator(submission_schema).validate(solution)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        suffix = f" at {location}" if location else ""
        failures.append(f"{solution_path.relative_to(ROOT)}{suffix}: {exc.message}")
    failures.extend(_validate_gold_witness(solution, solution_path, task_dir))
    return failures


def _regular_file_inside(root: Path, relative: str) -> Path | None:
    """Return a regular non-symlink file contained in ``root``, else None."""

    if not relative or Path(relative).is_absolute():
        return None
    try:
        if root.is_symlink() or not root.is_dir():
            return None
    except OSError:
        return None
    current = root
    for part in Path(relative).parts:
        if part in {"", ".", ".."}:
            return None
        current = current / part
        try:
            if current.is_symlink():
                return None
        except OSError:
            return None
    try:
        if current.is_symlink() or not current.is_file():
            return None
        current.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return current


def _validate_gold_witness(
    solution: object, solution_path: Path, task_dir: Path
) -> list[str]:
    """Check that gold witness descriptors point to real, matching artifacts."""

    if not isinstance(solution, dict):
        return []
    witness = solution.get("witness")
    if not isinstance(witness, list):
        return []
    rel = solution_path.relative_to(ROOT)
    failures: list[str] = []
    for i, descriptor in enumerate(witness):
        if not isinstance(descriptor, dict):
            failures.append(f"{rel}.witness[{i}]: not an object")
            continue
        path_str = descriptor.get("path")
        sha256 = descriptor.get("sha256")
        if not isinstance(path_str, str) or not isinstance(sha256, str):
            failures.append(f"{rel}.witness[{i}]: missing path or sha256")
            continue
        if ".." in Path(path_str).parts or Path(path_str).is_absolute():
            failures.append(f"{rel}.witness[{i}]: path escapes task directory")
            continue
        artifact = _regular_file_inside(task_dir / "solution", path_str)
        if artifact is None:
            failures.append(
                f"{rel}.witness[{i}]: artifact {path_str} is not a regular "
                "non-symlink file inside solution/"
            )
            continue

        digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != sha256:
            failures.append(f"{rel}.witness[{i}]: sha256 mismatch for {path_str}")
    return failures


def _observation_pair_failures() -> list[str]:
    treatment_path = (
        BENCHMARKS
        / "datasets"
        / "mathematical-benchmarks-v1"
        / "jobs"
        / "jacobian-observation.json"
    )
    control_path = BENCHMARKS / "config" / "mathematical-benchmarks-v1-control.json"
    treatment = _read_json(treatment_path)
    control = _read_json(control_path)
    if not isinstance(treatment, dict) or not isinstance(control, dict):
        return ["observation/control pair contains malformed (non-object) JSON"]

    def normalized(value: dict[str, Any]) -> dict[str, Any]:
        copy: dict[str, Any] = json.loads(json.dumps(value))
        copy.pop("jobs_dir", None)
        artifacts = copy.get("artifacts")
        if isinstance(artifacts, list):
            copy["artifacts"] = [
                entry
                for entry in artifacts
                if entry != {"source": "/logs/jacobian/mcp.log", "service": "jacobian"}
            ]
        environment = copy.get("environment")
        if isinstance(environment, dict):
            compose = environment.get("extra_docker_compose")
            if isinstance(compose, list):
                environment["extra_docker_compose"] = [
                    item
                    for item in compose
                    if "jacobian-observation.compose.yaml" not in item
                ]
        return copy

    if normalized(treatment) != normalized(control):
        return [
            "agent workflow control/treatment jobs differ outside the allowed "
            "jobs_dir, Jacobian sidecar composition, and sidecar telemetry "
            "artifact"
        ]
    return []


def _suite_contract_failures(suites: tuple[Suite, ...]) -> list[str]:
    failures: list[str] = []
    global_ids: dict[str, str] = {}
    for suite in suites:
        failures.extend(
            _validate(
                _read_toml(suite.suite_manifest),
                "suite.schema.json",
                suite.suite_manifest,
            )
        )
        for member in sorted((suite.path / "members").glob("*.toml")):
            failures.extend(
                _validate(_read_toml(member), "suite-entry.schema.json", member)
            )
            member_id = member.stem
            raw = _read_toml(member)
            if raw.get("task_id") != member_id:
                failures.append(
                    f"{member.relative_to(ROOT)}: filename must match task_id"
                )
        for ref in suite.tasks:
            previous = global_ids.setdefault(ref.path.name, suite.id)
            if previous != suite.id:
                failures.append(
                    f"global task id collision: {ref.path.name} belongs to {previous} and {suite.id}"
                )
            failures.extend(_validate_task(suite, ref.path))
        failures.extend(_validate_job(suite.job_oracle, suite))
        if suite.job_observation is not None:
            failures.extend(_validate_job(suite.job_observation, suite))
    return failures


def _snapshot_contract_failures(lock_paths: Iterable[Path]) -> list[str]:
    failures: list[str] = []
    for lock_path in lock_paths:
        try:
            lock = validate_lock(lock_path)
        except HarborSuiteError as exc:
            failures.append(f"{lock_path.relative_to(ROOT)}: {exc}")
            continue
        expected_name = str(lock["snapshot_id"]).removeprefix("sha256:")
        if lock_path.name.removesuffix(".lock.json") != expected_name:
            failures.append(
                f"{lock_path.relative_to(ROOT)}: filename must match snapshot_id"
            )
        if lock_path.parent.name != lock["suite"]["id"]:
            failures.append(
                f"{lock_path.relative_to(ROOT)}: parent directory must match suite id"
            )
    return failures


def collect_contract_failures(
    validators: Iterable[Callable[[], list[str]]],
) -> list[str]:
    """Run every aggregate contract phase and retain deterministic failure order."""

    return [failure for validate in validators for failure in validate()]


def validate_all() -> list[str]:
    """Return every benchmark contract failure without stopping at the first."""
    inventory = benchmark_contract_inventory()
    for schema_path in inventory.schemas:
        _validator(schema_path.name)
    failures = _validate(
        _read_toml(BENCHMARKS / "registry.toml"),
        "registry.schema.json",
        BENCHMARKS / "registry.toml",
    )
    suites = load_registry()
    failures.extend(_suite_contract_failures(suites))
    failures.extend(
        _validate_job(
            BENCHMARKS / "config" / "mathematical-benchmarks-v1-control.json", suites[0]
        )
    )
    failures.extend(
        collect_contract_failures(
            partial(_validate_job, path, suites[0]) for path in inventory.proxy_jobs
        )
    )
    failures.extend(_observation_pair_failures())
    failures.extend(_snapshot_contract_failures(inventory.snapshot_locks))
    return failures


__all__ = [
    "BenchmarkContractInventory",
    "benchmark_contract_inventory",
    "collect_contract_failures",
    "validate_all",
    "validate_job_contract",
]
