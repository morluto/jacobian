"""Validate, fetch, and render private held-out Harbor evaluation bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tarfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.command_runner import operator_environment, run_operator_command
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import (
    BENCHMARKS,
    task_digest,
)
from benchmarks.tooling.strict_boundaries import HeldoutManifest, raise_strict_model

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_AWS_ENVIRONMENT_VARS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    }
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"invalid JSON {path}: {exc}") from exc


def _digest(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HarborSuiteError(f"unable to digest held-out file {path}: {exc}") from exc


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _tree_digest(root: Path) -> str:
    """Bind a complete regular-file tree, including names and empty-tree state."""
    if not root.is_dir():
        raise HarborSuiteError(f"held-out tree is not a directory: {root}")
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HarborSuiteError(f"held-out tree contains a symlink: {path}")
        if path.is_file():
            entries.append(
                {"path": path.relative_to(root).as_posix(), "digest": _digest(path)}
            )
    return _json_digest(entries)


def _bundle_path(root: Path, declared: str) -> Path:
    path = root / declared
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HarborSuiteError(f"held-out path escapes bundle: {declared}") from exc
    return path


def _validate_task_contracts(manifest: dict[str, Any]) -> set[str]:
    task_ids = [item["id"] for item in manifest["tasks"]]
    if len(set(task_ids)) != len(task_ids):
        raise HarborSuiteError("held-out task ids must be unique")
    families = {item["family"] for item in manifest["tasks"]}
    if len(families) < manifest["dataset"]["minimum_independent_families"]:
        raise HarborSuiteError("held-out bundle has too few independent families")
    for task in manifest["tasks"]:
        expected_prefix = f"dataset/{task['id']}/"
        roots = (task["verifier_root"], task["oracle_root"])
        if any(not root.startswith(expected_prefix) for root in roots):
            raise HarborSuiteError(
                f"held-out verifier/oracle roots must belong to task {task['id']}"
            )
    return set(task_ids)


def _validate_experiment(manifest: dict[str, Any], task_ids: set[str]) -> None:
    stages = manifest["experiment"]["stages"]
    for stage, config in stages.items():
        unknown = sorted(set(config["task_ids"]) - task_ids)
        if unknown:
            raise HarborSuiteError(f"{stage} references unknown task ids: {unknown}")
    if len(stages["pilot"]["task_ids"]) != 3:
        raise HarborSuiteError("pilot must freeze exactly three tasks")
    decision = stages["decision"]
    if len(decision["task_ids"]) < 5 or decision["repetitions"] < 5:
        raise HarborSuiteError(
            "decision stage requires at least five tasks and repetitions"
        )


def _validate_snapshot_lock(manifest: dict[str, Any]) -> None:
    lock = manifest["snapshot_lock"]
    if not _DIGEST_RE.match(lock["lock_id"]):
        raise HarborSuiteError("held-out snapshot_lock.lock_id must be a sha256 digest")
    if not _DIGEST_RE.match(lock["lock_digest"]):
        raise HarborSuiteError(
            "held-out snapshot_lock.lock_digest must be a sha256 digest"
        )
    if not isinstance(lock["lock_uri"], str) or not lock["lock_uri"]:
        raise HarborSuiteError("held-out snapshot_lock.lock_uri must be non-empty")


def validate_manifest(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    # Typed Pydantic boundary first: extra="forbid", strict scalars, no
    # semantic .get/indexing before typed parse.
    model = raise_strict_model(HeldoutManifest, raw, label=str(path))
    manifest = model.model_dump(mode="json", exclude_none=True)
    # JSON Schema as a secondary contract check for pattern constraints
    # (digest format, S3 URI shape, path patterns) not expressible in the
    # strict Pydantic model.
    schema = _read_json(BENCHMARKS / "schemas" / "held-out-manifest.schema.json")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        messages = [
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        ]
        raise HarborSuiteError("held-out manifest is invalid:\n" + "\n".join(messages))
    _validate_snapshot_lock(manifest)
    task_ids = _validate_task_contracts(manifest)
    _validate_experiment(manifest, task_ids)
    conditions = {
        item["id"]: (item["role"], item["jacobian_enabled"])
        for item in manifest["conditions"]
    }
    if conditions != {
        "C1": ("PRIMARY_CONTROL", False),
        "C2": ("PRIMARY_TREATMENT", True),
    }:
        raise HarborSuiteError("held-out conditions must be the frozen C1/C2 pair")
    return manifest


def _safe_extract(archive: Path, output: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            try:
                target.relative_to(output.resolve())
            except ValueError as exc:
                raise HarborSuiteError(
                    f"held-out archive path escapes output: {member.name}"
                ) from exc
            if member.issym() or member.islnk() or member.isdev():
                raise HarborSuiteError(
                    f"held-out archive contains a forbidden entry: {member.name}"
                )
        tar.extractall(output, members=members, filter="data")


def _verify_dataset_manifest(manifest: dict[str, Any], dataset_manifest: Path) -> None:
    if _digest(dataset_manifest) != manifest["dataset"]["manifest_digest"]:
        raise HarborSuiteError("held-out dataset manifest digest mismatch")
    if dataset_manifest.is_symlink():
        raise HarborSuiteError("held-out dataset manifest is a forbidden symlink")
    try:
        dataset_value = tomllib.loads(dataset_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarborSuiteError(f"held-out dataset manifest is invalid: {exc}") from exc
    entries = dataset_value.get("tasks") if isinstance(dataset_value, dict) else None
    if not isinstance(entries, list):
        raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")
    declared: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")
        name = entry.get("name")
        digest = entry.get("digest")
        if not isinstance(name, str) or not isinstance(digest, str) or name in declared:
            raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")
        declared[name] = digest
    expected = {
        f"jacobian/{task['id']}": str(task["digest"]) for task in manifest["tasks"]
    }
    if declared != expected:
        raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")


def _verify_snapshot_lock(manifest: dict[str, Any], bundle_root: Path) -> None:
    """Verify the archive's snapshot lock agrees with the manifest.

    The snapshot lock is canonical: its task IDs/digests must match the
    manifest's tasks, and its lock_digest/snapshot_id must match the
    manifest's snapshot_lock reference.
    """

    lock_ref = manifest["snapshot_lock"]
    lock_path = bundle_root / "snapshot-lock.json"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise HarborSuiteError("held-out bundle is missing snapshot-lock.json")
    actual_digest = _digest(lock_path)
    if actual_digest != lock_ref["lock_digest"]:
        raise HarborSuiteError("held-out snapshot lock digest mismatch")
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        raise HarborSuiteError("held-out snapshot lock must be a JSON object")
    lock_snapshot_id = lock.get("snapshot_id")
    if not isinstance(lock_snapshot_id, str) or lock_snapshot_id != lock_ref["lock_id"]:
        raise HarborSuiteError("held-out snapshot lock_id mismatch")
    lock_tasks = lock.get("tasks")
    if not isinstance(lock_tasks, list) or not lock_tasks:
        raise HarborSuiteError("held-out snapshot lock has no tasks")
    lock_task_map: dict[str, str] = {}
    for entry in lock_tasks:
        if not isinstance(entry, dict):
            raise HarborSuiteError(
                "held-out snapshot lock task entries must be objects"
            )
        entry_id = entry.get("id")
        entry_digest = entry.get("digest")
        if (
            not isinstance(entry_id, str)
            or not isinstance(entry_digest, str)
            or entry_id in lock_task_map
        ):
            raise HarborSuiteError("held-out snapshot lock task set/digest mismatch")
        lock_task_map[entry_id] = entry_digest
    manifest_task_map = {
        str(task["id"]): str(task["digest"]) for task in manifest["tasks"]
    }
    if lock_task_map != manifest_task_map:
        raise HarborSuiteError("held-out archive tasks do not agree with snapshot lock")


def verify_bundle(manifest: dict[str, Any], root: Path) -> None:
    _verify_snapshot_lock(manifest, root)
    dataset_root = _bundle_path(root, manifest["dataset"]["path"])
    dataset_manifest = dataset_root / "dataset.toml"
    _verify_dataset_manifest(manifest, dataset_manifest)
    prompt = _bundle_path(root, manifest["experiment"]["prompt_path"])
    if _digest(prompt) != manifest["experiment"]["prompt_digest"]:
        raise HarborSuiteError("held-out prompt digest mismatch")
    for task in manifest["tasks"]:
        task_root = dataset_root / task["id"]
        actual = "sha256:" + task_digest(task_root).removeprefix("sha256:")
        if actual != task["digest"]:
            raise HarborSuiteError(f"held-out task digest mismatch: {task['id']}")
        for root_key, digest_key in (
            ("verifier_root", "verifier_tree_digest"),
            ("oracle_root", "oracle_tree_digest"),
        ):
            declared = _bundle_path(root, task[root_key])
            if _tree_digest(declared) != task[digest_key]:
                raise HarborSuiteError(
                    f"held-out {root_key} tree digest mismatch: {task['id']}"
                )


def _run_command(
    command: str,
    arguments: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    result = run_operator_command(
        command,
        arguments,
        cwd=cwd,
        timeout_seconds=600.0,
        environment=environment,
    )
    if result.exit_code is None or result.exit_code != 0:
        diagnostic = result.diagnostic or result.stderr.decode("utf-8", "replace")
        raise HarborSuiteError(f"held-out command {command} failed: {diagnostic}")


def fetch_bundle(manifest_uri: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    aws_env = operator_environment(include=_AWS_ENVIRONMENT_VARS)
    manifest_path = output / "manifest.json"
    _run_command(
        "aws",
        ["s3", "cp", manifest_uri, str(manifest_path)],
        cwd=output,
        environment=aws_env,
    )
    manifest = validate_manifest(manifest_path)
    lock_uri = manifest["snapshot_lock"]["lock_uri"]
    lock_path = output / "snapshot-lock.json"
    _run_command(
        "aws",
        ["s3", "cp", lock_uri, str(lock_path)],
        cwd=output,
        environment=aws_env,
    )
    if _digest(lock_path) != manifest["snapshot_lock"]["lock_digest"]:
        raise HarborSuiteError("held-out snapshot lock digest mismatch")
    archive = output / "bundle.tar.gz"
    _run_command(
        "aws",
        ["s3", "cp", manifest["archive"]["uri"], str(archive)],
        cwd=output,
        environment=aws_env,
    )
    if _digest(archive) != manifest["archive"]["sha256"]:
        raise HarborSuiteError("held-out archive digest mismatch")
    extracted = output / "bundle"
    extracted.mkdir()
    _safe_extract(archive, extracted)
    shutil_lock = extracted / "snapshot-lock.json"
    shutil_lock.write_bytes(lock_path.read_bytes())
    verify_bundle(manifest, extracted)
    return extracted


def _compose(image: str) -> dict[str, Any]:
    return {
        "services": {
            "jacobian": {
                "image": image,
                "command": [
                    "--transport",
                    "streamable-http",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--allow-anonymous",
                    "--stateless-http",
                ],
            }
        }
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def render_plan(
    manifest_path: Path,
    bundle_root: Path,
    output: Path,
    stage: str,
    *,
    max_tokens: int,
    max_cost_usd: float,
) -> Path:
    manifest = validate_manifest(manifest_path)
    verify_bundle(manifest, bundle_root)
    experiment = manifest["experiment"]
    if max_tokens != experiment["max_tokens"] or not math_isclose(
        max_cost_usd, experiment["max_cost_usd"]
    ):
        raise HarborSuiteError("runtime budget must exactly match the frozen manifest")
    stage_config = experiment["stages"][stage]
    output.mkdir(parents=True, exist_ok=False)
    manifest_digest = _digest(manifest_path)
    conditions = {item["id"]: item for item in manifest["conditions"]}
    treatment = conditions["C2"]
    compose_path = output / "c2.compose.json"
    compose_path.write_text(
        json.dumps(_compose(treatment["image"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prompt_path = (bundle_root / experiment["prompt_path"]).resolve()
    pairs = [
        (task, repetition)
        for task in stage_config["task_ids"]
        for repetition in range(stage_config["repetitions"])
    ]
    rng = random.Random(experiment["randomization_seed"])
    rng.shuffle(pairs)
    runs: list[dict[str, Any]] = []
    for pair_index, (task, repetition) in enumerate(pairs):
        pair_id = f"{task}-r{repetition + 1:03d}"
        order = ["C1", "C2"]
        rng.shuffle(order)
        for condition_id in order:
            condition = conditions[condition_id]
            run_root = output / "runs" / pair_id / condition_id.lower()
            run_root.mkdir(parents=True)
            job_path = run_root / "job.json"
            jobs_dir = run_root / "results"
            compose = [str(BENCHMARKS / "config" / "agent-eval-proxy.compose.yaml")]
            agent: dict[str, Any] = {
                "name": experiment["agent"]["name"],
                "model_name": experiment["model"],
                "kwargs": {
                    "version": experiment["agent"]["version"],
                    "prompt_template_path": str(prompt_path),
                    "reasoning_effort": experiment["reasoning_effort"],
                },
            }
            if condition["jacobian_enabled"]:
                compose.append(str(compose_path))
                agent["mcp_servers"] = [
                    {
                        "name": "jacobian",
                        "transport": "streamable-http",
                        "url": "http://jacobian:8000/mcp",
                    }
                ]
            job = {
                "jobs_dir": str(jobs_dir),
                "n_attempts": 1,
                "timeout_multiplier": 1,
                "orchestrator": {
                    "type": "local",
                    "n_concurrent_trials": 1,
                    "quiet": False,
                },
                "environment": {
                    "type": "docker",
                    "force_build": True,
                    "delete": True,
                    "extra_docker_compose": compose,
                },
                "agents": [agent],
                "datasets": [
                    {
                        "path": str(bundle_root / manifest["dataset"]["path"]),
                        "task_names": [task],
                    }
                ],
            }
            job_path.write_text(
                json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            snapshot_path = run_root / "runtime.json"
            image_identity = (
                {
                    "source_sha": treatment["source_sha"],
                    "source_dirty": False,
                    "reference": treatment["image"],
                    "digest_reference": treatment["image"],
                    "platform": treatment["platform"],
                    "jacobian_package_version": treatment["server_version"],
                }
                if condition["jacobian_enabled"]
                else None
            )
            runtime_snapshot = {
                "manifest_digest": manifest_digest,
                "condition": condition,
                "stage": stage,
                "pair_id": pair_id,
                "pair_index": pair_index,
                "task": task,
                "repetition": repetition,
            }
            if image_identity is not None:
                runtime_snapshot["jacobian_image"] = image_identity
            snapshot_path.write_text(
                json.dumps(
                    runtime_snapshot,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            runs.append(
                {
                    "pair_id": pair_id,
                    "pair_index": pair_index,
                    "task": task,
                    "repetition": repetition,
                    "condition": condition_id,
                    "jacobian_enabled": condition["jacobian_enabled"],
                    "job": _relative(job_path, output),
                    "runtime_snapshot": _relative(snapshot_path, output),
                    "jobs_dir": _relative(jobs_dir, output),
                }
            )
    plan: dict[str, Any] = {
        "schema_version": "3",
        "manifest_digest": manifest_digest,
        "stage": stage,
        "budget": {
            "max_tokens": max_tokens,
            "max_cost_usd": max_cost_usd,
            "enforcement": "PAIR_BOUNDARY_POST_RUN",
            "missing_accounting": "INCOMPLETE",
            "overage": "INCOMPLETE",
        },
        "pair_count": len(pairs),
        "runs": runs,
    }
    plan["plan_digest"] = _json_digest(plan)
    run_plan = output / "run-plan.json"
    run_plan.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_plan


def math_isclose(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--manifest-uri", required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--manifest", type=Path, required=True)
    render_parser.add_argument("--bundle-root", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--stage", choices=("pilot", "decision"), required=True)
    render_parser.add_argument("--max-tokens", type=int, required=True)
    render_parser.add_argument("--max-cost-usd", type=float, required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    preflight_parser.add_argument("--mcp-url", default="")
    preflight_parser.add_argument("--probe-timeout-seconds", type=float, default=120.0)
    preflight_parser.add_argument("--readiness-retries", type=int, default=3)
    preflight_parser.add_argument(
        "--readiness-retry-delay-seconds", type=float, default=5.0
    )
    control_parser = subparsers.add_parser("control-routing-status")
    control_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        validate_manifest(args.manifest)
        print(args.manifest)
    elif args.command == "fetch":
        print(fetch_bundle(args.manifest_uri, args.output))
    elif args.command == "preflight":
        from benchmarks.tooling.heldout_routing import treatment_readiness_preflight

        contract = treatment_readiness_preflight(
            args.manifest,
            mcp_url=args.mcp_url,
            probe_timeout_seconds=args.probe_timeout_seconds,
            readiness_retries=args.readiness_retries,
            readiness_retry_delay_seconds=args.readiness_retry_delay_seconds,
        )
        print(json.dumps(contract, indent=2, sort_keys=True))
    elif args.command == "control-routing-status":
        from benchmarks.tooling.heldout_routing import control_routing_status

        contract = control_routing_status(args.manifest)
        print(json.dumps(contract, indent=2, sort_keys=True))
    else:
        print(
            render_plan(
                args.manifest,
                args.bundle_root,
                args.output,
                args.stage,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "fetch_bundle",
    "render_plan",
    "validate_manifest",
    "verify_bundle",
]
