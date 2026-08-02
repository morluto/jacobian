"""Validate, fetch, and render private held-out Harbor evaluation bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.harbor_suite import BENCHMARKS, HarborSuiteError, task_digest


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


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
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
    assert isinstance(manifest, dict)
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


def verify_bundle(manifest: dict[str, Any], root: Path) -> None:
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


def fetch_bundle(manifest_uri: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    subprocess.run(["aws", "s3", "cp", manifest_uri, str(manifest_path)], check=True)
    manifest = validate_manifest(manifest_path)
    archive = output / "bundle.tar.gz"
    subprocess.run(
        ["aws", "s3", "cp", manifest["archive"]["uri"], str(archive)], check=True
    )
    if _digest(archive) != manifest["archive"]["sha256"]:
        raise HarborSuiteError("held-out archive digest mismatch")
    extracted = output / "bundle"
    extracted.mkdir()
    _safe_extract(archive, extracted)
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
                    "--state-dir",
                    "/state",
                ],
                "volumes": ["jacobian-state:/state"],
            }
        },
        "volumes": {"jacobian-state": {}},
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
            snapshot_path.write_text(
                json.dumps(
                    {
                        "bundle_id": manifest["bundle_id"],
                        "bundle_version": manifest["bundle_version"],
                        "bundle_manifest_digest": _digest(manifest_path),
                        "dataset_manifest_digest": manifest["dataset"][
                            "manifest_digest"
                        ],
                        "snapshot_id": manifest["dataset"]["snapshot_id"],
                        "condition": condition,
                        "harbor_version": experiment["harbor_version"],
                        "agent": experiment["agent"],
                        "model": experiment["model"],
                        "prompt_path": experiment["prompt_path"],
                        "prompt_digest": experiment["prompt_digest"],
                        "reasoning_effort": experiment["reasoning_effort"],
                        "randomization_seed": experiment["randomization_seed"],
                        "stage": stage,
                        "pair_id": pair_id,
                        "pair_index": pair_index,
                        "task": task,
                        "repetition": repetition,
                        "max_tokens": max_tokens,
                        "max_cost_usd": max_cost_usd,
                    },
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
        "schema_version": "2",
        "stage": stage,
        "bundle_manifest_digest": _digest(manifest_path),
        "snapshot_id": manifest["dataset"]["snapshot_id"],
        "harbor_version": experiment["harbor_version"],
        "agent": experiment["agent"],
        "model": experiment["model"],
        "prompt_path": experiment["prompt_path"],
        "prompt_digest": experiment["prompt_digest"],
        "reasoning_effort": experiment["reasoning_effort"],
        "randomization_seed": experiment["randomization_seed"],
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
    args = parser.parse_args()
    if args.command == "validate":
        validate_manifest(args.manifest)
        print(args.manifest)
    elif args.command == "fetch":
        print(fetch_bundle(args.manifest_uri, args.output))
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


__all__ = ["fetch_bundle", "render_plan", "validate_manifest", "verify_bundle"]
