"""Render deterministic paired held-out Harbor execution plans."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import BENCHMARKS
from benchmarks.tooling.heldout_integrity import verify_bundle
from benchmarks.tooling.heldout_manifest import _digest, _json_digest, validate_manifest


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
