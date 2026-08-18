"""Run a zero-model-cost smoke against a synthetic non-mathematical bundle."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tools.command_runner import run_operator_command

from benchmarks.tooling.harbor_suite import task_digest
from benchmarks.tooling.heldout_manifest import (
    _digest,
    _json_digest,
    _tree_digest,
)
from benchmarks.tooling.heldout_plan import render_plan

PYTHON_IMAGE = (
    "python:3.12-slim@sha256:"
    "57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _task(root: Path, task_id: str) -> None:
    _write(
        root / "task.toml",
        f"""schema_version = "1.4"
artifacts = ["/app/submission.txt"]

[task]
name = "jacobian/{task_id}"
version = "1.0.0"
description = "Copy a frozen color token into the submission file."
keywords = ["infrastructure", "smoke", "copy"]

[metadata]
evaluation_kind = "infrastructure-smoke"
domain = "software-engineering"
field = "benchmark-infrastructure"
assurance_ceiling = "UNVERIFIED"
answer_visibility = "hidden-at-runtime"
author_name = "Jacobian contributors"
difficulty = "easy"
category = "infrastructure"
tags = ["offline", "non-mathematical"]

[agent]
timeout_sec = 60.0

[verifier]
timeout_sec = 60.0
environment_mode = "separate"

[environment]
network_mode = "no-network"
cpus = 1
memory_mb = 256
storage_mb = 1024

[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 256
storage_mb = 1024
""",
    )
    _write(
        root / "instruction.md",
        "Read `token.txt` and copy its one-word value to `/app/submission.txt`.\n",
    )
    _write(
        root / "environment" / "Dockerfile",
        f"FROM {PYTHON_IMAGE}\nWORKDIR /app\nCOPY token.txt /app/token.txt\n",
    )
    _write(root / "environment" / "token.txt", "blue\n")
    _write(
        root / "tests" / "Dockerfile",
        f"FROM {PYTHON_IMAGE}\nCOPY test.sh verifier.py /tests/\nRUN chmod +x /tests/test.sh\n",
    )
    _write(
        root / "tests" / "test.sh", "#!/bin/sh\nset -eu\npython /tests/verifier.py\n"
    )
    _write(
        root / "tests" / "verifier.py",
        """import json
from pathlib import Path

correct = Path("/app/submission.txt").read_text().strip() == "blue"
logs = Path("/logs/verifier")
logs.mkdir(parents=True, exist_ok=True)
(logs / "reward.json").write_text(json.dumps({"reward": float(correct)}))
(logs / "reward-details.json").write_text(json.dumps({"correctness": float(correct), "false_certification": 0.0}))
""",
    )
    _write(
        root / "solution" / "solve.sh",
        "#!/bin/sh\nset -eu\nprintf 'blue\\n' > /app/submission.txt\n",
    )


def create_bundle(root: Path) -> tuple[Path, Path]:
    dataset = root / "bundle" / "dataset"
    tasks: list[dict[str, str]] = []
    for index in range(5):
        task_id = f"copy-token-{index}"
        task_root = dataset / task_id
        _task(task_root, task_id)
        tasks.append(
            {
                "id": task_id,
                "family": "copy-a" if index < 3 else "copy-b",
                "digest": "sha256:" + task_digest(task_root).removeprefix("sha256:"),
                "verifier_root": f"dataset/{task_id}/tests",
                "verifier_tree_digest": _tree_digest(task_root / "tests"),
                "oracle_root": f"dataset/{task_id}/solution",
                "oracle_tree_digest": _tree_digest(task_root / "solution"),
            }
        )
    dataset_toml = [
        "[dataset]",
        'name = "jacobian/heldout-smoke-v1"',
        'description = "Synthetic non-mathematical held-out infrastructure smoke."',
        'keywords = ["infrastructure", "smoke"]',
        'authors = [{ name = "Jacobian contributors" }]',
        "",
    ]
    for task in tasks:
        dataset_toml.extend(
            [
                "[[tasks]]",
                f'name = "jacobian/{task["id"]}"',
                f'digest = "{task["digest"]}"',
                "",
            ]
        )
    _write(dataset / "dataset.toml", "\n".join(dataset_toml))
    dataset_manifest_digest = _digest(dataset / "dataset.toml")
    snapshot_id = _json_digest(
        {
            "dataset_manifest_digest": dataset_manifest_digest,
            "task_digests": [task["digest"] for task in tasks],
        }
    )
    lock = {
        "schema_version": "1",
        "snapshot_id": snapshot_id,
        "lock_digest": "sha256:" + "0" * 64,
        "suite": {
            "id": "heldout-smoke-v1",
            "name": "jacobian/heldout-smoke-v1",
            "title": "Held-out smoke",
            "purpose": "Synthetic infrastructure smoke.",
            "claim_class": "infrastructure-smoke",
            "answer_visibility": "hidden-at-runtime",
            "default_execution_profile": "oracle-only",
            "evaluation_kind": "infrastructure-smoke",
            "publication_status": "local",
            "scored": True,
            "required_provider": "core",
            "runtime_profile": "core",
            "suite_header_digest": "sha256:" + "0" * 64,
        },
        "harbor_version": "0.20.0",
        "source": {
            "tree_sha": "0" * 40,
            "dirty": False,
            "registry_digest": "sha256:" + "0" * 64,
            "environment_profiles_digest": "sha256:" + "0" * 64,
        },
        "environment": {
            "profiles": ["core"],
            "summary_digest": "sha256:" + "0" * 64,
        },
        "tasks": [
            {
                "id": task["id"],
                "name": f"jacobian/{task['id']}",
                "digest": task["digest"],
                "assurance_ceiling": "UNVERIFIED",
                "required_provider": "core",
                "environment_profile": "core",
                "environment": {
                    "profile": "core",
                    "agent_image": PYTHON_IMAGE,
                    "verifier_image": PYTHON_IMAGE,
                    "allow_apt": False,
                },
                "member_digest": "sha256:" + "0" * 64,
            }
            for task in tasks
        ],
        "evaluation": {
            "task_ids": [task["id"] for task in tasks],
            "oracle_job_digest": "sha256:" + "0" * 64,
            "oracle_jobs_dir": "jobs/oracle.json",
        },
    }
    lock_path = root / "bundle" / "snapshot-lock.json"
    _write(lock_path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    lock_digest = _digest(lock_path)
    lock["lock_digest"] = lock_digest
    _write(lock_path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    lock_digest = _digest(lock_path)
    prompt = root / "bundle" / "prompts" / "heldout.md"
    _write(prompt, "{instruction}\n")
    manifest = {
        "schema_version": "3",
        "bundle_id": "heldout-smoke-v1",
        "bundle_version": "1.0.0",
        "snapshot_lock": {
            "lock_id": snapshot_id,
            "lock_uri": "s3://invalid/snapshot-lock.json",
            "lock_digest": lock_digest,
        },
        "archive": {
            "uri": "s3://invalid/heldout-smoke.tar.gz",
            "sha256": "sha256:" + "0" * 64,
        },
        "dataset": {
            "id": "heldout-smoke-v1",
            "path": "dataset",
            "manifest_digest": dataset_manifest_digest,
            "minimum_independent_families": 2,
        },
        "tasks": tasks,
        "conditions": [
            {"id": "C1", "role": "PRIMARY_CONTROL", "jacobian_enabled": False},
            {
                "id": "C2",
                "role": "PRIMARY_TREATMENT",
                "jacobian_enabled": True,
                "image": "registry.invalid/jacobian@sha256:" + "1" * 64,
                "source_sha": "a" * 40,
                "platform": "linux/amd64",
                "server_version": "0.0.0",
                "catalog_digest": "sha256:" + "2" * 64,
            },
        ],
        "experiment": {
            "harbor_version": "0.20.0",
            "agent": {"name": "codex", "version": "0.0.0"},
            "model": "offline-smoke",
            "prompt_path": "prompts/heldout.md",
            "prompt_digest": _digest(prompt),
            "reasoning_effort": "low",
            "randomization_seed": 104729,
            "max_tokens": 100,
            "max_cost_usd": 1.0,
            "stages": {
                "pilot": {
                    "task_ids": [task["id"] for task in tasks[:3]],
                    "repetitions": 3,
                },
                "decision": {
                    "task_ids": [task["id"] for task in tasks],
                    "repetitions": 5,
                },
            },
        },
    }
    manifest_path = root / "manifest.json"
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, root / "bundle"


def smoke(*, run_harbor: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="jacobian-heldout-smoke-") as directory:
        root = Path(directory)
        manifest, bundle = create_bundle(root)
        plan_path = render_plan(
            manifest,
            bundle,
            root / "rendered",
            "pilot",
            max_tokens=100,
            max_cost_usd=1.0,
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["pair_count"] != 9 or len(plan["runs"]) != 18:
            raise RuntimeError("synthetic held-out plan did not expand into nine pairs")
        if run_harbor:
            dataset = bundle / "dataset"
            task = "copy-token-0"
            common_args = [
                "--from",
                "harbor==0.20.0",
                "harbor",
                "run",
                "-p",
                str(dataset),
                "--include-task-name",
                task,
                "--jobs-dir",
                str(root / "harbor-results"),
                "--yes",
            ]
            for extra in (
                ["--agent", "nop", "--disable-verification"],
                ["--agent", "oracle"],
            ):
                result = run_operator_command(
                    "uvx",
                    [*common_args, *extra],
                    cwd=root,
                    timeout_seconds=600.0,
                )
                if result.exit_code is None or result.exit_code != 0:
                    raise RuntimeError(
                        f"harbor smoke command failed: {result.diagnostic or result.stderr}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-harbor", action="store_true")
    args = parser.parse_args()
    smoke(run_harbor=args.run_harbor)
    print("held-out smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["create_bundle", "smoke"]
