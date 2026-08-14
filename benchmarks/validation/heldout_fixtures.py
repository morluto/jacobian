"""Minimal consistent held-out world for host-side validation tests."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tooling.heldout_bundle import _digest, _tree_digest


def _manifest() -> dict:
    tasks = [
        {
            "id": f"held-out-{index}",
            "family": "family-a" if index < 3 else "family-b",
            "digest": "sha256:" + "a" * 64,
            "verifier_root": f"dataset/held-out-{index}/tests",
            "verifier_tree_digest": "sha256:" + "b" * 64,
            "oracle_root": f"dataset/held-out-{index}/solution",
            "oracle_tree_digest": "sha256:" + "c" * 64,
        }
        for index in range(5)
    ]
    snapshot_id = "sha256:" + "f" * 64
    return {
        "schema_version": "3",
        "bundle_id": "operation-held-out-v1",
        "bundle_version": "1.0.0",
        "snapshot_lock": {
            "lock_id": snapshot_id,
            "lock_uri": "s3://private-bucket/snapshot-lock.json",
            "lock_digest": "sha256:" + "0" * 64,
        },
        "archive": {
            "uri": "s3://private-bucket/bundle.tar.gz",
            "sha256": "sha256:" + "d" * 64,
        },
        "dataset": {
            "id": "operation-held-out-v1",
            "path": "dataset",
            "manifest_digest": "sha256:" + "e" * 64,
            "minimum_independent_families": 2,
        },
        "tasks": tasks,
        "conditions": [
            {"id": "C1", "role": "PRIMARY_CONTROL", "jacobian_enabled": False},
            {
                "id": "C2",
                "role": "PRIMARY_TREATMENT",
                "jacobian_enabled": True,
                "image": "registry.invalid/jacobian@sha256:" + "4" * 64,
                "source_sha": "b" * 40,
                "platform": "linux/amd64",
                "server_version": "1.2.3",
                "catalog_digest": "sha256:" + "5" * 64,
            },
        ],
        "experiment": {
            "harbor_version": "0.20.0",
            "agent": {"name": "codex", "version": "1.2.3"},
            "model": "model",
            "prompt_path": "prompts/heldout.md",
            "prompt_digest": "sha256:" + "7" * 64,
            "reasoning_effort": "high",
            "randomization_seed": 104729,
            "max_tokens": 100000,
            "max_cost_usd": 100.0,
            "stages": {
                "pilot": {
                    "task_ids": [item["id"] for item in tasks[:3]],
                    "repetitions": 3,
                },
                "decision": {
                    "task_ids": [item["id"] for item in tasks],
                    "repetitions": 5,
                },
            },
        },
    }


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _bundle(tmp_path: Path, value: dict) -> Path:
    root = tmp_path / "bundle"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    prompt = root / "prompts" / "heldout.md"
    prompt.parent.mkdir()
    prompt.write_text("{instruction}\n", encoding="utf-8")
    value["experiment"]["prompt_digest"] = _digest(prompt)
    for task in value["tasks"]:
        task_root = dataset / task["id"]
        tests = task_root / "tests"
        solution = task_root / "solution"
        tests.mkdir(parents=True)
        solution.mkdir()
        (tests / "verifier.py").write_text("print('ok')\n", encoding="utf-8")
        (solution / "submission.json").write_text("{}\n", encoding="utf-8")
        task["verifier_tree_digest"] = _tree_digest(tests)
        task["oracle_tree_digest"] = _tree_digest(solution)
    dataset_entries = [
        "[dataset]",
        'name = "jacobian/operation-held-out-v1"',
        "",
    ]
    for task in value["tasks"]:
        dataset_entries.extend(
            [
                "[[tasks]]",
                f'name = "jacobian/{task["id"]}"',
                f'digest = "{task["digest"]}"',
                "",
            ]
        )
    (dataset / "dataset.toml").write_text("\n".join(dataset_entries), encoding="utf-8")
    value["dataset"]["manifest_digest"] = _digest(dataset / "dataset.toml")
    lock = {
        "schema_version": "1",
        "snapshot_id": value["snapshot_lock"]["lock_id"],
        "lock_digest": "sha256:" + "0" * 64,
        "suite": {
            "id": "operation-held-out-v1",
            "name": "jacobian/operation-held-out-v1",
            "title": "Held-out",
            "purpose": "Held-out evaluation",
            "claim_class": "held-out-comparative-evaluation",
            "answer_visibility": "hidden-at-runtime",
            "default_execution_profile": "oracle-and-observation",
            "evaluation_kind": "workflow",
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
                    "agent_image": "registry.invalid/agent@sha256:" + "0" * 64,
                    "verifier_image": "registry.invalid/verifier@sha256:" + "0" * 64,
                    "allow_apt": False,
                },
                "member_digest": "sha256:" + "0" * 64,
            }
            for task in value["tasks"]
        ],
        "evaluation": {
            "task_ids": [task["id"] for task in value["tasks"]],
            "oracle_job_digest": "sha256:" + "0" * 64,
            "oracle_jobs_dir": "jobs/oracle.json",
        },
    }
    lock_path = root / "snapshot-lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    value["snapshot_lock"]["lock_digest"] = _digest(lock_path)
    return root
