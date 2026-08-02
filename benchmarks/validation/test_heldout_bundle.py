from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from benchmarks.tooling import heldout_bundle
from benchmarks.tooling.harbor_suite import HarborSuiteError
from benchmarks.tooling.heldout_bundle import (
    _digest,
    _safe_extract,
    _tree_digest,
    render_plan,
    validate_manifest,
    verify_bundle,
)
from benchmarks.tooling.observation_results import (
    collect_heldout_evidence,
    compare_evidence,
)

ROOT = Path(__file__).parents[2]


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
    return {
        "schema_version": "2",
        "bundle_id": "capability-held-out-v1",
        "bundle_version": "1.0.0",
        "archive": {
            "uri": "s3://private-bucket/bundle.tar.gz",
            "sha256": "sha256:" + "d" * 64,
        },
        "dataset": {
            "id": "capability-held-out-v1",
            "path": "dataset",
            "manifest_digest": "sha256:" + "e" * 64,
            "snapshot_id": "sha256:" + "f" * 64,
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
                "server_version": "1.2.3",
                "policy_profile": "DEFAULT",
                "catalog_digest": "sha256:" + "5" * 64,
                "policy_digest": "sha256:" + "6" * 64,
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
        'name = "jacobian/capability-held-out-v1"',
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
    return root


def test_valid_manifest_freezes_c1_c2_and_budget_ladder(tmp_path: Path) -> None:
    manifest = validate_manifest(_write(tmp_path, _manifest()))

    assert manifest["conditions"][0] == {
        "id": "C1",
        "role": "PRIMARY_CONTROL",
        "jacobian_enabled": False,
    }
    assert manifest["experiment"]["stages"]["pilot"]["repetitions"] == 3


def test_manifest_rejects_unknown_stage_task(tmp_path: Path) -> None:
    value = _manifest()
    value["experiment"]["stages"]["pilot"]["task_ids"][0] = "unknown"

    with pytest.raises(HarborSuiteError, match="unknown task ids"):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_control_with_jacobian_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][0]["image"] = "registry.invalid/c1@sha256:" + "1" * 64

    with pytest.raises(HarborSuiteError, match="held-out manifest is invalid"):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_non_digest_pinned_treatment_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][1]["image"] = "registry.invalid/c2:latest"

    with pytest.raises(HarborSuiteError, match="held-out manifest is invalid"):
        validate_manifest(_write(tmp_path, value))


def test_bundle_binds_complete_verifier_and_oracle_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_bundle, "task_digest", lambda _path: "a" * 64)
    verify_bundle(value, root)
    (root / value["tasks"][0]["verifier_root"] / "extra.py").write_text(
        "pass\n", encoding="utf-8"
    )

    with pytest.raises(HarborSuiteError, match="tree digest mismatch"):
        verify_bundle(value, root)


def test_bundle_rejects_dataset_manifest_task_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_bundle, "task_digest", lambda _path: "a" * 64)
    manifest = root / value["dataset"]["path"] / "dataset.toml"
    manifest.write_text(
        '[dataset]\nname = "jacobian/capability-held-out-v1"\n\n'
        '[[tasks]]\nname = "jacobian/held-out-0"\n'
        f'digest = "{value["tasks"][0]["digest"]}"\n',
        encoding="utf-8",
    )
    value["dataset"]["manifest_digest"] = _digest(manifest)

    with pytest.raises(HarborSuiteError, match="manifest task set/digest mismatch"):
        verify_bundle(value, root)


def test_render_expands_stable_pairs_and_keeps_control_jacobian_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    manifest_path = _write(tmp_path, value)
    monkeypatch.setattr(heldout_bundle, "task_digest", lambda _path: "a" * 64)

    plan_path = render_plan(
        manifest_path,
        root,
        tmp_path / "rendered",
        "pilot",
        max_tokens=100000,
        max_cost_usd=100.0,
    )
    plan = json.loads(plan_path.read_text())

    assert plan["pair_count"] == 9
    assert len(plan["runs"]) == 18
    assert len({run["pair_id"] for run in plan["runs"]}) == 9
    assert all(not Path(run["job"]).is_absolute() for run in plan["runs"])
    for run in plan["runs"]:
        job = json.loads((plan_path.parent / run["job"]).read_text())
        assert job["n_attempts"] == 1
        assert len(job["datasets"][0]["task_names"]) == 1
        if run["condition"] == "C1":
            assert run["jacobian_enabled"] is False
            assert "mcp_servers" not in job["agents"][0]
            assert len(job["environment"]["extra_docker_compose"]) == 1
        else:
            assert run["jacobian_enabled"] is True
            assert job["agents"][0]["mcp_servers"][0]["name"] == "jacobian"
            assert len(job["environment"]["extra_docker_compose"]) == 2
    assert plan["budget"]["missing_accounting"] == "INCOMPLETE"


def test_complete_plan_collects_exact_pairs_and_derives_heldout_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    manifest_path = _write(tmp_path, value)
    monkeypatch.setattr(heldout_bundle, "task_digest", lambda _path: "a" * 64)
    plan_path = render_plan(
        manifest_path,
        root,
        tmp_path / "rendered",
        "pilot",
        max_tokens=100000,
        max_cost_usd=100.0,
    )
    plan = json.loads(plan_path.read_text())
    ledger_runs = {}
    for run in plan["runs"]:
        result_root = plan_path.parent / run["jobs_dir"] / "job"
        result_root.mkdir(parents=True)
        result = {
            "stats": {
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
            "trial_results": [
                {
                    "task_name": f"jacobian/{run['task']}",
                    "task_checksum": "sha256:" + "a" * 64,
                    "trial_name": "attempt-0",
                    "agent_info": {
                        "name": "codex",
                        "version": "1.2.3",
                        "model_info": {"name": "model"},
                    },
                    "agent_result": {
                        "n_input_tokens": 10,
                        "n_output_tokens": 5,
                        "cost_usd": 0.01,
                    },
                    "verifier_result": {
                        "rewards": {
                            "correctness": 1.0,
                            "false_certification": 0.0,
                            "reward": 1.0,
                        }
                    },
                    "exception_info": None,
                }
            ],
        }
        result_path = result_root / "result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        ledger_runs[f"{run['pair_id']}/{run['condition']}"] = {
            "status": "COMPLETE",
            "result_digest": _digest(result_path),
        }
    ledger_path = plan_path.parent / "execution-ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "plan_digest": plan["plan_digest"],
                "status": "COMPLETE",
                "runs": ledger_runs,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "benchmarks.tooling.observation_results._git_sha", lambda: "b" * 40
    )

    control, control_failures = collect_heldout_evidence(
        run_plan_path=plan_path,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        condition="C1",
    )
    treatment, treatment_failures = collect_heldout_evidence(
        run_plan_path=plan_path,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        condition="C2",
    )
    report = compare_evidence(control, treatment)

    assert control_failures == treatment_failures == []
    assert len(control["trials"]) == 9
    assert report["status"] == "VALID"
    assert report["evidence_class"] == "held-out-comparison"
    assert report["pair_count"] == 9


def test_private_archive_rejects_workspace_escape(tmp_path: Path) -> None:
    source = tmp_path / "secret.txt"
    source.write_text("oracle", encoding="utf-8")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="../oracle.txt")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(HarborSuiteError, match="escapes output"):
        _safe_extract(archive, output)


def test_heldout_workflow_is_main_only_manifest_driven_and_sanitized() -> None:
    workflow = (ROOT / ".github/workflows/heldout-benchmarks.yml").read_text(
        encoding="utf-8"
    )

    assert "github.ref == 'refs/heads/main'" in workflow
    assert "max_tokens:" not in workflow.split("confirmation:", 1)[0]
    assert "--mcp-config" not in workflow
    assert "benchmarks.tooling.heldout_runner" in workflow
    assert "catalog.catalog_digest" in workflow
    assert "steps.bundle.outputs.root != ''" in workflow
    assert "path: ${{ steps.bundle.outputs.root }}/sanitized" in workflow
