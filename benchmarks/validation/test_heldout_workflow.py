from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling import heldout_integrity
from benchmarks.tooling.heldout_manifest import _digest
from benchmarks.tooling.heldout_observations import collect_heldout_evidence
from benchmarks.tooling.heldout_plan import render_plan
from benchmarks.tooling.observation_comparison import compare_evidence
from benchmarks.validation.heldout_fixtures import _bundle, _manifest, _write

ROOT = Path(__file__).parents[2]


def test_complete_plan_collects_exact_pairs_and_derives_heldout_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    manifest_path = _write(tmp_path, value)
    monkeypatch.setattr(heldout_integrity, "task_digest", lambda _path: "a" * 64)
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
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
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
                "schema_version": "2",
                "manifest_digest": plan["manifest_digest"],
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
    assert treatment["runtime_snapshot"]["jacobian_image"] == {
        "source_sha": "b" * 40,
        "source_dirty": False,
        "reference": "registry.invalid/jacobian@sha256:" + "4" * 64,
        "digest_reference": "registry.invalid/jacobian@sha256:" + "4" * 64,
        "platform": "linux/amd64",
        "jacobian_package_version": "1.2.3",
    }
    assert report["status"] == "VALID"
    assert report["evidence_class"] == "held-out-comparison"
    assert report["pair_count"] == 9


def test_heldout_workflow_is_main_only_manifest_driven_and_sanitized() -> None:
    workflow = (ROOT / ".github/workflows/heldout-benchmarks.yml").read_text(
        encoding="utf-8"
    )

    assert "github.ref == 'refs/heads/main'" in workflow
    assert "max_tokens:" not in workflow.split("confirmation:", 1)[0]
    assert "--mcp-config" not in workflow
    assert "benchmarks.tooling.heldout_runner" in workflow
    assert "--manifest" in workflow
    assert "--probe-url" in workflow
    assert "routing-status-c1.json" in workflow
    assert "routing-status-c2.json" in workflow
    assert "steps.bundle.outputs.root != ''" in workflow
    assert "path: ${{ steps.bundle.outputs.root }}/sanitized" in workflow
