from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from benchmarks.tooling import heldout_bundle
from benchmarks.tooling.command_runner import operator_environment
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.heldout_bundle import (
    _AWS_ENVIRONMENT_VARS,
    _digest,
    _safe_extract,
    _tree_digest,
    render_plan,
    treatment_readiness_preflight,
    validate_manifest,
    verify_bundle,
)
from benchmarks.tooling.observation_comparison import compare_evidence
from benchmarks.tooling.observation_results import (
    _heldout_plan_failures,
    _mark_invoked_if_capability_used,
    collect_heldout_evidence,
)

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("runs", (["bad-run"], None))
def test_heldout_evidence_rejects_malformed_run_entries(runs: object) -> None:
    selected, failures = _heldout_plan_failures(
        {
            "runs": runs,
            "pair_count": 1,
            "manifest_digest": "sha256:" + "a" * 64,
            "plan_digest": "sha256:" + "b" * 64,
        },
        {
            "plan_digest": "sha256:" + "b" * 64,
            "manifest_digest": "sha256:" + "a" * 64,
            "status": "COMPLETE",
        },
        "C1",
    )

    assert selected == []
    assert any("held-out plan runs" in failure for failure in failures)


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
        "bundle_id": "capability-held-out-v1",
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
            "id": "capability-held-out-v1",
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
                "reasoning_log_mode": "OFF",
                "image": "registry.invalid/jacobian@sha256:" + "4" * 64,
                "source_sha": "b" * 40,
                "platform": "linux/amd64",
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
    lock = {
        "schema_version": "1",
        "snapshot_id": value["snapshot_lock"]["lock_id"],
        "lock_digest": "sha256:" + "0" * 64,
        "suite": {
            "id": "capability-held-out-v1",
            "name": "jacobian/capability-held-out-v1",
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

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_non_digest_pinned_treatment_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][1]["image"] = "registry.invalid/c2:latest"

    with pytest.raises(HarborSuiteError, match="held-out manifest is invalid"):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_v2_schema_version(tmp_path: Path) -> None:
    value = _manifest()
    value["schema_version"] = "2"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_extra_top_level_field(tmp_path: Path) -> None:
    value = _manifest()
    value["legacy_provenance"] = "should be rejected"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_extra_nested_field(tmp_path: Path) -> None:
    value = _manifest()
    value["snapshot_lock"]["extra_field"] = "rejected"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_string_coercion_of_integer(tmp_path: Path) -> None:
    value = _manifest()
    value["experiment"]["max_tokens"] = "100000"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_bool_coercion_of_integer(tmp_path: Path) -> None:
    value = _manifest()
    value["dataset"]["minimum_independent_families"] = True

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_wrong_root_type(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(path)


def test_manifest_rejects_missing_required_field(tmp_path: Path) -> None:
    value = _manifest()
    del value["snapshot_lock"]

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_wrong_condition_id(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][0]["id"] = "C3"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_wrong_agent_name(tmp_path: Path) -> None:
    value = _manifest()
    value["experiment"]["agent"]["name"] = "claude"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
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

    assert plan["schema_version"] == "3"
    assert plan["manifest_digest"] == _digest(manifest_path)
    assert plan["pair_count"] == 9
    assert len(plan["runs"]) == 18
    assert len({run["pair_id"] for run in plan["runs"]}) == 9
    assert all(not Path(run["job"]).is_absolute() for run in plan["runs"])
    for run in plan["runs"]:
        job = json.loads((plan_path.parent / run["job"]).read_text())
        runtime = json.loads((plan_path.parent / run["runtime_snapshot"]).read_text())
        assert "manifest_digest" in runtime
        assert "harbor_version" not in runtime
        assert "model" not in runtime
        assert job["n_attempts"] == 1
        assert len(job["datasets"][0]["task_names"]) == 1
        if run["condition"] == "C1":
            assert run["jacobian_enabled"] is False
            assert "mcp_servers" not in job["agents"][0]
            assert len(job["environment"]["extra_docker_compose"]) == 1
        else:
            assert run["jacobian_enabled"] is True
            assert runtime["jacobian_image"] == {
                "source_sha": "b" * 40,
                "source_dirty": False,
                "reference": "registry.invalid/jacobian@sha256:" + "4" * 64,
                "digest_reference": "registry.invalid/jacobian@sha256:" + "4" * 64,
                "platform": "linux/amd64",
                "jacobian_package_version": "1.2.3",
            }
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


def test_bundle_rejects_missing_snapshot_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_bundle, "task_digest", lambda _path: "a" * 64)
    (root / "snapshot-lock.json").unlink()

    with pytest.raises(HarborSuiteError, match=r"missing snapshot-lock\.json"):
        verify_bundle(value, root)


def test_bundle_rejects_snapshot_lock_task_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_bundle, "task_digest", lambda _path: "a" * 64)
    lock_path = root / "snapshot-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["tasks"][0]["digest"] = "sha256:" + "z" * 64
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    value["snapshot_lock"]["lock_digest"] = _digest(lock_path)

    with pytest.raises(HarborSuiteError, match="do not agree with snapshot lock"):
        verify_bundle(value, root)


def _ready_probe(
    *, mcp_url, expected_version, expected_policy_profile, timeout_seconds
):
    return {
        "reachable": True,
        "report": {
            "server": {"name": "jacobian", "version": "1.2.3"},
            "tool_names": ["math.find", "math.run"],
            "catalog": {
                "catalog_version": "1",
                "capabilities": 1,
                "policy_profile": "DEFAULT",
                "catalog_digest": "sha256:" + "5" * 64,
                "policy_digest": "sha256:" + "6" * 64,
                "sha256": "abc",
            },
            "discovery": {"bytes": 100, "matches": ["cap-1"]},
        },
    }


def _unreachable_probe(
    *, mcp_url, expected_version, expected_policy_profile, timeout_seconds
):
    return {"reachable": False, "diagnostic": "connection refused"}


def test_treatment_readiness_preflight_ready_with_successful_probe(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=_ready_probe,
    )

    assert contract["infrastructure_status"] == "READY"
    assert contract["routing_status"] == "AVAILABLE_UNUSED"
    assert contract["manifest_digest"] == _digest(manifest_path)
    assert contract["condition_id"] == "C2"
    assert contract["checks"]["image_digest_pinned"] is True
    assert contract["checks"]["catalog_digest_bound"] is True
    assert contract["checks"]["policy_digest_bound"] is True
    assert contract["checks"]["server_version_bound"] is True
    assert contract["checks"]["policy_profile_bound"] is True
    assert contract["checks"]["server_version_match"] is True
    assert contract["checks"]["catalog_digest_match"] is True
    assert contract["checks"]["policy_digest_match"] is True
    assert contract["checks"]["required_tools_present"] is True
    assert contract["checks"]["describe_responded"] is True
    assert contract["failures"] == []
    assert contract["probe"]["reachable"] is True
    assert contract["probe"]["server_version_observed"] == "1.2.3"
    assert contract["probe"]["catalog_digest_observed"] == "sha256:" + "5" * 64


def test_treatment_readiness_preflight_fail_closed_without_probe_url(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = treatment_readiness_preflight(manifest_path)

    assert contract["infrastructure_status"] == "MISCONFIGURED"
    assert contract["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert any("probe URL is not configured" in f for f in contract["failures"])
    assert contract["probe"]["reachable"] is False


def test_treatment_readiness_preflight_unavailable_when_probe_fails(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=_unreachable_probe,
        readiness_retries=0,
    )

    assert contract["infrastructure_status"] == "UNAVAILABLE"
    assert contract["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert any("not reachable" in f for f in contract["failures"])
    assert contract["probe"]["reachable"] is False
    assert contract["probe"]["diagnostic"] == "connection refused"


def test_treatment_readiness_preflight_retries_until_probe_succeeds(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    call_count = 0

    def eventually_ready(
        *, mcp_url, expected_version, expected_policy_profile, timeout_seconds
    ):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {"reachable": False, "diagnostic": "connection refused"}
        return _ready_probe(
            mcp_url=mcp_url,
            expected_version=expected_version,
            expected_policy_profile=expected_policy_profile,
            timeout_seconds=timeout_seconds,
        )

    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=eventually_ready,
        readiness_retries=5,
        readiness_retry_delay_seconds=0,
    )

    assert contract["infrastructure_status"] == "READY"
    assert contract["routing_status"] == "AVAILABLE_UNUSED"
    assert contract["probe"]["reachable"] is True
    assert call_count == 3


def test_treatment_readiness_preflight_exhausts_retries_and_fails_closed(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    call_count = 0

    def always_unreachable(
        *, mcp_url, expected_version, expected_policy_profile, timeout_seconds
    ):
        nonlocal call_count
        call_count += 1
        return {"reachable": False, "diagnostic": "connection refused"}

    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=always_unreachable,
        readiness_retries=3,
        readiness_retry_delay_seconds=0,
    )

    assert contract["infrastructure_status"] == "UNAVAILABLE"
    assert contract["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert contract["probe"]["reachable"] is False
    assert call_count == 4


def test_treatment_readiness_preflight_misconfigured_on_digest_mismatch(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)

    def mismatched_probe(
        *, mcp_url, expected_version, expected_policy_profile, timeout_seconds
    ):
        return {
            "reachable": True,
            "report": {
                "server": {"name": "jacobian", "version": "1.2.3"},
                "tool_names": ["math.find", "math.run"],
                "catalog": {
                    "catalog_version": "1",
                    "capabilities": 1,
                    "policy_profile": "DEFAULT",
                    "catalog_digest": "sha256:" + "9" * 64,
                    "policy_digest": "sha256:" + "6" * 64,
                    "sha256": "abc",
                },
                "discovery": {"bytes": 100, "matches": ["cap-1"]},
            },
        }

    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=mismatched_probe,
    )

    assert contract["infrastructure_status"] == "MISCONFIGURED"
    assert contract["routing_status"] == "MISROUTED"
    assert contract["checks"]["catalog_digest_match"] is False
    assert any("catalog_digest" in f for f in contract["failures"])


def test_control_routing_status_is_not_configured(tmp_path: Path) -> None:
    from benchmarks.tooling.heldout_bundle import control_routing_status

    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = control_routing_status(manifest_path)

    assert contract["condition_id"] == "C1"
    assert contract["infrastructure_status"] == "NOT_CONFIGURED"
    assert contract["routing_status"] == "NOT_APPLICABLE"
    assert contract["treatment"] is None
    assert contract["routing"] is None
    assert contract["probe"] is None
    assert contract["failures"] == []


def test_treatment_readiness_preflight_fails_for_unpinned_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][1]["image"] = "registry.invalid/jacobian:latest"
    manifest_path = _write(tmp_path, value)

    with pytest.raises(HarborSuiteError, match="held-out manifest is invalid"):
        treatment_readiness_preflight(manifest_path)


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
    assert "--manifest" in workflow
    assert "--probe-url" in workflow
    assert "routing-status-c1.json" in workflow
    assert "routing-status-c2.json" in workflow
    assert "steps.bundle.outputs.root != ''" in workflow
    assert "path: ${{ steps.bundle.outputs.root }}/sanitized" in workflow


def _c2_routing_contract(routing_status: str = "AVAILABLE_UNUSED") -> dict:
    return {
        "schema_version": "2",
        "manifest_digest": "sha256:" + "a" * 64,
        "condition_id": "C2",
        "infrastructure_status": "READY",
        "routing_status": routing_status,
        "treatment": {
            "image": "registry.invalid/jacobian@sha256:" + "1" * 64,
            "server_version": "1.0.0",
            "policy_profile": "DEFAULT",
            "catalog_digest": "sha256:" + "2" * 64,
            "policy_digest": "sha256:" + "3" * 64,
        },
        "routing": {"compose_file": "c2.compose.json", "mcp_url": "http://x/mcp"},
        "probe": {
            "reachable": True,
            "server_version_observed": "1.0.0",
            "catalog_digest_observed": "sha256:" + "2" * 64,
            "policy_digest_observed": "sha256:" + "3" * 64,
            "policy_profile_observed": "DEFAULT",
            "tool_names": ["math.find", "math.run"],
            "discovery_matches": ["cap-1"],
            "probe_digest": "sha256:" + "0" * 64,
            "diagnostic": None,
        },
        "checks": {
            "image_digest_pinned": True,
            "catalog_digest_bound": True,
            "policy_digest_bound": True,
            "server_version_bound": True,
            "policy_profile_bound": True,
            "server_version_match": True,
            "catalog_digest_match": True,
            "policy_digest_match": True,
            "required_tools_present": True,
            "describe_responded": True,
        },
        "failures": [],
    }


def test_mark_invoked_transitions_on_successful_capability_invoke(
    tmp_path: Path,
) -> None:
    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_capability_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_INVOKED"
    assert (tmp_path / "routing-status-c2.json").is_file()


def test_mark_invoked_fail_closed_on_errored_invocation(tmp_path: Path) -> None:
    """A failed/errored math.run must not transition to AVAILABLE_INVOKED."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 2,
        }
    ]
    _mark_invoked_if_capability_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_UNUSED"
    assert not (tmp_path / "routing-status-c2.json").exists()


def test_mark_invoked_fail_closed_on_non_completed_trial(tmp_path: Path) -> None:
    """A non-COMPLETED trial with math.run must not transition."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "ERROR",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_capability_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_UNUSED"


def test_mark_invoked_fail_closed_on_timeout_trial(tmp_path: Path) -> None:
    """A timed-out trial with math.run must not transition."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "TIMEOUT",
            "tool_calls": {"math.run": 3},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_capability_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_UNUSED"


def test_mark_invoked_no_transition_when_already_invoked(tmp_path: Path) -> None:
    """If routing_status is already AVAILABLE_INVOKED, do not re-write."""

    ledger = {"routing_status": {"C2": _c2_routing_contract("AVAILABLE_INVOKED")}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_capability_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_INVOKED"
    assert not (tmp_path / "routing-status-c2.json").exists()


def test_mark_invoked_mixed_trials_one_success_transitions(tmp_path: Path) -> None:
    """One successful invocation among errored trials is enough to transition."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 2,
        },
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        },
    ]
    _mark_invoked_if_capability_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_INVOKED"


def test_aws_environment_vars_include_only_credentials_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("UNRELATED_SECRET", "should-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = operator_environment(include=_AWS_ENVIRONMENT_VARS)

    assert env["AWS_ACCESS_KEY_ID"] == "AKIATEST"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert env["AWS_SESSION_TOKEN"] == "token"
    assert env["AWS_REGION"] == "us-east-1"
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert "UNRELATED_SECRET" not in env


def test_aws_environment_vars_exclude_non_aws_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("DATABASE_URL", "postgres://should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")

    env = operator_environment(include=_AWS_ENVIRONMENT_VARS)

    assert env["AWS_ACCESS_KEY_ID"] == "AKIATEST"
    assert "DATABASE_URL" not in env
    assert "OPENAI_API_KEY" not in env


def test_fetch_bundle_passes_aws_environment_to_s3_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LEAKED_VAR", "should-not-appear")

    captured_envs: list[object] = []
    manifest = _manifest()

    def fake_run_command(
        command: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout_seconds: float = 600.0,
        environment: object | None = None,
    ) -> object:
        captured_envs.append(environment)
        dest = Path(arguments[-1])
        if dest.suffix == ".json":
            dest.write_text("{}", encoding="utf-8")
        else:
            dest.write_text("fake", encoding="utf-8")
        return type(
            "Result",
            (),
            {"exit_code": 0, "diagnostic": None, "stderr": b""},
        )()

    monkeypatch.setattr(heldout_bundle, "run_operator_command", fake_run_command)
    monkeypatch.setattr(heldout_bundle, "validate_manifest", lambda _p: manifest)
    monkeypatch.setattr(heldout_bundle, "verify_bundle", lambda _m, _r: None)
    monkeypatch.setattr(heldout_bundle, "_safe_extract", lambda _a, _o: None)

    def fake_digest(path: Path) -> str:
        name = Path(path).name
        if name == "snapshot-lock.json":
            return manifest["snapshot_lock"]["lock_digest"]
        if name == "bundle.tar.gz":
            return manifest["archive"]["sha256"]
        return "sha256:" + "0" * 64

    monkeypatch.setattr(heldout_bundle, "_digest", fake_digest)

    heldout_bundle.fetch_bundle("s3://bucket/manifest.json", tmp_path / "out")

    assert len(captured_envs) == 3
    for env in captured_envs:
        env_dict = dict(env) if env is not None else {}
        assert env_dict.get("AWS_ACCESS_KEY_ID") == "AKIATEST"
        assert env_dict.get("AWS_SECRET_ACCESS_KEY") == "secret"
        assert env_dict.get("AWS_REGION") == "us-east-1"
        assert "LEAKED_VAR" not in env_dict
