from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling import heldout_integrity
from benchmarks.tooling.heldout_manifest import _digest
from benchmarks.tooling.heldout_plan import render_plan
from benchmarks.validation.heldout_fixtures import _bundle, _manifest, _write


def test_render_expands_stable_pairs_and_keeps_control_jacobian_free(
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
