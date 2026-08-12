"""Planner and topology digest contract tests."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
from benchmarks.validation._planner_helpers import (
    PLANNER_DIGEST_SOURCES,
    ROOT,
    _assert_plan_valid,
    _build_temp_topology,
    _load_script,
    planner,
)


@pytest.mark.parametrize("preserve_existing", [False, True])
def test_load_script_scopes_sys_modules_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preserve_existing: bool,
) -> None:
    module_name = "_benchmark_planner_module_probe"
    source = tmp_path / "module_probe.py"
    source.write_text(
        "import sys\nregistered_while_loading = sys.modules[__name__]\n",
        encoding="utf-8",
    )
    sentinel = ModuleType("sentinel")
    if preserve_existing:
        monkeypatch.setitem(sys.modules, module_name, sentinel)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    loaded = _load_script(module_name, source)

    assert vars(loaded)["registered_while_loading"] is loaded
    if preserve_existing:
        assert sys.modules[module_name] is sentinel
    else:
        assert module_name not in sys.modules


def test_plan_is_versioned_and_bound_to_event_base_head_sha() -> None:
    base = "0" * 40
    head = "1" * 40
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py"
        ],
        event="pull_request",
        base=base,
        head=head,
    )

    assert result["benchmark-plan-version"] == "2"
    assert result["benchmark-plan-event"] == "pull_request"
    assert result["benchmark-plan-base-sha"] == base
    assert result["benchmark-plan-head-sha"] == head
    assert result["benchmark-planner-digest"].startswith("sha256:")
    assert len(result["benchmark-planner-digest"]) == 71
    assert result["benchmark-topology-digest"].startswith("sha256:")
    assert len(result["benchmark-topology-digest"]) == 71
    _assert_plan_valid(result)


def test_planner_digest_binds_to_planner_and_path_policy_sources() -> None:
    payload = "\n".join(
        f"{path.relative_to(ROOT).as_posix()}\t{path.read_bytes().hex()}"
        for path in PLANNER_DIGEST_SOURCES
    ).encode()
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py"
        ],
        event="pull_request",
    )
    assert result["benchmark-planner-digest"] == expected


def test_member_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bench, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (suite.path / "members" / "alpha-task.toml").write_text(
        'task_id = "alpha-task"\nassurance_ceiling = "VERIFIED"\n',
        encoding="utf-8",
    )
    after = planner._topology_digest([suite])

    assert before.startswith("sha256:")
    assert after.startswith("sha256:")
    assert before != after


def test_environment_profile_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bench, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (bench / "environment-profiles.toml").write_text(
        '[profiles.default]\nimage = "changed"\n', encoding="utf-8"
    )
    after = planner._topology_digest([suite])

    assert before != after


def test_suite_manifest_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (suite.suite_manifest).write_text(
        'schema_version = "2"\n[dataset]\nid = "jacobian/alpha-v1"\ntitle = "Changed"\n',
        encoding="utf-8",
    )
    after = planner._topology_digest([suite])

    assert before != after


def test_registry_change_alters_topology_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bench, suite = _build_temp_topology(tmp_path)
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suites = {"alpha-v1": suite}
    by_task = {"alpha-task": [("alpha-v1", Path("alpha-task"))]}
    monkeypatch.setattr(planner, "_membership", lambda: (by_task, suites))

    before = planner._topology_digest([suite])

    (bench / "registry.toml").write_text(
        'schema_version = "1"\n[[datasets]]\nid = "jacobian/alpha-v1"\n',
        encoding="utf-8",
    )
    after = planner._topology_digest([suite])

    assert before != after
