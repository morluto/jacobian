from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from benchmarks.tooling import harbor_suite
from benchmarks.tooling.benchmark_snapshots import (
    build_lock,
    generate_publication,
    load_all_locks,
    lock_digest_of,
    publication_dir,
    render_publication_dataset,
    snapshot_id,
    validate_lock,
)
from benchmarks.tooling.harbor_suite import HarborSuiteError

ROOT = Path(__file__).parents[2]

# A fixed, content-addressed stub task digest keyed only on the task id, so the
# lock is reproducible without Harbor installed.  Mirrors the ``heldout_bundle``
# test pattern of substituting a deterministic digest fn.
_TASK_DIGITS = {
    "alpha-task": "a" * 64,
    "beta-task": "b" * 64,
    "gamma-task": "c" * 64,
}

_TREE_SHA = "89d3f59c5d4a8c3035aa717d162d145094837bcb"


def _stub_digest(task_dir: Path) -> str:
    if task_dir.name in _TASK_DIGITS:
        return _TASK_DIGITS[task_dir.name]
    # Deterministic fallback for tasks added during drift tests.
    return hashlib.sha256(task_dir.name.encode()).hexdigest()


def _clean_git(args: list[str]) -> str:
    if args == ["rev-parse", "HEAD^{tree}"]:
        return _TREE_SHA
    if args == ["status", "--porcelain"]:
        return ""
    if args == ["cat-file", "-t", _TREE_SHA]:
        return "tree"
    if args == ["rev-list", "--all", "--objects"]:
        return f"{_TREE_SHA}\n"
    raise AssertionError(f"unexpected git call: {args}")


def _dirty_git(args: list[str]) -> str:
    if args == ["rev-parse", "HEAD^{tree}"]:
        return _TREE_SHA
    if args == ["status", "--porcelain"]:
        return " M benchmarks/registry.toml\n"
    raise AssertionError(f"unexpected git call: {args}")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _member_toml(
    task_id: str,
    *,
    dataset_name: str = "jacobian/snap-test",
    environment_profile: str = "core-python",
    assurance_ceiling: str = "VERIFIED",
) -> str:
    return f"""
schema_version = "1"
task_id = "{task_id}"
task_name = "jacobian/{task_id}"
evaluation_kind = "workflow"
domain = "mathematical-sciences"
field = "logic-satisfiability"
provenance_class = "hand-designed-structural-variant"
provenance_ref = "Fixed fixture task."
assurance_ceiling = "{assurance_ceiling}"
required_provider = "core"
environment_profile = "{environment_profile}"
verifier_contract_version = "1"
evaluation_owner = "{dataset_name}"
"""


def _build_tree(base: Path) -> dict[str, Path]:
    """Build a hermetic benchmark tree under ``base/repo`` and return its paths."""
    root = base / "repo"
    benchmarks = root / "benchmarks"
    dataset = benchmarks / "datasets" / "snap-test"

    _write(
        benchmarks / "registry.toml",
        """
schema_version = "1"

[[datasets]]
id = "jacobian/snap-test"
directory = "benchmarks/datasets/snap-test"
evaluation_kind = "workflow"
scored = true
publication_status = "local"
required_provider = "core"
runtime_profile = "core"
title = "Snapshot test suite"
purpose = "Hermetic snapshot lock fixture."
claim_class = "workflow-observation"
answer_visibility = "hidden-at-runtime"
default_execution_profile = "oracle-and-observation"
oracle_jobs_dir = "benchmarks/results/snap-test-oracle"
observation_jobs_dir = "benchmarks/results/snap-test"
[datasets.jobs]
oracle = "jobs/oracle.json"
observation = "jobs/observation.json"
""",
    )
    _write(
        benchmarks / "environment-profiles.toml",
        """
schema_version = "1"

[profiles.core-python]
agent_image = "python:3.12-slim@sha256:1111111111111111111111111111111111111111111111111111111111111111"
verifier_image = "python:3.12-slim@sha256:2222222222222222222222222222222222222222222222222222222222222222"
verifier_runtime_digest = "sha256:3333333333333333333333333333333333333333333333333333333333333333"
allow_apt = false

[profiles.uv-provider]
agent_image = "ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:4444444444444444444444444444444444444444444444444444444444444444"
verifier_image = "python:3.12-slim@sha256:2222222222222222222222222222222222222222222222222222222222222222"
verifier_runtime_digest = "sha256:5555555555555555555555555555555555555555555555555555555555555555"
allow_apt = true
""",
    )
    _write(
        dataset / "suite.toml",
        """
schema_version = "2"

[dataset]
id = "jacobian/snap-test"
title = "Snapshot test suite"
purpose = "Hermetic snapshot lock fixture."
claim_class = "workflow-observation"
answer_visibility = "hidden-at-runtime"
default_execution_profile = "oracle-and-observation"
keywords = ["mathematics", "snapshot"]
authors = [{ name = "Jacobian contributors" }]
""",
    )
    _write(dataset / "jobs" / "oracle.json", '{"jobs_dir": "out", "n_attempts": 1}\n')
    _write(
        dataset / "jobs" / "observation.json",
        '{"jobs_dir": "out", "n_attempts": 1}\n',
    )
    for task_id in _TASK_DIGITS:
        _write(
            dataset / task_id / "task.toml",
            f'task = {{ name = "jacobian/{task_id}" }}\n',
        )
        _write(
            dataset / "members" / f"{task_id}.toml",
            _member_toml(task_id),
        )
    return {
        "root": root,
        "benchmarks": benchmarks,
        "dataset": dataset,
        "registry": benchmarks / "registry.toml",
        "profiles": benchmarks / "environment-profiles.toml",
    }


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Build the hermetic tree once and repoint harbor_suite.ROOT at it."""
    tree = _build_tree(tmp_path)
    monkeypatch.setattr(harbor_suite, "ROOT", tree["root"])
    return tree


def _build_from(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return build_lock(
        "jacobian/snap-test",
        harbor_version="0.20.0",
        digest_fn=_stub_digest,
        git_fn=_clean_git,
        profiles_path=paths["profiles"],
        registry_path=paths["registry"],
        **kwargs,
    )


def _write_lock(tmp_path: Path, lock: dict[str, Any]) -> Path:
    path = tmp_path / "snapshot-lock.json"
    path.write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# build_lock
# ---------------------------------------------------------------------------


def test_build_lock_is_content_addressed_and_schema_valid(
    paths: dict[str, Path],
) -> None:
    lock = _build_from(paths)

    assert lock["schema_version"] == "1"
    assert lock["snapshot_id"] == lock["lock_digest"]
    assert lock["snapshot_id"].startswith("sha256:")
    assert lock["lock_digest"] == lock_digest_of(lock)
    assert lock["suite"]["name"] == "jacobian/snap-test"
    assert lock["suite"]["suite_header_digest"].startswith("sha256:")
    assert "version" not in lock["suite"]
    assert lock["harbor_version"] == "0.20.0"
    assert lock["source"]["tree_sha"] == _TREE_SHA
    assert lock["source"]["dirty"] is False
    assert lock["source"]["registry_digest"].startswith("sha256:")
    assert lock["source"]["environment_profiles_digest"].startswith("sha256:")


def test_build_lock_binds_git_tree_not_commit(paths: dict[str, Path]) -> None:
    lock = _build_from(paths)
    assert lock["source"]["tree_sha"] == _TREE_SHA
    # tree_sha is 40-char hex, same length as commit SHA but semantically a tree.
    assert len(lock["source"]["tree_sha"]) == 40


def test_build_lock_fails_closed_on_dirty_tree(paths: dict[str, Path]) -> None:
    with pytest.raises(HarborSuiteError, match="dirty"):
        build_lock(
            "jacobian/snap-test",
            digest_fn=_stub_digest,
            git_fn=_dirty_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
        )


def test_build_lock_accepts_explicit_source_tree(paths: dict[str, Path]) -> None:
    explicit = _TREE_SHA
    lock = build_lock(
        "jacobian/snap-test",
        digest_fn=_stub_digest,
        git_fn=_clean_git,
        profiles_path=paths["profiles"],
        registry_path=paths["registry"],
        source_tree=explicit,
    )
    assert lock["source"]["tree_sha"] == explicit
    assert lock["source"]["dirty"] is False


def test_build_lock_rejects_unreachable_explicit_source_tree(
    paths: dict[str, Path],
) -> None:
    explicit = "abcdef0123456789abcdef0123456789abcdef01"

    def unreachable_git(args: list[str]) -> str:
        if args == ["cat-file", "-t", explicit]:
            return "tree"
        if args == ["rev-list", "--all", "--objects"]:
            return ""
        raise AssertionError(f"unexpected git call: {args}")

    with pytest.raises(HarborSuiteError, match="reachable git tree"):
        build_lock(
            "jacobian/snap-test",
            digest_fn=_stub_digest,
            git_fn=unreachable_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
            source_tree=explicit,
        )


def test_build_lock_orders_tasks_by_id_with_harbor_native_digests(
    paths: dict[str, Path],
) -> None:
    lock = _build_from(paths)

    ids = ["alpha-task", "beta-task", "gamma-task"]
    assert [task["id"] for task in lock["tasks"]] == ids
    assert lock["evaluation"]["task_ids"] == ids
    assert [task["digest"] for task in lock["tasks"]] == [
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
        "sha256:" + "c" * 64,
    ]
    assert all(task["name"] == f"jacobian/{task['id']}" for task in lock["tasks"])
    assert all(task["member_digest"].startswith("sha256:") for task in lock["tasks"])


def test_build_lock_resolves_environment_per_task_from_member_profile(
    paths: dict[str, Path],
) -> None:
    # Override beta's environment_profile to uv-provider.
    _write(
        paths["dataset"] / "members" / "beta-task.toml",
        _member_toml("beta-task", environment_profile="uv-provider"),
    )
    lock = _build_from(paths)

    # Suite-level environment is a summary, not a resolved image profile.
    assert "agent_image" not in lock["environment"]
    assert "profiles" in lock["environment"]
    assert lock["environment"]["profiles"] == ["core-python", "uv-provider"]
    assert lock["environment"]["summary_digest"].startswith("sha256:")

    # Each task carries its own resolved environment.
    alpha = next(t for t in lock["tasks"] if t["id"] == "alpha-task")
    beta = next(t for t in lock["tasks"] if t["id"] == "beta-task")
    assert alpha["environment_profile"] == "core-python"
    assert alpha["environment"]["agent_image"].endswith("1" * 64)
    assert alpha["environment"]["allow_apt"] is False
    assert beta["environment_profile"] == "uv-provider"
    assert beta["environment"]["agent_image"].endswith("4" * 64)
    assert beta["environment"]["allow_apt"] is True


def test_build_lock_suite_runtime_profile_is_label_not_image_profile(
    paths: dict[str, Path],
) -> None:
    lock = _build_from(paths)
    # runtime_profile is "core" (a label), not "core-python" (a profile key).
    assert lock["suite"]["runtime_profile"] == "core"
    # The environment summary uses member environment_profile names, not the
    # suite runtime_profile label.
    assert lock["environment"]["profiles"] == ["core-python"]


def test_build_lock_records_observation_and_oracle_job_digests(
    paths: dict[str, Path],
) -> None:
    lock = _build_from(paths)
    evaluation = lock["evaluation"]
    assert evaluation["oracle_job_digest"].startswith("sha256:")
    assert evaluation["observation_job_digest"].startswith("sha256:")
    assert evaluation["oracle_jobs_dir"] == "benchmarks/results/snap-test-oracle"
    assert evaluation["observation_jobs_dir"] == "benchmarks/results/snap-test"
    assert "compose_file_digest" not in evaluation


def test_build_lock_rejects_unknown_environment_profile(
    paths: dict[str, Path],
) -> None:
    _write(
        paths["dataset"] / "members" / "alpha-task.toml",
        _member_toml("alpha-task", environment_profile="missing"),
    )
    with pytest.raises(HarborSuiteError, match="environment profile"):
        _build_from(paths)


# ---------------------------------------------------------------------------
# validate_lock — historical (default)
# ---------------------------------------------------------------------------


def test_validate_lock_historical_round_trips_against_unchanged_tree(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    assert validate_lock(path) == lock


def test_validate_lock_historical_remains_valid_after_adding_a_task(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)

    # Add a new task to the tree — the old lock must stay historically valid.
    new_id = "delta-task"
    _write(
        paths["dataset"] / new_id / "task.toml",
        f'task = {{ name = "jacobian/{new_id}" }}\n',
    )
    _write(paths["dataset"] / "members" / f"{new_id}.toml", _member_toml(new_id))
    # Historical validation must not re-read the tree.
    assert validate_lock(path) == lock


def test_validate_lock_historical_remains_valid_after_member_drift(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    _write(
        paths["dataset"] / "members" / "alpha-task.toml",
        _member_toml("alpha-task", assurance_ceiling="COMPUTED"),
    )
    assert validate_lock(path) == lock


def test_validate_lock_rejects_tampered_digest(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    lock["lock_digest"] = "sha256:" + "0" * 64
    path = _write_lock(tmp_path, lock)
    with pytest.raises(HarborSuiteError, match="lock_digest is stale"):
        validate_lock(path)


def test_validate_lock_rejects_snapshot_id_not_equal_to_lock_digest(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    lock["snapshot_id"] = "sha256:" + "9" * 64
    path = _write_lock(tmp_path, lock)
    with pytest.raises(HarborSuiteError, match="snapshot_id must equal lock_digest"):
        validate_lock(path)


def test_validate_lock_rejects_unordered_tasks(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    lock["tasks"] = list(reversed(lock["tasks"]))
    path = _write_lock(tmp_path, lock)
    with pytest.raises(HarborSuiteError, match="not ordered by id"):
        validate_lock(path)


def test_validate_lock_rejects_task_ids_mismatch(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    lock["evaluation"]["task_ids"] = list(reversed(lock["evaluation"]["task_ids"]))
    path = _write_lock(tmp_path, lock)
    with pytest.raises(HarborSuiteError, match="task_ids does not match"):
        validate_lock(path)


# ---------------------------------------------------------------------------
# validate_lock — prospective (reproduce=True)
# ---------------------------------------------------------------------------


def test_validate_lock_reproduce_succeeds_against_unchanged_tree(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    assert (
        validate_lock(
            path,
            digest_fn=_stub_digest,
            git_fn=_clean_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
            reproduce=True,
        )
        == lock
    )


def test_validate_lock_reproduce_detects_added_task(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    new_id = "delta-task"
    _write(
        paths["dataset"] / new_id / "task.toml",
        f'task = {{ name = "jacobian/{new_id}" }}\n',
    )
    _write(paths["dataset"] / "members" / f"{new_id}.toml", _member_toml(new_id))
    with pytest.raises(HarborSuiteError, match="no longer reproduces"):
        validate_lock(
            path,
            digest_fn=_stub_digest,
            git_fn=_clean_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
            reproduce=True,
        )


def test_validate_lock_reproduce_detects_member_drift(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    _write(
        paths["dataset"] / "members" / "alpha-task.toml",
        _member_toml("alpha-task", assurance_ceiling="COMPUTED"),
    )
    with pytest.raises(HarborSuiteError, match="no longer reproduces"):
        validate_lock(
            path,
            digest_fn=_stub_digest,
            git_fn=_clean_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
            reproduce=True,
        )


def test_validate_lock_reproduce_detects_profile_drift(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    _write(
        paths["profiles"],
        paths["profiles"]
        .read_text(encoding="utf-8")
        .replace("allow_apt = false", "allow_apt = true"),
    )
    with pytest.raises(HarborSuiteError, match="no longer reproduces"):
        validate_lock(
            path,
            digest_fn=_stub_digest,
            git_fn=_clean_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
            reproduce=True,
        )


def test_validate_lock_reproduce_detects_oracle_job_drift(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    path = _write_lock(tmp_path, lock)
    _write(
        paths["dataset"] / "jobs" / "oracle.json",
        '{"jobs_dir": "out", "n_attempts": 2}\n',
    )
    with pytest.raises(HarborSuiteError, match="no longer reproduces"):
        validate_lock(
            path,
            digest_fn=_stub_digest,
            git_fn=_clean_git,
            profiles_path=paths["profiles"],
            registry_path=paths["registry"],
            reproduce=True,
        )


# ---------------------------------------------------------------------------
# snapshot_id / publication
# ---------------------------------------------------------------------------


def test_snapshot_id_and_publication_dir_are_content_addressed(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    sid = snapshot_id(lock)
    assert sid == lock["lock_digest"].removeprefix("sha256:")
    assert len(sid) == 64
    dest = tmp_path / "dist" / "harbor"
    assert publication_dir(lock, dest) == dest / "snap-test" / sid


def _strip_header(text: str) -> str:
    lines = text.splitlines()
    body_start = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    return "\n".join(lines[body_start:]) + "\n"


def test_generate_publication_writes_frozen_dataset_toml(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    dest = tmp_path / "dist" / "harbor"
    dataset_path = generate_publication(lock, dest_root=dest)
    assert dataset_path == publication_dir(lock, dest) / "dataset.toml"
    assert dataset_path.is_file()
    assert (publication_dir(lock, dest) / "snapshot-lock.json").is_file()

    rendered = render_publication_dataset(lock)
    assert dataset_path.read_text(encoding="utf-8") == rendered
    assert "Generated from benchmark snapshot lock" in rendered
    parsed = tomllib.loads(_strip_header(rendered))
    assert parsed["dataset"]["name"] == "jacobian/snap-test"
    assert parsed["dataset"]["description"] == "Hermetic snapshot lock fixture."
    assert parsed["dataset"]["keywords"] == ["mathematics", "snapshot"]
    assert [task["name"] for task in parsed["tasks"]] == [
        "jacobian/alpha-task",
        "jacobian/beta-task",
        "jacobian/gamma-task",
    ]
    assert parsed["tasks"][0]["digest"] == "sha256:" + "a" * 64
    # Harbor DatasetInfo supports version; the snapshot ID is the content version.
    assert parsed["dataset"]["version"] == lock["snapshot_id"]


def test_generate_publication_is_deterministic_across_writes(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    dest = tmp_path / "dist" / "harbor"
    first = generate_publication(lock, dest_root=dest).read_text(encoding="utf-8")
    second = generate_publication(lock, dest_root=dest).read_text(encoding="utf-8")
    assert first == second


def test_generate_publication_remains_stable_after_tree_drift(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    dest = tmp_path / "dist" / "harbor"
    first = generate_publication(lock, dest_root=dest).read_text(encoding="utf-8")
    # Drift the tree — the publication from the frozen lock must not change.
    _write(
        paths["dataset"] / "members" / "alpha-task.toml",
        _member_toml("alpha-task", assurance_ceiling="COMPUTED"),
    )
    second = generate_publication(lock, dest_root=dest).read_text(encoding="utf-8")
    assert first == second


def test_load_all_locks_discovers_committed_locks_and_filters_invalid(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    # Committed locks live under benchmarks/snapshots/<suite>/<digest>.lock.json.
    snapshots_dir = paths["root"] / "benchmarks" / "snapshots" / "snap-test"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    sid = snapshot_id(lock)
    (snapshots_dir / f"{sid}.lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Corrupt a sibling lock file: it should be skipped, not raise.
    (snapshots_dir / "deadbeef.lock.json").write_text(
        json.dumps({"schema_version": "1", "lock_digest": "sha256:" + "0" * 64}),
        encoding="utf-8",
    )
    discovered = load_all_locks(paths["root"] / "benchmarks" / "snapshots")
    assert len(discovered) == 1
    assert discovered[0]["snapshot_id"] == lock["snapshot_id"]


def test_load_all_locks_ignores_dist_harbor_copies(
    paths: dict[str, Path], tmp_path: Path
) -> None:
    lock = _build_from(paths)
    # Publish into dist/harbor — the snapshot-lock.json copy there must NOT be
    # discovered by load_all_locks, which reads only committed benchmarks/snapshots.
    dest = tmp_path / "dist" / "harbor"
    generate_publication(lock, dest_root=dest)
    # No committed locks under benchmarks/snapshots.
    discovered = load_all_locks(paths["root"] / "benchmarks" / "snapshots")
    assert discovered == ()
    # Even if we point load_all_locks at the dist dir, it must not pick up
    # snapshot-lock.json (it only scans *.lock.json, not snapshot-lock.json).
    dist_discovered = load_all_locks(dest)
    assert dist_discovered == ()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_lock_digest_excludes_identity_fields(paths: dict[str, Path]) -> None:
    lock = _build_from(paths)
    assert lock_digest_of(lock) == lock_digest_of(
        {**lock, "snapshot_id": "sha256:" + "0" * 64}
    )
    body = {k: v for k, v in lock.items() if k not in ("lock_digest", "snapshot_id")}
    assert lock_digest_of(lock) == lock_digest_of(
        body | {"snapshot_id": lock["lock_digest"]}
    )


def test_two_distinct_trees_produce_distinct_snapshot_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree_a = _build_tree(tmp_path / "a")
    monkeypatch.setattr(harbor_suite, "ROOT", tree_a["root"])
    lock_a = _build_from(tree_a)

    tree_b = _build_tree(tmp_path / "b")
    monkeypatch.setattr(harbor_suite, "ROOT", tree_b["root"])
    new_purpose = "A different fixture purpose."
    _write(
        tree_b["dataset"] / "suite.toml",
        (tree_b["dataset"] / "suite.toml")
        .read_text(encoding="utf-8")
        .replace("Hermetic snapshot lock fixture.", new_purpose),
    )
    _write(
        tree_b["registry"],
        tree_b["registry"]
        .read_text(encoding="utf-8")
        .replace("Hermetic snapshot lock fixture.", new_purpose),
    )
    lock_b = _build_from(tree_b)
    assert lock_a["snapshot_id"] != lock_b["snapshot_id"]
