"""Immutable content-addressed Harbor benchmark snapshot locks.

A snapshot lock pins one registered Jacobian Harbor dataset to a reproducible,
content-addressed state.  It captures:

* the ordered Harbor-native task digests for every member task;
* the suite-header digest (the canonical ``[dataset]`` table of ``suite.toml``);
* the pinned Harbor version;
* the resolved per-task agent/verifier image and verifier runtime digests,
  resolved from ``benchmarks/environment-profiles.toml`` via each member's
  ``environment_profile`` field (the registry ``runtime_profile`` is an
  evaluation runtime label, not an image-profile key);
* the source git **tree** SHA (not the commit SHA), with ``dirty=false`` — a
  dirty tree is not reproducible and generation fails closed;
* the evaluation split/config (ordered task ids and Oracle/observation job
  digests).

The ``lock_digest`` is the canonical-JSON digest of the lock body with
``lock_digest`` and ``snapshot_id`` removed; ``snapshot_id`` equals
``lock_digest`` and is the content version.

Existing locks are validated historically by default: schema, content address,
internal ordering, and ``dirty=false``.  This keeps a committed lock valid
after membership or task drift — the lock is a pin of the past, not a claim
about the present.  The optional ``reproduce=True`` prospective check re-builds
from the current tree and reports drift.

A publication ``dataset.toml`` can be regenerated deterministically from a lock
into ``dist/harbor/<suite>/<snapshot>/``.

Harbor is not a runtime dependency of this module: the task-digest function is
injectable so tests can substitute a deterministic stub, mirroring the
``heldout_bundle`` pattern.  The default uses Harbor's native
``Task(...).checksum`` through ``harbor_suite.task_digest``.
"""

from __future__ import annotations

import hashlib
import json
import string
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import tomli_w
from jsonschema import Draft202012Validator

from benchmarks.tooling import harbor_suite
from benchmarks.tooling.harbor_suite import (
    EnvironmentProfile,
    HarborSuiteError,
    Suite,
    TaskRef,
    get_suite,
    task_digest,
)

SCHEMA_VERSION = "1"
DEFAULT_HARBOR_VERSION = "0.20.0"
DIGEST_PREFIX = "sha256:"
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "snapshot-lock.schema.json"
)

DigestFn = Callable[[Path], str]
GitFn = Callable[[list[str]], str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Any) -> str:
    return DIGEST_PREFIX + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    try:
        return DIGEST_PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HarborSuiteError(f"unable to digest {path}: {exc}") from exc


def _json_file_digest(path: Path) -> str:
    return _canonical_digest(_read_json(path))


def _toml_file_digest(path: Path) -> str:
    return _canonical_digest(_read_toml(path))


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarborSuiteError(f"{path}: invalid TOML: {exc}") from exc
    if not isinstance(value, dict):
        raise HarborSuiteError(f"{path}: TOML root must be a table")
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"invalid JSON {path}: {exc}") from exc


def _normalize_digest(value: str) -> str:
    """Accept either a bare hex digest or a ``sha256:``-prefixed digest."""
    if value.startswith(DIGEST_PREFIX):
        return value
    return DIGEST_PREFIX + value


def _default_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=harbor_suite.ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarborSuiteError(f"git {' '.join(args)} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Suite / member reading
# ---------------------------------------------------------------------------


def _suite_header(suite: Suite) -> dict[str, Any]:
    raw = _read_toml(suite.suite_manifest)
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise HarborSuiteError(f"{suite.suite_manifest}: [dataset] is required")
    return dataset


def _member_path(suite: Suite, task_id: str) -> Path:
    return suite.path / "members" / f"{task_id}.toml"


def _read_member(suite: Suite, ref: TaskRef) -> dict[str, Any]:
    path = _member_path(suite, ref.path.name)
    if not path.is_file():
        raise HarborSuiteError(
            f"{path.relative_to(harbor_suite.ROOT)}: member fragment is missing "
            f"for {ref.name}"
        )
    return _read_toml(path)


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------


def _environment_dict(profile: EnvironmentProfile) -> dict[str, Any]:
    return {
        "profile": profile.name,
        "agent_image": profile.agent_image,
        "verifier_image": profile.verifier_image,
        "verifier_runtime_digest": profile.verifier_runtime_digest,
        "allow_apt": profile.allow_apt,
    }


def _resolve_task_environment(
    member: Mapping[str, Any],
    profiles: Mapping[str, EnvironmentProfile],
    label: str,
) -> EnvironmentProfile:
    """Resolve a task's environment from its member ``environment_profile``.

    The registry ``runtime_profile`` is an evaluation runtime label, not an
    environment-profiles.toml key.  Each member declares its own
    ``environment_profile``. The named profile is the only image/runtime
    source; member records cannot override digest-pinned environment fields.
    """
    profile_name = member.get("environment_profile")
    if not isinstance(profile_name, str) or not profile_name:
        raise HarborSuiteError(f"{label}: environment_profile is required")
    if profile_name not in profiles:
        raise HarborSuiteError(
            f"{label}: environment profile {profile_name!r} is not defined"
        )
    base = profiles[profile_name]
    agent_image = base.agent_image
    verifier_image = base.verifier_image
    verifier_runtime_digest = base.verifier_runtime_digest
    if "@sha256:" not in agent_image:
        raise HarborSuiteError(f"{label}: agent_image must be a digest-pinned image")
    if "@sha256:" not in verifier_image:
        raise HarborSuiteError(f"{label}: verifier_image must be a digest-pinned image")
    if not verifier_runtime_digest.startswith(DIGEST_PREFIX):
        raise HarborSuiteError(
            f"{label}: verifier_runtime_digest must be a sha256 digest"
        )
    return EnvironmentProfile(
        name=profile_name,
        agent_image=agent_image,
        verifier_image=verifier_image,
        verifier_runtime_digest=verifier_runtime_digest,
        allow_apt=base.allow_apt,
    )


# ---------------------------------------------------------------------------
# Lock body construction
# ---------------------------------------------------------------------------


def _task_records(
    suite: Suite,
    *,
    profiles: Mapping[str, EnvironmentProfile],
    digest_fn: DigestFn,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ref in sorted(suite.tasks, key=lambda item: item.path.name):
        member = _read_member(suite, ref)
        label = str(_member_path(suite, ref.path.name).relative_to(harbor_suite.ROOT))
        env_profile = _resolve_task_environment(member, profiles, label)
        records.append(
            {
                "id": ref.path.name,
                "name": ref.name,
                "digest": _normalize_digest(digest_fn(ref.path)),
                "assurance_ceiling": ref.maximum_assurance,
                "required_provider": ref.required_provider,
                "environment_profile": env_profile.name,
                "environment": _environment_dict(env_profile),
                "member_digest": _canonical_digest(member),
            }
        )
    return records


def _evaluation_record(suite: Suite) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_ids": [
            ref.path.name for ref in sorted(suite.tasks, key=lambda r: r.path.name)
        ],
        "oracle_job_digest": _json_file_digest(suite.job_oracle),
        "oracle_jobs_dir": suite.oracle_jobs_dir,
    }
    if suite.job_observation is not None:
        record["observation_job_digest"] = _json_file_digest(suite.job_observation)
        record["observation_jobs_dir"] = suite.observation_jobs_dir
    if suite.compose_file is not None:
        record["compose_file_digest"] = _file_digest(suite.compose_file)
    return record


def _environment_summary(task_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic suite-level environment summary.

    Image fields are resolved per task from member ``environment_profile``
    records.  The suite-level summary is the set of distinct profile names plus
    a digest over the ordered per-task resolved environments.
    """
    profiles = sorted({t["environment_profile"] for t in task_records})
    envs = [t["environment"] for t in sorted(task_records, key=lambda t: t["id"])]
    return {
        "profiles": profiles,
        "summary_digest": _canonical_digest(envs),
    }


def _resolve_source(*, git_fn: GitFn, source_tree: str | None) -> dict[str, Any]:
    """Resolve the source git tree SHA.

    By default, binds ``HEAD^{tree}`` and fails closed on a dirty worktree — a
    dirty tree is not reproducible.  An explicit ``source_tree`` bypasses the
    git calls for controlled generation (e.g. CI after a clean checkout).
    """
    if source_tree is not None:
        if (
            not isinstance(source_tree, str)
            or len(source_tree) != 40
            or any(char not in string.hexdigits for char in source_tree)
        ):
            raise HarborSuiteError(
                f"source_tree must be a 40-character git tree SHA, got {source_tree!r}"
            )
        try:
            object_type = git_fn(["cat-file", "-t", source_tree]).strip()
            reachable = git_fn(["rev-list", "--all", "--objects"])
        except HarborSuiteError as exc:
            raise HarborSuiteError(
                f"source_tree must identify a reachable git tree: {source_tree}"
            ) from exc
        if object_type != "tree" or not any(
            line.split(maxsplit=1)[0] == source_tree
            for line in reachable.splitlines()
            if line.strip()
        ):
            raise HarborSuiteError(
                f"source_tree must identify a reachable git tree: {source_tree}"
            )
        return {"tree_sha": source_tree, "dirty": False}
    tree_sha = git_fn(["rev-parse", "HEAD^{tree}"])
    dirty = bool(git_fn(["status", "--porcelain"]))
    if dirty:
        raise HarborSuiteError(
            "source tree is dirty; commit or stash changes before generating a "
            "snapshot lock, or pass an explicit source_tree for controlled generation"
        )
    return {"tree_sha": tree_sha, "dirty": False}


def _lock_body(
    suite: Suite,
    *,
    harbor_version: str,
    profiles: Mapping[str, EnvironmentProfile],
    digest_fn: DigestFn,
    git_fn: GitFn,
    source_tree: str | None,
    registry_path: Path,
    profiles_path: Path,
) -> dict[str, Any]:
    header = _suite_header(suite)
    task_records = _task_records(suite, profiles=profiles, digest_fn=digest_fn)
    suite_section: dict[str, Any] = {
        "id": suite.id,
        "name": suite.dataset_name,
        "title": suite.title,
        "purpose": suite.purpose,
        "claim_class": suite.claim_class,
        "answer_visibility": suite.answer_visibility,
        "default_execution_profile": suite.default_execution_profile,
        "evaluation_kind": suite.evaluation_kind,
        "publication_status": suite.publication_status,
        "scored": suite.scored,
        "required_provider": suite.required_provider,
        "runtime_profile": suite.runtime_profile,
        "suite_header_digest": _canonical_digest(header),
    }
    if isinstance(header.get("keywords"), list):
        suite_section["keywords"] = header["keywords"]
    if isinstance(header.get("authors"), list):
        suite_section["authors"] = header["authors"]
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite_section,
        "harbor_version": harbor_version,
        "source": {
            **_resolve_source(git_fn=git_fn, source_tree=source_tree),
            "registry_digest": _toml_file_digest(registry_path),
            "environment_profiles_digest": _toml_file_digest(profiles_path),
        },
        "environment": _environment_summary(task_records),
        "tasks": task_records,
        "evaluation": _evaluation_record(suite),
    }


def _strip_identity(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return the lock body without the content-addressed identity fields."""
    return {k: v for k, v in lock.items() if k not in ("lock_digest", "snapshot_id")}


def _with_digest(body: dict[str, Any]) -> dict[str, Any]:
    digest = _canonical_digest(_strip_identity(body))
    return {**body, "snapshot_id": digest, "lock_digest": digest}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_schema(lock: Mapping[str, Any]) -> None:
    schema = _read_json(SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(lock),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        messages = [
            f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        ]
        raise HarborSuiteError(
            "benchmark snapshot lock is invalid:\n" + "\n".join(messages)
        )


def _validate_internal(lock: Mapping[str, Any]) -> None:
    """Verify internal consistency: ordering, identity, and content address.

    This is historical validation — it does not re-read the current tree.  A
    committed lock remains intrinsically valid after membership or task drift.
    """
    tasks = lock.get("tasks", [])
    ids = [t["id"] for t in tasks]
    if ids != sorted(ids):
        raise HarborSuiteError("snapshot lock tasks are not ordered by id")
    if len(set(ids)) != len(ids):
        raise HarborSuiteError("snapshot lock has duplicate task ids")
    eval_ids = lock.get("evaluation", {}).get("task_ids", [])
    if eval_ids != ids:
        raise HarborSuiteError("evaluation.task_ids does not match task ids")
    expected = lock_digest_of(lock)
    if lock.get("lock_digest") != expected:
        raise HarborSuiteError(
            f"lock_digest is stale (recorded {lock.get('lock_digest')!r}, "
            f"recomputed {expected!r})"
        )
    if lock.get("snapshot_id") != expected:
        raise HarborSuiteError("snapshot_id must equal lock_digest")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_lock(
    dataset: str,
    *,
    harbor_version: str = DEFAULT_HARBOR_VERSION,
    digest_fn: DigestFn | None = None,
    profiles_path: Path | None = None,
    registry_path: Path | None = None,
    git_fn: GitFn | None = None,
    source_tree: str | None = None,
) -> dict[str, Any]:
    """Build a content-addressed snapshot lock for one registered dataset.

    The source must be clean (``dirty=false``); a dirty worktree fails closed.
    Pass ``source_tree`` to bypass git for controlled generation.

    ``digest_fn`` defaults to Harbor's native task digest and may be substituted
    in tests.  ``git_fn`` defaults to a ``git`` subprocess against the
    repository root.
    """
    if registry_path is None:
        registry_path = harbor_suite.REGISTRY_PATH
    if profiles_path is None:
        profiles_path = harbor_suite.ENVIRONMENT_PROFILES_PATH
    suite = get_suite(dataset, path=registry_path)
    profiles = harbor_suite.load_environment_profiles(profiles_path)
    body = _lock_body(
        suite,
        harbor_version=harbor_version,
        profiles=profiles,
        digest_fn=digest_fn or task_digest,
        git_fn=git_fn or _default_git,
        source_tree=source_tree,
        registry_path=registry_path,
        profiles_path=profiles_path,
    )
    lock = _with_digest(body)
    _validate_schema(lock)
    _validate_internal(lock)
    return lock


def lock_digest_of(lock: Mapping[str, Any]) -> str:
    """Recompute the canonical lock digest from a lock body."""
    return _canonical_digest(_strip_identity(lock))


def validate_lock(
    path: Path,
    *,
    digest_fn: DigestFn | None = None,
    profiles_path: Path | None = None,
    registry_path: Path | None = None,
    git_fn: GitFn | None = None,
    source_tree: str | None = None,
    reproduce: bool = False,
) -> dict[str, Any]:
    """Validate an existing immutable snapshot lock.

    By default (``reproduce=False``) this is **historical** validation: schema,
    content address, internal ordering, and ``dirty=false``.  It does not
    re-read the current tree, so a committed lock remains valid after
    membership or task drift.

    With ``reproduce=True`` (the **prospective** check), the lock is re-built
    from the current tree and compared field-by-field; any drift is reported.
    Use this before regenerating from the live tree, not for validating
    committed historical locks.
    """
    lock = _read_json(path)
    if not isinstance(lock, dict):
        raise HarborSuiteError(f"{path}: lock root must be a JSON object")
    _validate_schema(lock)
    _validate_internal(lock)
    if not reproduce:
        return lock

    suite_name = lock["suite"]["name"]
    harbor_version = lock["harbor_version"]
    rebuilt = build_lock(
        suite_name,
        harbor_version=harbor_version,
        digest_fn=digest_fn,
        profiles_path=profiles_path,
        registry_path=registry_path,
        git_fn=git_fn,
        source_tree=source_tree,
    )
    if _strip_identity(rebuilt) != _strip_identity(lock):
        raise HarborSuiteError(
            f"{path}: snapshot lock no longer reproduces from the current tree; "
            "the pinned benchmark state has drifted"
        )
    return lock


def snapshot_id(lock: Mapping[str, Any]) -> str:
    """Return the content-addressed snapshot id (the lock digest hex)."""
    value = lock.get("lock_digest")
    if not isinstance(value, str) or not value.startswith(DIGEST_PREFIX):
        raise HarborSuiteError("lock is missing a valid lock_digest")
    return value.removeprefix(DIGEST_PREFIX)


def publication_dir(lock: Mapping[str, Any], dest_root: Path | None = None) -> Path:
    if dest_root is None:
        dest_root = harbor_suite.ROOT / "dist" / "harbor"
    return dest_root / lock["suite"]["id"] / snapshot_id(lock)


def render_publication_dataset(lock: Mapping[str, Any]) -> str:
    """Render the Harbor publication ``dataset.toml`` from a frozen lock.

    The publication manifest uses the lock's frozen Harbor-native task digests
    so it stays stable regardless of later tree drift.  Task entries are
    ordered by task id.  The ``snapshot_id`` is the content version.
    """
    suite = lock["suite"]
    dataset: dict[str, Any] = {
        "name": suite["name"],
        "version": lock["snapshot_id"],
        "description": suite["purpose"],
        "keywords": suite.get("keywords", []),
        "authors": suite.get("authors", [{"name": "Jacobian contributors"}]),
    }
    value: dict[str, Any] = {
        "dataset": dataset,
        "tasks": [
            {"name": task["name"], "digest": task["digest"]}
            for task in sorted(lock["tasks"], key=lambda item: item["id"])
        ],
    }
    body = tomli_w.dumps(value)
    header = (
        f"# Generated from benchmark snapshot lock {lock['snapshot_id']}.\n"
        f"# Suite: {suite['name']}; Harbor {lock['harbor_version']}; "
        f"tree {lock['source']['tree_sha']}.\n"
        "# Do not edit by hand: regenerate via "
        "tools/manage_benchmark_snapshots.py publish.\n"
    )
    return header + body


def generate_publication(
    lock: Mapping[str, Any], dest_root: Path | None = None
) -> Path:
    """Write the publication ``dataset.toml`` for a lock under ``dest_root``.

    Returns the path of the written ``dataset.toml``.  The lock itself is also
    copied alongside as ``snapshot-lock.json`` so the publication directory is
    self-describing.
    """
    target = publication_dir(lock, dest_root)
    target.mkdir(parents=True, exist_ok=True)
    dataset_path = target / "dataset.toml"
    dataset_path.write_text(render_publication_dataset(lock), encoding="utf-8")
    (target / "snapshot-lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def load_all_locks(
    lock_dir: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Discover committed snapshot locks under ``benchmarks/snapshots``.

    Committed locks are the source of truth and live at
    ``benchmarks/snapshots/<suite>/<digest>.lock.json``.  The ``dist/harbor``
    publication directory is git-ignored output and is **never** scanned here:
    a published ``snapshot-lock.json`` copy is a convenience artifact, not a
    committed lock.  Locks that fail internal integrity are skipped silently.
    """
    if lock_dir is None:
        lock_dir = harbor_suite.ROOT / "benchmarks" / "snapshots"
    if not lock_dir.is_dir():
        return ()
    locks: list[dict[str, Any]] = []
    for path in sorted(lock_dir.rglob("*.lock.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            lock = _read_json(path)
        except HarborSuiteError:
            continue
        if not isinstance(lock, dict):
            continue
        try:
            _validate_schema(lock)
            _validate_internal(lock)
        except HarborSuiteError:
            continue
        if lock.get("snapshot_id") == lock.get("lock_digest"):
            locks.append(lock)
    return tuple(locks)


__all__ = [
    "DEFAULT_HARBOR_VERSION",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "build_lock",
    "generate_publication",
    "load_all_locks",
    "lock_digest_of",
    "publication_dir",
    "render_publication_dataset",
    "snapshot_id",
    "validate_lock",
]
