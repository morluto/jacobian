"""Repository-owned control plane for Jacobian's Harbor datasets.

Each dataset is a Harbor-local dataset directory whose executable task bundles
live directly under ``benchmarks/datasets/<dataset>/``.  Authoritative
``members/*.toml`` records bind those bundles to provenance, execution, and
verification contracts.  Mutable Harbor publication manifests are generated
only from immutable snapshot locks and never live in dataset roots.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from benchmarks.tooling.harbor_digest import (
    HarborDigestError,
)
from benchmarks.tooling.harbor_digest import (
    task_digest as _native_task_digest,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
REGISTRY_PATH = BENCHMARKS / "registry.toml"
ENVIRONMENT_PROFILES_PATH = BENCHMARKS / "environment-profiles.toml"
DATASET_PREFIX = "jacobian/"
DIGEST_PREFIX = "sha256:"
TASK_SCHEMA_VERSION = "1.4"
REQUIRED_METADATA = {
    "evaluation_kind",
    "domain",
    "field",
    "assurance_ceiling",
    "answer_visibility",
    "provenance_class",
    "fixture_digest",
    "required_provider",
}
REQUIRED_ENVIRONMENT = ("Dockerfile", "input.json", "submission_schema.json")
REQUIRED_TESTS = ("Dockerfile", "test.sh", "verifier.py", "verifier_support.py")
DATASET_SUPPORT_DIRS = frozenset({"jobs", "members"})
MEMBER_SCHEMA_VERSION = "1"
VERIFIER_CONTRACT_VERSION = "1"
NETWORK_MODES = frozenset({"public", "no-network", "allowlist"})
FORBIDDEN_VISIBLE_NAMES = frozenset(
    {
        "answer.txt",
        "authorized_record.json",
        "authorized_records.json",
        "expected.json",
        "submission.json",
        "verification-record.json",
        "verification_record.json",
        "verifier.py",
        "verifier_support.py",
    }
)
_HOST_PATH = re.compile(r"(?:^|[\s\"'=])/(?:home|Users|root)\b")
_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|passwd|token|private[_-]?key)"
    r"\s*[:=]\s*[\"'][A-Za-z0-9_\-./+=]{12,}[\"']"
)
_FLOATING = re.compile(r"\bpip(?:3)?\s+install\s+([^\s#|&;]+)")


class HarborSuiteError(ValueError):
    """A registry, suite, task, or generated-artifact contract failure."""


@dataclasses.dataclass(frozen=True)
class TaskRef:
    name: str
    path: Path
    maximum_assurance: str
    required_provider: str
    evaluation_kind: str
    domain: str
    field: str
    provenance_class: str
    provenance_ref: str
    environment_profile: str
    verifier_contract_version: str
    evaluation_owner: str


@dataclasses.dataclass(frozen=True)
class Suite:
    id: str
    dataset_name: str
    path: Path
    suite_manifest: Path
    tasks_dir: Path
    job_oracle: Path
    job_observation: Path | None
    compose_file: Path | None
    oracle_jobs_dir: str
    observation_jobs_dir: str
    evaluation_kind: str
    scored: bool
    publication_status: str
    required_provider: str
    runtime_profile: str
    title: str
    purpose: str
    claim_class: str
    answer_visibility: str
    default_execution_profile: str
    tasks: tuple[TaskRef, ...]

    @property
    def dataset_short_name(self) -> str:
        return self.id

    def dataset_path(self) -> str:
        return self.tasks_dir.relative_to(ROOT).as_posix()


@dataclasses.dataclass(frozen=True)
class TaskDigest:
    short_name: str
    full_name: str
    digest: str


@dataclasses.dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    agent_image: str
    verifier_image: str
    verifier_runtime_digest: str
    allow_apt: bool


def _dataset_task_directories(root: Path) -> set[Path]:
    """Return Harbor task bundles directly contained by one dataset."""

    if not root.exists():
        return set()
    if root.is_symlink() or not root.is_dir():
        raise HarborSuiteError(f"dataset task root must be a directory: {root}")

    tasks: set[Path] = set()
    for entry in sorted(root.iterdir()):
        if entry.is_symlink():
            raise HarborSuiteError(f"dataset contains a symlink: {entry}")
        if not entry.is_dir():
            continue
        manifest = entry / "task.toml"
        if not manifest.exists():
            nested = sorted(entry.rglob("task.toml"))
            if nested:
                raise HarborSuiteError(
                    "Harbor task bundles must be direct children of the dataset: "
                    + ", ".join(str(path) for path in nested)
                )
            if entry.name not in DATASET_SUPPORT_DIRS:
                raise HarborSuiteError(
                    f"dataset contains a non-task directory: {entry}"
                )
            continue
        if manifest.is_symlink() or not manifest.is_file():
            raise HarborSuiteError(f"task manifest is invalid: {manifest}")
        nested = sorted(
            candidate for candidate in entry.rglob("task.toml") if candidate != manifest
        )
        if nested:
            raise HarborSuiteError(
                "Harbor task bundles must be one directory deep: "
                + ", ".join(str(path) for path in nested)
            )
        tasks.add(entry.resolve())
    return tasks


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else (base / path).absolute()
    # Resolve containment only after checking the lexical path.  Calling
    # ``resolve`` first would hide a symlink in a declared task path, while
    # task bundles are deliberately symlink-free.
    current = candidate
    while current != current.parent:
        if current.is_symlink():
            raise HarborSuiteError(f"symlink path is forbidden: {value!r}")
        current = current.parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise HarborSuiteError(f"path escapes repository: {value!r}") from exc
    return resolved


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarborSuiteError(f"{label} must be a non-empty string")
    return value


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarborSuiteError(
            f"{path.relative_to(ROOT)}: invalid TOML: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HarborSuiteError(f"{path.relative_to(ROOT)}: TOML root must be a table")
    return value


def load_environment_profiles(
    path: Path = ENVIRONMENT_PROFILES_PATH,
) -> dict[str, EnvironmentProfile]:
    raw = _read_toml(path)
    if raw.get("schema_version") != "1":
        raise HarborSuiteError(f"{path.relative_to(ROOT)}: schema_version must be '1'")
    values = raw.get("profiles")
    if not isinstance(values, dict) or not values:
        raise HarborSuiteError(f"{path.relative_to(ROOT)}: [profiles] are required")
    profiles: dict[str, EnvironmentProfile] = {}
    for name, value in values.items():
        if not isinstance(value, dict):
            raise HarborSuiteError(
                f"{path.relative_to(ROOT)}: profile {name} is invalid"
            )
        agent_image = _require_string(
            value.get("agent_image"), f"profile {name} agent_image"
        )
        verifier_image = _require_string(
            value.get("verifier_image"), f"profile {name} verifier_image"
        )
        for label, image in (("agent", agent_image), ("verifier", verifier_image)):
            if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) is None:
                raise HarborSuiteError(
                    f"{path.relative_to(ROOT)}: profile {name} {label} image "
                    "must be digest-pinned"
                )
        runtime_digest = _require_string(
            value.get("verifier_runtime_digest"),
            f"profile {name} verifier_runtime_digest",
        )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_digest) is None:
            raise HarborSuiteError(
                f"{path.relative_to(ROOT)}: profile {name} verifier runtime digest is invalid"
            )
        profiles[name] = EnvironmentProfile(
            name=name,
            agent_image=agent_image,
            verifier_image=verifier_image,
            verifier_runtime_digest=runtime_digest,
            allow_apt=value.get("allow_apt") is True,
        )
    return profiles


def _task_ref(
    *,
    task_id: str,
    task_path: Path,
    assurance: Any,
    provider: Any,
    member: dict[str, Any],
    task_root: Path,
    label: str,
) -> TaskRef:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", task_id):
        raise HarborSuiteError(f"{label}: invalid canonical task id {task_id!r}")
    if task_path.name != task_id or task_path.parent != task_root:
        raise HarborSuiteError(f"{label}: task path must be a direct Harbor task")
    if not task_path.is_dir() or not (task_path / "task.toml").is_file():
        raise HarborSuiteError(f"{label}: Harbor task is missing: {task_id}")
    return TaskRef(
        name=f"jacobian/{task_id}",
        path=task_path,
        maximum_assurance=_require_string(assurance, f"{label} assurance_ceiling"),
        required_provider=str(provider or "core"),
        evaluation_kind=_require_string(
            member.get("evaluation_kind"), f"{label} evaluation_kind"
        ),
        domain=_require_string(member.get("domain"), f"{label} domain"),
        field=_require_string(member.get("field"), f"{label} field"),
        provenance_class=_require_string(
            member.get("provenance_class"), f"{label} provenance_class"
        ),
        provenance_ref=_require_string(
            member.get("provenance_ref"), f"{label} provenance_ref"
        ),
        environment_profile=_require_string(
            member.get("environment_profile"), f"{label} environment_profile"
        ),
        verifier_contract_version=_require_string(
            member.get("verifier_contract_version"),
            f"{label} verifier_contract_version",
        ),
        evaluation_owner=_require_string(
            member.get("evaluation_owner"), f"{label} evaluation_owner"
        ),
    )


def _parse_member_file(suite: Suite, member_file: Path) -> TaskRef:
    if member_file.is_symlink():
        raise HarborSuiteError(
            f"{member_file.relative_to(ROOT)}: member symlink is forbidden"
        )
    member = _read_toml(member_file)
    if member.get("schema_version") != MEMBER_SCHEMA_VERSION:
        raise HarborSuiteError(
            f"{member_file.relative_to(ROOT)}: schema_version must be "
            f"{MEMBER_SCHEMA_VERSION!r}"
        )
    task_id = _require_string(member.get("task_id"), f"{member_file.name} task_id")
    ref = _task_ref(
        task_id=task_id,
        task_path=_resolve(task_id, suite.path),
        assurance=member.get("assurance_ceiling"),
        provider=member.get("required_provider", "core"),
        member=member,
        task_root=suite.path,
        label=str(member_file.relative_to(ROOT)),
    )
    if member.get("task_name") != ref.name:
        raise HarborSuiteError(
            f"{member_file.relative_to(ROOT)}: task_name must be {ref.name!r}"
        )
    if ref.evaluation_owner != suite.dataset_name:
        raise HarborSuiteError(
            f"{member_file.relative_to(ROOT)}: evaluation_owner must be "
            f"{suite.dataset_name!r}"
        )
    if ref.verifier_contract_version != VERIFIER_CONTRACT_VERSION:
        raise HarborSuiteError(
            f"{member_file.relative_to(ROOT)}: unsupported verifier contract "
            f"{ref.verifier_contract_version!r}"
        )
    return ref


def _parse_suite_manifest(
    suite: Suite,
) -> tuple[dict[str, Any], tuple[TaskRef, ...]]:
    """Parse the v2 dataset header and canonical-task member fragments."""

    raw = _read_toml(suite.suite_manifest)
    if raw.get("schema_version") != "2":
        raise HarborSuiteError(f"{suite.suite_manifest}: schema_version must be '2'")
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise HarborSuiteError(f"{suite.suite_manifest}: [dataset] is required")
    if dataset.get("id") != suite.dataset_name:
        raise HarborSuiteError(
            f"{suite.suite_manifest}: dataset.id disagrees with registry"
        )
    if "tasks" in raw or "fields" in raw:
        raise HarborSuiteError(
            f"{suite.suite_manifest}: membership belongs in members/*.toml"
        )
    members_dir = suite.path / "members"
    if not members_dir.is_dir():
        raise HarborSuiteError(f"{suite.id}: members directory is missing")
    if members_dir.is_symlink():
        raise HarborSuiteError(f"{suite.id}: members directory symlink is forbidden")
    refs: list[TaskRef] = []
    names: set[str] = set()
    for member_file in sorted(members_dir.glob("*.toml")):
        ref = _parse_member_file(suite, member_file)
        if ref.name in names:
            raise HarborSuiteError(f"{suite.id}: duplicate task id {ref.path.name}")
        names.add(ref.name)
        refs.append(ref)
    return dataset, tuple(refs)


def validate_global_task_ids(suites: tuple[Suite, ...] | list[Suite]) -> None:
    owners: dict[str, str] = {}
    for suite in suites:
        for task in suite.tasks:
            task_id = task.path.name
            previous = owners.setdefault(task_id, suite.id)
            if previous != suite.id:
                raise HarborSuiteError(
                    f"global task id {task_id!r} belongs to both {previous} and {suite.id}"
                )


def _registry_suite(entry: Any, ids: set[str]) -> Suite:
    if not isinstance(entry, dict):
        raise HarborSuiteError("registry dataset entries must be tables")
    dataset_name = _require_string(entry.get("id"), "registry dataset id")
    if not dataset_name.startswith(DATASET_PREFIX):
        raise HarborSuiteError(f"dataset id must start with {DATASET_PREFIX!r}")
    short_id = dataset_name.removeprefix(DATASET_PREFIX)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", short_id):
        raise HarborSuiteError(f"invalid dataset id {dataset_name!r}")
    if short_id in ids:
        raise HarborSuiteError(f"duplicate dataset id {dataset_name!r}")
    ids.add(short_id)
    suite_path = _resolve(
        _require_string(entry.get("directory"), f"{short_id} directory"), ROOT
    )
    if not suite_path.is_dir():
        raise HarborSuiteError(f"dataset directory missing: {suite_path}")
    jobs = entry.get("jobs")
    if not isinstance(jobs, dict):
        raise HarborSuiteError(f"{short_id}: jobs table is required")
    observation_value = jobs.get("observation")
    compose_value = entry.get("compose_file")
    return Suite(
        id=short_id,
        dataset_name=dataset_name,
        path=suite_path,
        suite_manifest=_resolve(
            str(entry.get("suite_manifest", "suite.toml")), suite_path
        ),
        tasks_dir=suite_path,
        job_oracle=_resolve(
            _require_string(jobs.get("oracle"), f"{short_id} oracle job"),
            suite_path,
        ),
        job_observation=(
            _resolve(str(observation_value), suite_path) if observation_value else None
        ),
        compose_file=(
            _resolve(str(compose_value), suite_path) if compose_value else None
        ),
        oracle_jobs_dir=str(
            entry.get("oracle_jobs_dir", f"benchmarks/results/{short_id}-oracle")
        ),
        observation_jobs_dir=str(
            entry.get("observation_jobs_dir", f"benchmarks/results/{short_id}")
        ),
        evaluation_kind=_require_string(
            entry.get("evaluation_kind"), f"{short_id} evaluation_kind"
        ),
        scored=bool(entry.get("scored", False)),
        publication_status=_require_string(
            entry.get("publication_status"), f"{short_id} publication_status"
        ),
        required_provider=_require_string(
            entry.get("required_provider"), f"{short_id} required_provider"
        ),
        runtime_profile=_require_string(
            entry.get("runtime_profile"), f"{short_id} runtime_profile"
        ),
        title=str(entry.get("title", short_id)),
        purpose=str(entry.get("purpose", "")),
        claim_class=str(entry.get("claim_class", "diagnostic")),
        answer_visibility=str(entry.get("answer_visibility", "public")),
        default_execution_profile=str(
            entry.get("default_execution_profile", "oracle-only")
        ),
        tasks=(),
    )


def _validate_loaded_suite(suite: Suite, dataset: dict[str, Any]) -> None:
    if (
        dataset.get("title") != suite.title
        or dataset.get("purpose") != suite.purpose
        or any(
            key in dataset and dataset[key] != expected
            for key, expected in (
                ("claim_class", suite.claim_class),
                ("answer_visibility", suite.answer_visibility),
                ("default_execution_profile", suite.default_execution_profile),
            )
        )
    ):
        raise HarborSuiteError(f"{suite.id}: suite metadata disagrees with registry")
    legacy_manifest = suite.path / "dataset.toml"
    if legacy_manifest.exists():
        raise HarborSuiteError(
            f"{legacy_manifest.relative_to(ROOT)}: mutable dataset manifests are "
            "forbidden; generate publication output from a snapshot lock"
        )
    discovered = _dataset_task_directories(suite.path)
    missing = sorted(discovered - {ref.path.resolve() for ref in suite.tasks})
    if missing:
        raise HarborSuiteError(
            f"{suite.id}: Harbor task is not assigned in members/*.toml: "
            + ", ".join(path.name for path in missing)
        )


def load_registry(path: Path = REGISTRY_PATH) -> tuple[Suite, ...]:
    raw = _read_toml(path)
    if raw.get("schema_version") != "1":
        raise HarborSuiteError("registry schema_version must be '1'")
    entries = raw.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise HarborSuiteError("registry must contain [[datasets]] entries")
    suites: list[Suite] = []
    ids: set[str] = set()
    for entry in entries:
        suite = _registry_suite(entry, ids)
        dataset, tasks = _parse_suite_manifest(suite)
        object.__setattr__(suite, "tasks", tasks)
        _validate_loaded_suite(suite, dataset)
        suites.append(suite)
    validate_global_task_ids(suites)
    return tuple(suites)


def get_suite(dataset: str, *, path: Path = REGISTRY_PATH) -> Suite:
    short = dataset.removeprefix(DATASET_PREFIX)
    for suite in load_registry(path):
        if suite.id == short or suite.dataset_name == dataset:
            return suite
    raise HarborSuiteError(f"unknown dataset {dataset!r}")


def select_task_refs(suite: Suite, task_names: tuple[str, ...]) -> tuple[TaskRef, ...]:
    """Resolve an explicit, duplicate-free leaf selection for one dataset."""

    if not task_names:
        raise HarborSuiteError("at least one task must be selected")
    if len(set(task_names)) != len(task_names):
        raise HarborSuiteError("task selection contains duplicates")
    refs_by_id = {ref.path.name: ref for ref in suite.tasks}
    unknown = sorted(set(task_names) - refs_by_id.keys())
    if unknown:
        raise HarborSuiteError(f"unknown task(s) for {suite.id}: {', '.join(unknown)}")
    return tuple(refs_by_id[name] for name in task_names)


def iter_task_dirs(suite: Suite) -> tuple[Path, ...]:
    return tuple(ref.path for ref in suite.tasks)


def task_short_name(task_dir: Path) -> str:
    return task_dir.name


def task_full_name(suite: Suite, task_dir: Path) -> str:
    for ref in suite.tasks:
        if ref.path == task_dir:
            return ref.name
    return f"{suite.dataset_name}-{task_dir.name}"


def _harbor_digest(task_dir: Path) -> str:
    try:
        return _native_task_digest(task_dir)
    except HarborDigestError as exc:
        raise HarborSuiteError(str(exc)) from exc


def task_digest(task_dir: Path) -> str:
    return _harbor_digest(task_dir)


def suite_digests(suite: Suite) -> tuple[TaskDigest, ...]:
    return tuple(
        TaskDigest(task_short_name(ref.path), ref.name, task_digest(ref.path))
        for ref in suite.tasks
    )


def _workflow_fixture_digest_failures(
    task_dir: Path, rel: str, metadata: dict[str, Any]
) -> list[str]:
    if metadata.get("evaluation_kind") != "workflow":
        return []
    fixture = task_dir / "environment" / "input.json"
    if not fixture.is_file():
        return []
    expected = DIGEST_PREFIX + hashlib.sha256(fixture.read_bytes()).hexdigest()
    if metadata["fixture_digest"] == expected:
        return []
    return [f"{rel}/task.toml: fixture_digest does not match environment/input.json"]


def _iter_files(root: Path) -> Iterator[Path]:
    yield from (p for p in sorted(root.rglob("*")) if p.is_file())


def _metadata_failures(
    suite: Suite, task_dir: Path, rel: str, metadata: Any
) -> list[str]:
    failures: list[str] = []
    if not isinstance(metadata, dict) or not set(metadata) >= REQUIRED_METADATA:
        return [f"{rel}/task.toml: required [metadata] fields are missing"]
    ref = next((item for item in suite.tasks if item.path == task_dir), None)
    if ref and metadata["assurance_ceiling"] != ref.maximum_assurance:
        failures.append(f"{rel}/task.toml: assurance ceiling disagrees with suite")
    if ref and metadata["required_provider"] != ref.required_provider:
        failures.append(f"{rel}/task.toml: provider disagrees with suite")
    if ref:
        for field, expected in {
            "evaluation_kind": ref.evaluation_kind,
            "domain": ref.domain,
            "field": ref.field,
            "provenance_class": ref.provenance_class,
        }.items():
            if metadata.get(field) != expected:
                failures.append(
                    f"{rel}/task.toml: {field} disagrees with member record"
                )
    failures.extend(_workflow_fixture_digest_failures(task_dir, rel, metadata))
    return failures


def _network_mode_failures(cfg: dict[str, Any], rel: str) -> list[str]:
    failures: list[str] = []
    environment = cfg.get("environment", {})
    verifier = cfg.get("verifier", {})
    verifier_environment = (
        verifier.get("environment", {}) if isinstance(verifier, dict) else {}
    )
    for label, value in (
        ("environment.network_mode", environment.get("network_mode")),
        ("verifier.environment.network_mode", verifier_environment.get("network_mode")),
    ):
        if value not in NETWORK_MODES:
            failures.append(
                f"{rel}/task.toml: {label} must be one of {sorted(NETWORK_MODES)}"
            )
    return failures


def _task_manifest_failures(suite: Suite, task_dir: Path, rel: str) -> list[str]:
    cfg_path = task_dir / "task.toml"
    if not cfg_path.is_file():
        return []
    failures: list[str] = []
    cfg = _read_toml(cfg_path)
    if cfg.get("schema_version") != TASK_SCHEMA_VERSION:
        failures.append(
            f"{rel}/task.toml: schema_version must be {TASK_SCHEMA_VERSION}"
        )
    if cfg.get("task", {}).get("name") != task_full_name(suite, task_dir):
        failures.append(f"{rel}/task.toml: task.name disagrees with suite manifest")
    failures.extend(_metadata_failures(suite, task_dir, rel, cfg.get("metadata", {})))
    failures.extend(_network_mode_failures(cfg, rel))
    return failures


def _agent_environment_failures(suite: Suite, task_dir: Path, rel: str) -> list[str]:
    env = task_dir / "environment"
    if not env.is_dir():
        return [f"{rel}/environment: directory missing"]
    failures = [
        f"{rel}/environment/{name}: required file missing"
        for name in REQUIRED_ENVIRONMENT
        if not (env / name).is_file()
    ]
    docker = env / "Dockerfile"
    if not docker.is_file():
        return failures
    docker_text = docker.read_text(encoding="utf-8")
    ref = next((item for item in suite.tasks if item.path == task_dir), None)
    profiles = load_environment_profiles()
    profile = profiles.get(ref.environment_profile) if ref else None
    if ref and profile is None:
        failures.append(
            f"{rel}: unknown environment profile {ref.environment_profile!r}"
        )
    if profile is not None:
        first_from = next(
            (
                line.strip()[5:].strip()
                for line in docker_text.splitlines()
                if line.strip().upper().startswith("FROM ")
            ),
            None,
        )
        if first_from != profile.agent_image:
            failures.append(
                f"{rel}/environment/Dockerfile: FROM does not match "
                f"environment profile {profile.name}"
            )
        if not profile.allow_apt and re.search(r"(?i)\bapt-get\b", docker_text):
            failures.append(
                f"{rel}/environment/Dockerfile: standard profile prohibits apt-get"
            )
    if re.search(r"(?i)COPY\s+(?:solution|tests)(?:[/\s])", docker_text):
        failures.append(f"{rel}/environment/Dockerfile: copies hidden material")
    return failures


def validate_task_topology(suite: Suite, task_dir: Path) -> list[str]:
    failures: list[str] = []
    rel = task_dir.relative_to(ROOT).as_posix()
    for name in ("README.md", "instruction.md", "task.toml"):
        if not (task_dir / name).is_file():
            failures.append(f"{rel}/{name}: required file missing")
    for path in task_dir.rglob("*"):
        if path.is_symlink():
            failures.append(f"{rel}: symlink is forbidden")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
            failures.append(
                f"{path.relative_to(ROOT)}: raw interpreter cache is forbidden"
            )
    try:
        failures.extend(_task_manifest_failures(suite, task_dir, rel))
    except HarborSuiteError as exc:
        failures.append(str(exc))
    failures.extend(_agent_environment_failures(suite, task_dir, rel))
    tests = task_dir / "tests"
    if not tests.is_dir():
        failures.append(f"{rel}/tests: directory missing")
    else:
        for name in REQUIRED_TESTS:
            if not (tests / name).is_file():
                failures.append(f"{rel}/tests/{name}: required file missing")
        docker = tests / "Dockerfile"
        if docker.is_file() and re.search(
            r"(?i)COPY\s+solution(?:[/\s])", docker.read_text()
        ):
            failures.append(f"{rel}/tests/Dockerfile: copies Oracle solution")
        if docker.is_file():
            docker_text = docker.read_text()
            ref = next((item for item in suite.tasks if item.path == task_dir), None)
            profile = (
                load_environment_profiles().get(ref.environment_profile)
                if ref
                else None
            )
            first_from = next(
                (
                    line.strip()[5:].strip()
                    for line in docker_text.splitlines()
                    if line.strip().upper().startswith("FROM ")
                ),
                None,
            )
            if profile is not None and first_from != profile.verifier_image:
                failures.append(
                    f"{rel}/tests/Dockerfile: FROM does not match verifier image "
                    f"for environment profile {profile.name}"
                )
            if "verifier_support.py" not in docker_text:
                failures.append(
                    f"{rel}/tests/Dockerfile: does not copy verifier_support.py"
                )
            checksum = re.search(r'jacobian\.checksum="([0-9a-f]{64})"', docker_text)
            expected_checksum = hashlib.sha256(
                (tests / "verifier.py").read_bytes()
            ).hexdigest()
            if checksum is None:
                failures.append(
                    f"{rel}/tests/Dockerfile: missing verifier checksum label"
                )
            elif checksum.group(1) != expected_checksum:
                failures.append(
                    f"{rel}/tests/Dockerfile: verifier checksum label is stale"
                )
    if not (task_dir / "solution").is_dir():
        failures.append(f"{rel}/solution: directory missing")
    for forbidden in (
        task_dir / "input.json",
        task_dir / "metadata.json",
        task_dir / "environment" / "metadata.json",
    ):
        if forbidden.exists():
            failures.append(
                f"{forbidden.relative_to(ROOT)}: deprecated duplicate fixture"
            )
    return failures


def validate_task_visibility(task_dir: Path) -> list[str]:
    failures: list[str] = []
    visible = [task_dir / "instruction.md", task_dir / "environment"]
    for root in visible:
        paths = (
            [root]
            if root.is_file()
            else list(_iter_files(root))
            if root.is_dir()
            else []
        )
        for path in paths:
            if path.name in FORBIDDEN_VISIBLE_NAMES:
                failures.append(
                    f"{path.relative_to(ROOT)}: Oracle/verifier material is agent-visible"
                )
            text = path.read_text(encoding="utf-8", errors="replace")
            if _HOST_PATH.search(text):
                failures.append(
                    f"{path.relative_to(ROOT)}: host path in agent-visible file"
                )
            if _SECRET.search(text):
                failures.append(
                    f"{path.relative_to(ROOT)}: possible secret in agent-visible file"
                )
            for match in _FLOATING.finditer(text):
                package = match.group(1)
                if package.startswith("-"):
                    continue
                if (
                    "==" not in package
                    and "@" not in package
                    and not package.startswith(("/", "."))
                ):
                    failures.append(
                        f"{path.relative_to(ROOT)}: unpinned dependency {package}"
                    )
    return failures


def validate_task(suite: Suite, task_dir: Path) -> list[str]:
    return validate_task_topology(suite, task_dir) + validate_task_visibility(task_dir)


def _canonical_support(suite: Suite) -> Path | None:
    candidate = ROOT / "benchmarks" / "tooling" / "verifier_support.py"
    return candidate if candidate.is_file() else None


def check_verifier_support(
    suite: Suite,
    refs: tuple[TaskRef, ...] | None = None,
) -> list[str]:
    failures: list[str] = []
    canonical = _canonical_support(suite)
    if canonical is None:
        return [
            "benchmarks/tooling/verifier_support.py: canonical verifier support is missing"
        ]
    expected = canonical.read_bytes()
    for ref in suite.tasks if refs is None else refs:
        target = ref.path / "tests" / "verifier_support.py"
        if not target.is_file() or target.read_bytes() != expected:
            failures.append(
                f"{target.relative_to(ROOT)}: verifier support differs from canonical source"
            )
    return failures


def sync_verifier_support(suite: Suite) -> int:
    canonical = _canonical_support(suite)
    if canonical is None:
        return 0
    for ref in suite.tasks:
        target = ref.path / "tests" / "verifier_support.py"
        target.write_bytes(canonical.read_bytes())
        docker = ref.path / "tests" / "Dockerfile"
        if docker.is_file():
            text = docker.read_text()
            digest = hashlib.sha256(
                (ref.path / "tests" / "verifier.py").read_bytes()
            ).hexdigest()
            text, count = re.subn(
                r'jacobian\.checksum="[0-9a-f]{64}"',
                f'jacobian.checksum="{digest}"',
                text,
            )
            if not count:
                text, count = re.subn(
                    r"^(FROM [^\n]+\n)",
                    f'\\1LABEL jacobian.checksum="{digest}"\n',
                    text,
                    count=1,
                    flags=re.MULTILINE,
                )
            if count:
                docker.write_text(text)
    return 0


def check_suite_topology(suite: Suite) -> list[str]:
    failures: list[str] = []
    for ref in suite.tasks:
        failures.extend(validate_task(suite, ref.path))
        try:
            task_digest(ref.path)
        except (HarborSuiteError, OSError, ValueError) as exc:
            failures.append(
                f"{ref.path.relative_to(ROOT)}: Harbor task model rejected bundle: {exc}"
            )
    failures.extend(check_verifier_support(suite))
    return failures


def check_selected_tasks(
    suite: Suite,
    task_names: tuple[str, ...],
) -> list[str]:
    """Validate only the explicitly selected leaf task bundles."""

    refs = select_task_refs(suite, task_names)
    failures: list[str] = []
    for ref in refs:
        failures.extend(validate_task(suite, ref.path))
        try:
            task_digest(ref.path)
        except (HarborSuiteError, OSError, ValueError) as exc:
            failures.append(
                f"{ref.path.relative_to(ROOT)}: Harbor task model rejected bundle: {exc}"
            )
    failures.extend(check_verifier_support(suite, refs))
    return failures


def check_suite(suite: Suite) -> list[str]:
    return check_suite_topology(suite)


def report_failures(failures: list[str], *, header: str) -> bool:
    if not failures:
        return False
    print(header + ":", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    return True


def report_ok(message: str) -> None:
    print(message)


__all__ = [
    "HarborSuiteError",
    "Suite",
    "TaskRef",
    "check_selected_tasks",
    "check_suite",
    "check_suite_topology",
    "check_verifier_support",
    "get_suite",
    "iter_task_dirs",
    "load_environment_profiles",
    "load_registry",
    "report_failures",
    "report_ok",
    "select_task_refs",
    "suite_digests",
    "sync_verifier_support",
    "task_digest",
    "task_full_name",
    "task_short_name",
    "validate_global_task_ids",
    "validate_task",
    "validate_task_topology",
    "validate_task_visibility",
]
