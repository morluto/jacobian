#!/usr/bin/env python3
"""Plan bounded CI evidence from normalized repository-relative paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

PLAN_VERSION = 3
SUPPORTED_EVENTS = frozenset(
    {"pull_request", "merge_group", "push", "schedule", "workflow_dispatch"}
)
_REVISION = re.compile(r"[0-9a-f]{7,64}\Z")
_SHARED_PATHS = frozenset(
    {
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "src/jacobian/_exact.py",
        "src/jacobian/_models.py",
        "src/jacobian/canonical.py",
    }
)
_SHARED_PREFIXES = (
    ".github/",
    "make/",
    "tools/",
    "src/jacobian/catalog/",
    "src/jacobian/mcp/",
)
_PUBLIC_MATH_FILES = frozenset(
    {
        "_admission.py",
        "_cnf.py",
        "_interval.py",
        "_models.py",
        "_operations.py",
        "_pseudomanifold.py",
        "_structural.py",
        "_sat.py",
        "_smt.py",
        "_tools.py",
        "operations.py",
        "values.py",
    }
)
_PYTHON_LANES = ("dispatch", "cli", "tooling", "integration")
_BOUNDARY_LANES = ("process", "mcp")
_SCALE_TEST_PREFIXES = ("tests/math/geometry/polytopes/lattice/",)
_FULL_MATH_SHARD_COUNT = 4


@dataclass(frozen=True)
class MathShard:
    group: int
    splits: int


@dataclass(frozen=True)
class TestPlan:
    version: int
    event: str
    base_revision: str
    head_revision: str
    topology_digest: str
    run_math: bool
    math_tests: tuple[str, ...]
    math_shards: tuple[MathShard, ...]
    run_catalog: bool
    run_catalog_examples: bool
    run_scale: bool
    python_lanes: tuple[str, ...]
    boundary_lanes: tuple[str, ...]
    run_singular: bool
    run_wheel: bool
    reasons: tuple[str, ...]

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PathDecision:
    math_tests: tuple[str, ...] = ()
    run_catalog: bool = False
    run_catalog_examples: bool = False
    run_scale: bool = False
    python_lanes: tuple[str, ...] = ()
    boundary_lanes: tuple[str, ...] = ()
    run_singular: bool = False
    run_wheel: bool = False
    full_math_reason: str | None = None


def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_path in paths:
        path = raw_path.strip()
        if not path:
            continue
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or "\\" in path or ".." in candidate.parts:
            raise ValueError(
                f"path must be normalized and repository-relative: {raw_path!r}"
            )
        normalized.append(candidate.as_posix())
    return tuple(sorted(set(normalized)))


def _topology_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_shared(path: str) -> bool:
    return path in _SHARED_PATHS or path.startswith(_SHARED_PREFIXES)


def _math_owner_tests(source_path: str, repository: Path) -> tuple[str, ...] | None:
    root = repository / "tests" / "math"
    relative = PurePosixPath(source_path).relative_to("src/jacobian/math")
    package_parts = relative.parts[:-1]
    for depth in range(len(package_parts), 0, -1):
        owner_root = root.joinpath(*package_parts[:depth])
        if owner_root.exists():
            return (owner_root.relative_to(repository).as_posix(),)

    if not package_parts:
        return None
    owner = package_parts[0]
    paths = tuple(
        path.relative_to(repository).as_posix()
        for path in sorted(root.glob(f"test_{owner}*.py"))
    )
    return paths or None


def _includes_scale_tests(paths: tuple[str, ...]) -> bool:
    return any(
        path == prefix.removesuffix("/") or path.startswith(prefix)
        for path in paths
        for prefix in _SCALE_TEST_PREFIXES
    )


def _math_test_change(path: str) -> tuple[str, ...] | None:
    relative = PurePosixPath(path).relative_to("tests/math")
    if relative.name.startswith("_") or relative.name == "conftest.py":
        return None
    return (path,)


def _is_public_math_path(path: str) -> bool:
    filename = PurePosixPath(path).name
    return filename in _PUBLIC_MATH_FILES or filename.endswith("_operations.py")


def _complete_decision(reason: str) -> PathDecision:
    return PathDecision(
        run_catalog=True,
        run_catalog_examples=True,
        run_scale=True,
        python_lanes=_PYTHON_LANES,
        boundary_lanes=_BOUNDARY_LANES,
        run_singular=True,
        run_wheel=True,
        full_math_reason=reason,
    )


def _math_shards(*, run_math: bool, full_suite: bool) -> tuple[MathShard, ...]:
    if not run_math:
        return ()
    count = _FULL_MATH_SHARD_COUNT if full_suite else 1
    return tuple(MathShard(group=group, splits=count) for group in range(1, count + 1))


def _classify_math_path(path: str, repository: Path) -> PathDecision:
    if path.startswith("src/jacobian/math/"):
        selected = _math_owner_tests(path, repository)
        if selected is None:
            return _complete_decision(f"math owner has no explicit test root: {path}")
        public_contract = _is_public_math_path(path)
        return PathDecision(
            math_tests=selected,
            run_catalog=public_contract,
            run_catalog_examples=public_contract,
            run_scale=_includes_scale_tests(selected),
        )
    return PathDecision()


def _classify_test_path(path: str) -> PathDecision | None:
    if path.startswith("tests/math/"):
        selected = _math_test_change(path)
        if selected is None:
            return PathDecision(
                full_math_reason=f"shared mathematical test support changed: {path}"
            )
        return PathDecision(
            math_tests=selected,
            run_scale=_includes_scale_tests((path,)),
        )
    if path.startswith("tests/catalog/"):
        return PathDecision(run_catalog=True)
    if path.startswith("tests/integration/catalog/"):
        return PathDecision(run_catalog_examples=True)
    if path.startswith("tests/integration/"):
        return PathDecision(python_lanes=("integration",))
    if path.startswith("tests/dispatch/"):
        return PathDecision(python_lanes=("dispatch",))
    if path.startswith("tests/cli/"):
        return PathDecision(python_lanes=("cli",))
    if path.startswith("tests/tooling/"):
        return PathDecision(python_lanes=("tooling",))
    if path.startswith("tests/process/"):
        return PathDecision(
            boundary_lanes=("process",),
            run_singular="/polynomials/" in path,
        )
    if path.startswith("tests/mcp/"):
        return PathDecision(boundary_lanes=("mcp",))
    if path.startswith(("tests/support/", "tests/fixtures/")):
        return _complete_decision(f"shared test support changed: {path}")
    return None


def _classify_runtime_path(path: str) -> PathDecision | None:
    if path == "src/jacobian/dispatch.py":
        return PathDecision(
            run_catalog=True,
            run_catalog_examples=True,
            python_lanes=("dispatch",),
            boundary_lanes=("mcp",),
        )
    if path == "src/jacobian/cli.py":
        return PathDecision(python_lanes=("cli",), run_wheel=True)
    if path == "src/jacobian/process.py":
        return PathDecision(boundary_lanes=("process",), run_singular=True)
    if path.startswith("src/jacobian/mcp/"):
        return PathDecision(boundary_lanes=("mcp",), run_wheel=True)
    if path.startswith("src/jacobian/"):
        return _complete_decision(f"unmapped Jacobian runtime path changed: {path}")
    return None


def _classify_path(path: str, repository: Path) -> PathDecision:
    if _is_shared(path):
        return _complete_decision(f"shared CI or runtime path changed: {path}")
    math_decision = _classify_math_path(path, repository)
    if math_decision != PathDecision():
        return math_decision
    test_decision = _classify_test_path(path)
    if test_decision is not None:
        return test_decision
    runtime_decision = _classify_runtime_path(path)
    if runtime_decision is not None:
        return runtime_decision
    return PathDecision()


def _pull_request_plan(
    *,
    base_revision: str,
    head_revision: str,
    paths: tuple[str, ...],
    repository: Path,
) -> TestPlan:
    math_tests: set[str] = set()
    run_catalog = False
    run_catalog_examples = False
    python_lanes: set[str] = set()
    boundary_lanes: set[str] = set()
    run_singular = False
    run_wheel = False
    full_math_reason: str | None = None
    reasons: list[str] = []

    for path in paths:
        decision = _classify_path(path, repository)
        math_tests.update(decision.math_tests)
        run_catalog = run_catalog or decision.run_catalog
        run_catalog_examples = run_catalog_examples or decision.run_catalog_examples
        python_lanes.update(decision.python_lanes)
        boundary_lanes.update(decision.boundary_lanes)
        run_singular = run_singular or decision.run_singular
        run_wheel = run_wheel or decision.run_wheel
        if decision.full_math_reason:
            full_math_reason = decision.full_math_reason
            break

    if full_math_reason:
        reasons.append(full_math_reason)
        math_tests.clear()
    else:
        if math_tests:
            reasons.append(
                "selected mathematical owners: " + ", ".join(sorted(math_tests))
            )
        if run_catalog:
            reasons.append(
                "public operation, model, admission, or canonical contract changed"
            )
        if run_catalog_examples:
            reasons.append(
                "catalog invocation examples cover the changed public contract"
            )
        if python_lanes:
            reasons.append("selected owner lanes: " + ", ".join(sorted(python_lanes)))
        if boundary_lanes:
            reasons.append(
                "selected boundary lanes: " + ", ".join(sorted(boundary_lanes))
            )
        if run_singular:
            reasons.append("pinned Singular boundary changed")
        if run_wheel:
            reasons.append("installed-wheel boundary changed")
        if not reasons:
            reasons.append("no mathematical or public-contract path changed")
    return TestPlan(
        version=PLAN_VERSION,
        event="pull_request",
        base_revision=base_revision,
        head_revision=head_revision,
        topology_digest=_topology_digest(),
        run_math=bool(math_tests) or full_math_reason is not None,
        math_tests=tuple(sorted(math_tests)),
        math_shards=_math_shards(
            run_math=bool(math_tests) or full_math_reason is not None,
            full_suite=full_math_reason is not None,
        ),
        run_catalog=run_catalog,
        run_catalog_examples=run_catalog_examples,
        run_scale=False,
        python_lanes=tuple(sorted(python_lanes)),
        boundary_lanes=tuple(sorted(boundary_lanes)),
        run_singular=run_singular,
        run_wheel=run_wheel,
        reasons=tuple(reasons),
    )


def _select_scale_evidence(paths: tuple[str, ...], repository: Path) -> bool:
    return any(_classify_path(path, repository).run_scale for path in paths)


def build_plan(
    *,
    event: str,
    base_revision: str,
    head_revision: str,
    changed_paths: Iterable[str],
    repository: Path,
) -> TestPlan:
    if event not in SUPPORTED_EVENTS:
        raise ValueError(f"unsupported CI event: {event!r}")
    if not _REVISION.fullmatch(base_revision) or not _REVISION.fullmatch(head_revision):
        raise ValueError("base and head revisions must be Git object IDs")

    paths = _normalize_paths(changed_paths)
    if event != "pull_request":
        run_scale = event in {"schedule", "workflow_dispatch"} or (
            event == "merge_group" and _select_scale_evidence(paths, repository)
        )
        return TestPlan(
            version=PLAN_VERSION,
            event=event,
            base_revision=base_revision,
            head_revision=head_revision,
            topology_digest=_topology_digest(),
            run_math=True,
            math_tests=(),
            math_shards=_math_shards(run_math=True, full_suite=True),
            run_catalog=True,
            run_catalog_examples=True,
            run_scale=run_scale,
            python_lanes=_PYTHON_LANES,
            boundary_lanes=_BOUNDARY_LANES,
            run_singular=True,
            run_wheel=True,
            reasons=(
                f"{event} owns the complete ordinary suite"
                + (" and optional scale evidence" if run_scale else ""),
            ),
        )

    return _pull_request_plan(
        base_revision=base_revision,
        head_revision=head_revision,
        paths=paths,
        repository=repository,
    )


def _write_github_output(plan: TestPlan, output: Path) -> None:
    values = {
        "plan": plan.as_json(),
        "run_math": str(plan.run_math).lower(),
        "math_tests": " ".join(plan.math_tests),
        "math_shards": json.dumps([asdict(shard) for shard in plan.math_shards]),
        "math_shard_count": str(len(plan.math_shards)),
        "run_catalog": str(plan.run_catalog).lower(),
        "run_catalog_examples": str(plan.run_catalog_examples).lower(),
        "run_scale": str(plan.run_scale).lower(),
        "run_python": str(bool(plan.python_lanes)).lower(),
        "python_lanes": json.dumps(plan.python_lanes),
        "run_boundaries": str(bool(plan.boundary_lanes)).lower(),
        "boundary_lanes": json.dumps(plan.boundary_lanes),
        "run_singular": str(plan.run_singular).lower(),
        "run_wheel": str(plan.run_wheel).lower(),
    }
    with output.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}<<JACOBIAN_CI_PLAN\n{value}\nJACOBIAN_CI_PLAN\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--changed-paths", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    plan = build_plan(
        event=arguments.event,
        base_revision=arguments.base,
        head_revision=arguments.head,
        changed_paths=arguments.changed_paths.read_text(encoding="utf-8").splitlines(),
        repository=arguments.repository.resolve(),
    )
    if arguments.github_output:
        _write_github_output(plan, arguments.github_output)
    print(plan.as_json())


if __name__ == "__main__":
    main()
