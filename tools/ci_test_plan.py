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

PLAN_VERSION = 1
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
        "_models.py",
        "_operations.py",
        "_tools.py",
        "operations.py",
        "values.py",
    }
)


@dataclass(frozen=True)
class TestPlan:
    version: int
    event: str
    base_revision: str
    head_revision: str
    topology_digest: str
    run_math: bool
    math_tests: tuple[str, ...]
    run_catalog: bool
    run_catalog_examples: bool
    reasons: tuple[str, ...]

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PathDecision:
    math_tests: tuple[str, ...] = ()
    run_catalog: bool = False
    run_catalog_examples: bool = False
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


def _math_owner_tests(owner: str, repository: Path) -> tuple[str, ...] | None:
    root = repository / "tests" / "math"
    paths: list[str] = []
    owner_root = root / owner
    if owner_root.exists():
        paths.append(owner_root.relative_to(repository).as_posix())
    paths.extend(
        path.relative_to(repository).as_posix()
        for path in sorted(root.glob(f"test_{owner}*.py"))
    )
    return tuple(dict.fromkeys(paths)) or None


def _math_test_change(path: str) -> tuple[str, ...] | None:
    relative = PurePosixPath(path).relative_to("tests/math")
    if relative.name.startswith("_") or relative.name == "conftest.py":
        return None
    return (path,)


def _is_public_math_path(path: str) -> bool:
    filename = PurePosixPath(path).name
    return filename in _PUBLIC_MATH_FILES or filename.endswith("_operations.py")


def _classify_path(path: str, repository: Path) -> PathDecision:
    if _is_shared(path):
        return PathDecision(
            run_catalog=True,
            run_catalog_examples=True,
            full_math_reason=f"shared CI or runtime path changed: {path}",
        )
    if path.startswith("src/jacobian/math/"):
        parts = PurePosixPath(path).parts
        owner = parts[3] if len(parts) > 3 else ""
        selected = _math_owner_tests(owner, repository) if owner else None
        if selected is None:
            return PathDecision(
                run_catalog=True,
                run_catalog_examples=True,
                full_math_reason=f"math owner has no explicit test root: {path}",
            )
        public_contract = _is_public_math_path(path)
        return PathDecision(
            math_tests=selected,
            run_catalog=public_contract,
            run_catalog_examples=public_contract,
        )
    if path.startswith("tests/math/"):
        selected = _math_test_change(path)
        if selected is None:
            return PathDecision(
                full_math_reason=f"shared mathematical test support changed: {path}"
            )
        return PathDecision(math_tests=selected)
    if path.startswith("tests/catalog/"):
        return PathDecision(run_catalog=True)
    if path.startswith("tests/integration/catalog/"):
        return PathDecision(run_catalog_examples=True)
    if path.startswith("src/jacobian/"):
        return PathDecision(
            run_catalog=True,
            run_catalog_examples=True,
            full_math_reason=f"unmapped Jacobian runtime path changed: {path}",
        )
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
    full_math_reason: str | None = None
    reasons: list[str] = []

    for path in paths:
        decision = _classify_path(path, repository)
        math_tests.update(decision.math_tests)
        run_catalog = run_catalog or decision.run_catalog
        run_catalog_examples = run_catalog_examples or decision.run_catalog_examples
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
        run_catalog=run_catalog,
        run_catalog_examples=run_catalog_examples,
        reasons=tuple(reasons),
    )


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
        return TestPlan(
            version=PLAN_VERSION,
            event=event,
            base_revision=base_revision,
            head_revision=head_revision,
            topology_digest=_topology_digest(),
            run_math=True,
            math_tests=(),
            run_catalog=True,
            run_catalog_examples=True,
            reasons=(
                f"{event} owns the complete mathematical and public-contract suite",
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
        "run_catalog": str(plan.run_catalog).lower(),
        "run_catalog_examples": str(plan.run_catalog_examples).lower(),
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
