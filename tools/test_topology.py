"""Validate and execute the repository's semantic pytest topology.

This module is deliberately a small control plane.  It reads
``tests/topology.toml``, validates ownership, and delegates collection and
execution to pytest.  Selection, filtering, retries, and fixture resolution
remain pytest concerns.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandStatus,
    operator_environment,
    run_operator_command,
)

from tools.pytest_lifecycle import run_pytest  # noqa: E402

DEFAULT_MANIFEST = _ROOT / "tests" / "topology.toml"
_DISTRIBUTIONS = {"none", "load", "loadscope", "loadfile", "loadgroup", "worksteal"}
_CI_TARGETS = {"pull_request", "merge_queue", "main", "scheduled"}

# Explicit per-requirement allowlist of host environment variables that a lane
# may forward to its pytest child.  No host variable is forwarded unless the
# lane declares the matching ``required_environment`` tag here.  ``PATH`` is
# forwarded for every lane because pytest children resolve executables by name
# (git, prlimit, optional solver binaries); the pinned Lean lane additionally
# needs ``HOME`` and ``ELAN_HOME`` so the elan-managed toolchain can be located
# by the lean runtime.  This is an explicit authorization, not a default: an
# arbitrary host variable never leaks through regardless of what a lane
# declares.
_LANE_ENVIRONMENT_ALLOWLIST: Mapping[str, tuple[str, ...]] = {
    "lean-4.31.0": ("HOME", "ELAN_HOME", "JACOBIAN_LEAN_RUNTIME"),
    "mathlib": ("HOME", "ELAN_HOME", "JACOBIAN_LEAN_RUNTIME"),
    "provider-readiness": (
        "HOME",
        "ELAN_HOME",
        "JACOBIAN_CHECKER_EXECUTABLE",
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "JACOBIAN_CHECKER_LAKE_DIGEST",
        "JACOBIAN_LEAN_RUNTIME",
    ),
}


class TopologyError(ValueError):
    """Raised when a topology manifest is malformed or incomplete."""


@dataclass(frozen=True)
class Lane:
    """One independently scheduled pytest resource lane."""

    name: str
    tier: str
    paths: tuple[str, ...]
    workers: int
    distribution: str
    timeout_seconds: int
    required_environment: tuple[str, ...]
    required_provider: str | None
    timing_sharding: bool
    ci: Mapping[str, bool]
    execution_profile: str = ""
    process_supervision: bool = False
    setup_affinity: str = ""


@dataclass(frozen=True)
class Topology:
    """A validated topology and the repository root it belongs to."""

    manifest: Path
    root: Path
    lanes: tuple[Lane, ...]

    def lane(self, name: str) -> Lane:
        for lane in self.lanes:
            if lane.name == name:
                return lane
        raise TopologyError(f"unknown topology lane: {name}")


def _as_string_list(
    value: Any, field: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TopologyError(f"{field} must be an array of strings")
    if not allow_empty and not value:
        raise TopologyError(f"{field} must not be empty")
    return tuple(value)


def _tracked_files(root: Path) -> set[str]:
    """Return tracked paths, with a filesystem fallback for extracted trees."""
    result = run_operator_command(
        "git",
        ("ls-files", "--cached", "--others", "--exclude-standard"),
        cwd=root,
        timeout_seconds=30.0,
        stdout_limit_bytes=16 * 1024 * 1024,
        stderr_limit_bytes=4096,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        return {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
    return {
        line
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line
    }


def _test_files(root: Path, tracked: set[str]) -> set[str]:
    return {
        path
        for path in tracked
        if path.startswith("tests/")
        and Path(path).name.startswith("test_")
        and Path(path).suffix == ".py"
        and (root / path).is_file()
    }


def _matches(root: Path, pattern: str) -> set[str]:
    """Expand one ownership path into relative files.

    Paths may name a file, directory, or a normal pathlib glob.  Directory
    ownership is recursive, which keeps the manifest readable as tests grow.
    """
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise TopologyError(f"lane path must be repository-relative: {pattern}")
    candidates = (
        list(root.glob(pattern))
        if any(char in pattern for char in "*?[")
        else [root / pattern]
    )
    if not candidates:
        raise TopologyError(f"lane path does not resolve: {pattern}")
    files: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_dir():
            files.update(
                path.relative_to(root).as_posix()
                for path in candidate.rglob("test_*.py")
                if path.is_file()
            )
        elif (
            candidate.is_file()
            and candidate.name.startswith("test_")
            and candidate.suffix == ".py"
        ):
            files.add(candidate.relative_to(root).as_posix())
    if not files:
        raise TopologyError(f"lane path resolves without test files: {pattern}")
    return files


def _validate_lane_fields(raw: dict[str, Any], index: int) -> dict[str, Any]:
    """Extract and type-check required lane fields, raising on the first error."""
    try:
        name = raw["name"]
        tier = raw["tier"]
        paths = _as_string_list(raw["paths"], f"lane {index}.paths", allow_empty=False)
        workers = raw["workers"]
        distribution = raw["distribution"]
        timeout = raw["timeout_seconds"]
        environment = _as_string_list(
            raw["required_environment"], f"lane {index}.required_environment"
        )
        provider = raw["required_provider"]
        timing = raw["timing_sharding"]
        ci = raw["ci"]
    except KeyError as exc:
        raise TopologyError(f"lane {index} missing field: {exc.args[0]}") from exc
    return {
        "name": name,
        "tier": tier,
        "paths": paths,
        "workers": workers,
        "distribution": distribution,
        "timeout": timeout,
        "environment": environment,
        "provider": provider,
        "timing": timing,
        "ci": ci,
        "execution_profile": raw.get("execution_profile", ""),
        "process_supervision": raw.get("process_supervision", False),
        "setup_affinity": raw.get("setup_affinity", ""),
    }


def _ci_ok(ci: Any) -> bool:
    return (
        isinstance(ci, dict)
        and set(ci) == _CI_TARGETS
        and all(isinstance(value, bool) for value in ci.values())
    )


def _validate_workers_distribution(name: str, workers: Any, distribution: Any) -> None:
    if not isinstance(workers, int) or workers < 0:
        raise TopologyError(f"lane {name}.workers must be a non-negative integer")
    if not isinstance(distribution, str) or distribution not in _DISTRIBUTIONS:
        raise TopologyError(f"lane {name}.distribution is invalid")
    if workers == 0 and distribution != "none":
        raise TopologyError(f"lane {name}: workers=0 requires distribution='none'")
    if workers > 0 and distribution == "none":
        raise TopologyError(f"lane {name}: workers>0 requires an xdist distribution")


def _validate_execution_profile_fields(
    name: str,
    *,
    execution_profile: Any,
    process_supervision: Any,
    setup_affinity: Any,
) -> tuple[str, bool, str]:
    if not isinstance(execution_profile, str):
        raise TopologyError(f"lane {name}.execution_profile must be a string")
    if not isinstance(process_supervision, bool):
        raise TopologyError(f"lane {name}.process_supervision must be boolean")
    if not isinstance(setup_affinity, str):
        raise TopologyError(f"lane {name}.setup_affinity must be a string")
    return execution_profile, process_supervision, setup_affinity


def _validate_lane_constraints(
    fields: dict[str, Any], index: int, names: set[str]
) -> Lane:
    name = fields["name"]
    tier = fields["tier"]
    workers = fields["workers"]
    distribution = fields["distribution"]
    timeout = fields["timeout"]
    provider = fields["provider"]
    timing = fields["timing"]
    ci = fields["ci"]
    if not isinstance(name, str) or not name or name in names:
        raise TopologyError(f"lane {index} has duplicate or invalid name")
    if not isinstance(tier, str) or not tier:
        raise TopologyError(f"lane {name}.tier must be a string")
    _validate_workers_distribution(name, workers, distribution)
    if not isinstance(timeout, int) or timeout <= 0:
        raise TopologyError(f"lane {name}.timeout_seconds must be positive")
    if provider is not None and not isinstance(provider, str):
        raise TopologyError(f"lane {name}.required_provider must be a string")
    if provider == "":
        provider = None
    if not isinstance(timing, bool):
        raise TopologyError(f"lane {name}.timing_sharding must be boolean")
    if not _ci_ok(ci):
        raise TopologyError(f"lane {name}.ci must define {_CI_TARGETS} as booleans")
    execution_profile, process_supervision, setup_affinity = (
        _validate_execution_profile_fields(
            name,
            execution_profile=fields["execution_profile"],
            process_supervision=fields["process_supervision"],
            setup_affinity=fields["setup_affinity"],
        )
    )
    names.add(name)
    return Lane(
        name,
        tier,
        fields["paths"],
        workers,
        distribution,
        timeout,
        fields["environment"],
        provider,
        timing,
        dict(ci),
        execution_profile,
        process_supervision,
        setup_affinity,
    )


def _validate_ownership(lanes: list[Lane], root: Path, tracked: set[str]) -> None:
    test_files = _test_files(root, tracked)
    ownership: dict[str, list[str]] = {path: [] for path in test_files}
    for lane in lanes:
        for pattern in lane.paths:
            matches = _matches(root, pattern)
            untracked = matches - tracked
            if untracked:
                raise TopologyError(
                    f"lane {lane.name} owns untracked paths: {sorted(untracked)}"
                )
            for path in matches:
                if path in ownership:
                    ownership[path].append(lane.name)
    missing = sorted(path for path, owners in ownership.items() if not owners)
    overlapping = sorted(
        (path, owners) for path, owners in ownership.items() if len(owners) != 1
    )
    if overlapping:
        details = "; ".join(
            f"{path} ({', '.join(owners)})" for path, owners in overlapping
        )
        raise TopologyError(f"test files belong to multiple lanes: {details}")
    if missing:
        raise TopologyError(f"test files have no lane: {', '.join(missing)}")


def validate_topology(
    data: Mapping[str, Any], *, root: Path, manifest: Path
) -> Topology:
    """Validate raw TOML data and return typed topology metadata."""
    if data.get("version") != 1:
        raise TopologyError("topology version must be 1")
    raw_lanes = data.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise TopologyError("topology must define a non-empty [[lanes]] array")

    lanes: list[Lane] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_lanes):
        if not isinstance(raw, dict):
            raise TopologyError(f"lane {index} must be a table")
        fields = _validate_lane_fields(raw, index)
        lanes.append(_validate_lane_constraints(fields, index, names))

    tracked = _tracked_files(root)
    _validate_ownership(lanes, root, tracked)
    return Topology(manifest, root, tuple(lanes))


def load_topology(path: Path = DEFAULT_MANIFEST) -> Topology:
    """Load and validate a topology TOML file."""
    manifest = path.resolve()
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TopologyError(f"cannot read topology manifest {manifest}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise TopologyError(f"invalid topology TOML: {exc}") from exc
    return validate_topology(data, root=manifest.parent.parent, manifest=manifest)


def _has_explicit_xdist_args(extra_args: list[str] | None) -> bool:
    """Return whether ``extra_args`` supply their own xdist configuration.

    pytest takes the last ``-n``/``--numprocesses`` value, so lane defaults
    appended after an explicit ``-n 0`` would override the user's request for
    serial execution.  Detect any explicit xdist worker count so the lane
    defaults can be suppressed in that case.
    """
    if not extra_args:
        return False
    for arg in extra_args:
        if arg in ("-n", "--numprocesses"):
            return True
        if arg.startswith(("-n=", "--numprocesses=")):
            return True
    return False


def pytest_command(
    topology: Topology,
    lane_name: str,
    selectors: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build one pytest invocation without interpreting selectors.

    An explicit selector is the focused ``TESTS=`` form used by local edit
    loops.  It does not need the lane's worker pool, and starting that pool can
    cost more than the selected test itself.  An omitted selector still means
    "the whole lane" and retains the configured CI parallelism.  Lanes with
    ``workers=0`` are unaffected by this distinction.

    Explicit xdist arguments in ``extra_args`` (e.g. ``-n 0`` for
    debugger-friendly serial execution) suppress the lane's configured worker
    pool and distribution so the user's choice is honored rather than
    overridden by the lane defaults that follow it.
    """
    lane = topology.lane(lane_name)
    command = [sys.executable, "-m", "pytest"]
    command.extend(selectors if selectors else list(lane.paths))
    command.extend(extra_args or ())
    if lane.workers and not selectors and not _has_explicit_xdist_args(extra_args):
        command.extend(["-n", str(lane.workers), "--dist", lane.distribution])
    command.extend(["--timeout", str(lane.timeout_seconds)])
    return command


def lane_environment(lane: Lane) -> Mapping[str, str]:
    """Build the bounded pytest environment for one lane.

    Only the operator allowlist, ``PATH``, and the explicitly allowlisted
    Lean/provider variables for the lane's declared ``required_environment``
    are forwarded from the host.  No arbitrary host environment leaks
    through: a variable not named by ``PATH`` or
    :data:`_LANE_ENVIRONMENT_ALLOWLIST` is never forwarded, and the lane
    tag is declared rather than inherited.
    """
    include: set[str] = {"PATH"}
    for requirement in lane.required_environment:
        include.update(_LANE_ENVIRONMENT_ALLOWLIST.get(requirement, ()))
    return operator_environment(
        include=include,
        declared={"JACOBIAN_TEST_LANE": lane.name},
    )


def run_lane(
    topology: Topology,
    lane_name: str,
    selectors: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Execute pytest via the bounded tooling command runner.

    Captured stdout/stderr from the bounded child are forwarded to the
    parent streams so failures are not silent.  On non-EXITED terminal
    states (timeout, overflow, start failure) a diagnostic line is emitted
    to stderr before returning a non-zero exit code.
    """
    command = pytest_command(topology, lane_name, selectors, extra_args)
    arguments = command[3:]
    environment = lane_environment(topology.lane(lane_name))
    result = run_pytest(
        arguments,
        root=topology.root,
        name=f"topology-{lane_name}",
        environment=environment,
        timeout_seconds=3600.0,
    )
    return result.exit_code


def _print_dry_run(
    lane: Lane, command: list[str], *, selectors: list[str] | None
) -> None:
    """Print lane metadata and the resolved pytest command for inspection.

    Metadata lines are prefixed with ``#`` so the command remains the final,
    un-prefixed line and stays consumable by callers that only need the
    invocation string.  The effective worker count and distribution reflect
    the focused-selector shortcut: an explicit selector disables the lane's
    configured worker pool, so the reported values drop to zero/``none`` even
    though the lane itself may declare parallelism.
    """
    focused = bool(selectors)
    workers = 0 if focused else lane.workers
    distribution = "none" if focused else lane.distribution
    print(f"# lane: {lane.name}")
    print(f"# tier: {lane.tier}")
    print(f"# workers: {workers}")
    print(f"# distribution: {distribution}")
    print(f"# timeout_seconds: {lane.timeout_seconds}")
    print(f"# timing_sharding: {'true' if lane.timing_sharding else 'false'}")
    if lane.execution_profile:
        print(f"# execution_profile: {lane.execution_profile}")
    print(
        f"# process_supervision: "
        f"{'true' if lane.process_supervision else 'false'}"
    )
    if lane.setup_affinity:
        print(f"# setup_affinity: {lane.setup_affinity}")
    if selectors:
        print(f"# selectors: {len(selectors)}")
    print(shlex.join(command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", help="topology lane to execute")
    parser.add_argument("selectors", nargs="*", help="exact pytest paths or node IDs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--pytest-args",
        default="",
        help="shell-style pytest arguments supplied by the repository wrapper",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print lane metadata and the pytest command without executing",
    )
    args, extra_args = parser.parse_known_args(argv)
    try:
        configured_pytest_args = shlex.split(args.pytest_args)
    except ValueError as exc:
        parser.error(f"invalid --pytest-args value: {exc}")
    extra_args = [*configured_pytest_args, *extra_args]
    try:
        topology = load_topology(args.manifest)
        lane = topology.lane(args.lane)
        command = pytest_command(topology, args.lane, args.selectors, extra_args)
    except TopologyError as exc:
        parser.error(str(exc))
    if args.dry_run:
        _print_dry_run(lane, command, selectors=args.selectors)
        return 0
    return run_lane(topology, args.lane, args.selectors, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
