"""Compile the authoritative test plan manifest into topology and CI impact views."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.test_plan.execution_profiles import (
    profile_for_lane,
    validate_lane_against_profile,
)

DEFAULT_MANIFEST = Path("tests/plan_manifest.toml")
DEFAULT_TOPOLOGY = Path("tests/topology.toml")
DEFAULT_IMPACT = Path(".github/ci-impact.json")
DEFAULT_FALLBACK_NAME = "unclassified-fail-closed"


@dataclass(frozen=True, slots=True)
class LaneSpec:
    name: str
    kind: str
    tier: str | None
    paths: tuple[str, ...]
    command: str
    workers: int
    distribution: str
    timeout_seconds: int
    required_environment: tuple[str, ...]
    required_provider: str
    timing_sharding: bool
    runs_on: Mapping[str, bool]
    matrix: bool
    providers: tuple[str, ...]
    local_subsumes: tuple[str, ...]
    local_only: bool
    topology_lane: bool


@dataclass(frozen=True, slots=True)
class ImpactRule:
    name: str
    patterns: tuple[str, ...]
    suites: tuple[str, ...]
    suppresses: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TestPlanManifest:
    version: int
    lanes: tuple[LaneSpec, ...]
    impact_rules: tuple[ImpactRule, ...]
    fallback_name: str
    fallback_suites: tuple[str, ...]
    source: Path

    @property
    def pytest_lanes(self) -> tuple[LaneSpec, ...]:
        return tuple(lane for lane in self.lanes if lane.kind == "pytest")

    @property
    def suite_order(self) -> tuple[str, ...]:
        return tuple(
            lane.name
            for lane in self.lanes
            if lane.kind in {"pytest", "gate"} and not lane.local_only
        )


def _bool_map(
    raw: Mapping[str, Any] | None, *, default: bool = True
) -> dict[str, bool]:
    keys = ("pull_request", "merge_queue", "main", "scheduled")
    raw = raw or {}
    return {key: bool(raw.get(key, default)) for key in keys}


def load_manifest(path: Path) -> TestPlanManifest:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 1))
    lanes: list[LaneSpec] = []
    for raw in payload.get("lanes", ()):
        lanes.append(
            LaneSpec(
                name=str(raw["name"]),
                kind=str(raw.get("kind", "pytest")),
                tier=str(raw["tier"]) if "tier" in raw else None,
                paths=tuple(str(item) for item in raw.get("paths", ())),
                command=str(raw["command"]),
                workers=int(raw.get("workers", 0)),
                distribution=str(raw.get("distribution", "none")),
                timeout_seconds=int(raw.get("timeout_seconds", 30)),
                required_environment=tuple(
                    str(item) for item in raw.get("required_environment", ())
                ),
                required_provider=str(raw.get("required_provider", "")),
                timing_sharding=bool(raw.get("timing_sharding", False)),
                runs_on=_bool_map(raw.get("ci") or raw.get("runs_on")),
                matrix=bool(raw.get("matrix", False)),
                providers=tuple(str(item) for item in raw.get("providers", ())),
                local_subsumes=tuple(
                    str(item) for item in raw.get("local_subsumes", ())
                ),
                local_only=bool(raw.get("local_only", False)),
                topology_lane=bool(
                    raw.get("topology_lane", raw.get("kind", "pytest") == "pytest")
                ),
            )
        )
    rules = tuple(
        ImpactRule(
            name=str(raw["name"]),
            patterns=tuple(str(item) for item in raw.get("patterns", ())),
            suites=tuple(str(item) for item in raw.get("suites", ())),
            suppresses=tuple(str(item) for item in raw.get("suppresses", ())),
        )
        for raw in payload.get("impact_rules", ())
    )
    fallback = payload.get("fallback", {})
    manifest = TestPlanManifest(
        version=version,
        lanes=tuple(lanes),
        impact_rules=rules,
        fallback_name=str(fallback.get("name", DEFAULT_FALLBACK_NAME)),
        fallback_suites=tuple(str(item) for item in fallback.get("suites", ())),
        source=path,
    )
    errors: list[str] = []
    for lane in manifest.pytest_lanes:
        errors.extend(
            validate_lane_against_profile(
                name=lane.name,
                required_environment=lane.required_environment,
                workers=lane.workers,
                distribution=lane.distribution,
                timeout_seconds=lane.timeout_seconds,
            )
        )
    if errors:
        raise ValueError("execution profile conflicts:\n" + "\n".join(errors))
    return manifest


def render_topology(manifest: TestPlanManifest) -> str:
    lines = [
        "version = 1",
        "",
        "# Generated by tools.test_plan.compile from tests/plan_manifest.toml.",
        "# Edit the manifest and run `make compile-test-plan`.",
        "",
    ]
    for lane in manifest.pytest_lanes:
        profile = profile_for_lane(
            name=lane.name,
            required_environment=lane.required_environment,
            workers=lane.workers,
            distribution=lane.distribution,
            timeout_seconds=lane.timeout_seconds,
        )
        lines.append("[[lanes]]")
        lines.append(f'name = "{lane.name}"')
        lines.append(f'tier = "{lane.tier or lane.name}"')
        paths = ", ".join(f'"{path}"' for path in lane.paths)
        lines.append(f"paths = [{paths}]")
        lines.append(f"workers = {lane.workers}")
        lines.append(f'distribution = "{lane.distribution}"')
        lines.append(f"timeout_seconds = {lane.timeout_seconds}")
        envs = ", ".join(f'"{item}"' for item in lane.required_environment)
        lines.append(f"required_environment = [{envs}]")
        lines.append(f'required_provider = "{lane.required_provider}"')
        lines.append(f"timing_sharding = {'true' if lane.timing_sharding else 'false'}")
        lines.append(f'execution_profile = "{profile.name}"')
        lines.append(
            f"process_supervision = "
            f"{'true' if profile.process_supervision else 'false'}"
        )
        if profile.setup_affinity is not None:
            lines.append(f'setup_affinity = "{profile.setup_affinity}"')
        else:
            lines.append('setup_affinity = ""')
        ci = lane.runs_on
        lines.append(
            "ci = { "
            f"pull_request = {'true' if ci['pull_request'] else 'false'}, "
            f"merge_queue = {'true' if ci['merge_queue'] else 'false'}, "
            f"main = {'true' if ci['main'] else 'false'}, "
            f"scheduled = {'true' if ci['scheduled'] else 'false'}"
            " }"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _catalog_entry(lane: LaneSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "command": lane.command,
        "topology_lane": lane.name if lane.topology_lane else None,
    }
    if lane.matrix:
        entry["matrix"] = True
    if lane.providers:
        entry["providers"] = list(lane.providers)
    if lane.local_subsumes:
        entry["local_subsumes"] = list(lane.local_subsumes)
    if lane.local_only:
        entry["local_only"] = True
    return entry


def _lane_test_patterns(lane: LaneSpec) -> tuple[str, ...]:
    return tuple(f"{path.rstrip('/')}/**" for path in lane.paths)


def _derived_test_rule(
    lane: LaneSpec, *, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    suites = [lane.name, "static"] if "static" in catalog else [lane.name]
    return {
        "name": f"{lane.name}-tests",
        "patterns": list(_lane_test_patterns(lane)),
        "suites": suites,
    }


def _explicit_covers_lane_dirs(
    rule: ImpactRule, lane: LaneSpec
) -> bool:
    """Return True when an explicit rule already owns this lane's test dirs."""

    derived = frozenset(_lane_test_patterns(lane))
    if not derived:
        return False
    explicit = frozenset(rule.patterns)
    if explicit == derived:
        return True
    # Same test directories under alternate naming (e.g. optional-provider-tests).
    lane_dirs = frozenset(path.rstrip("/") for path in lane.paths)
    explicit_dirs = frozenset(
        pattern[:-3] if pattern.endswith("/**") else pattern.rstrip("/")
        for pattern in rule.patterns
    )
    return bool(lane_dirs) and lane_dirs == explicit_dirs


def _generated_from(path: Path) -> str:
    """Stable relative label for compiled meta, independent of caller cwd."""

    as_posix = path.as_posix().replace("\\", "/")
    marker = "tests/plan_manifest.toml"
    if as_posix == marker or as_posix.endswith(f"/{marker}"):
        return marker
    return as_posix


def render_impact(manifest: TestPlanManifest) -> dict[str, Any]:
    catalog = {lane.name: _catalog_entry(lane) for lane in manifest.lanes}
    explicit_rules = [
        {
            "name": rule.name,
            "patterns": list(rule.patterns),
            "suites": list(rule.suites),
            **({"suppresses": list(rule.suppresses)} if rule.suppresses else {}),
        }
        for rule in manifest.impact_rules
    ]
    explicit_names = {rule.name for rule in manifest.impact_rules}
    derived_rules: list[dict[str, Any]] = []
    for lane in manifest.pytest_lanes:
        derived_name = f"{lane.name}-tests"
        if derived_name in explicit_names:
            continue
        if any(
            _explicit_covers_lane_dirs(rule, lane) for rule in manifest.impact_rules
        ):
            continue
        derived_rules.append(_derived_test_rule(lane, catalog=catalog))
    suites = list(manifest.suite_order)
    return {
        "version": 2,
        "suites": suites,
        "catalog": catalog,
        "rules": explicit_rules + derived_rules,
        "fallback": {
            "name": manifest.fallback_name,
            "suites": list(manifest.fallback_suites or suites),
        },
        "meta": {
            "generated_from": _generated_from(manifest.source),
            "compiler": "tools.test_plan.compile",
        },
    }


def suppression_map(manifest: TestPlanManifest) -> dict[str, tuple[str, ...]]:
    return {
        rule.name: rule.suppresses for rule in manifest.impact_rules if rule.suppresses
    }


@dataclass
class CompileResult:
    topology: str
    impact: dict[str, Any]
    suppressions: dict[str, tuple[str, ...]] = field(default_factory=dict)


def compile_manifest(path: Path) -> CompileResult:
    manifest = load_manifest(path)
    return CompileResult(
        topology=render_topology(manifest),
        impact=render_impact(manifest),
        suppressions=suppression_map(manifest),
    )


def impact_json(impact: Mapping[str, Any]) -> str:
    return json.dumps(impact, indent=2, sort_keys=False) + "\n"


def write_outputs(
    result: CompileResult,
    *,
    topology_path: Path,
    impact_path: Path,
) -> None:
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    impact_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.write_text(result.topology, encoding="utf-8")
    impact_path.write_text(impact_json(result.impact), encoding="utf-8")


def check_outputs(
    result: CompileResult,
    *,
    topology_path: Path,
    impact_path: Path,
) -> list[str]:
    errors: list[str] = []
    if not topology_path.is_file():
        errors.append(f"missing topology: {topology_path}")
    elif topology_path.read_text(encoding="utf-8") != result.topology:
        errors.append(f"stale topology: {topology_path}")
    if not impact_path.is_file():
        errors.append(f"missing impact: {impact_path}")
    elif impact_path.read_text(encoding="utf-8") != impact_json(result.impact):
        errors.append(f"stale impact: {impact_path}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--impact", type=Path, default=DEFAULT_IMPACT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when generated outputs are missing or stale.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write topology.toml and ci-impact.json from the manifest.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = compile_manifest(args.manifest)
    if args.check:
        errors = check_outputs(
            result, topology_path=args.topology, impact_path=args.impact
        )
        if errors:
            for error in errors:
                print(error)
            return 1
        print("test plan projections are up to date")
        return 0
    if args.write:
        write_outputs(
            result, topology_path=args.topology, impact_path=args.impact
        )
        print(f"wrote {args.topology}")
        print(f"wrote {args.impact}")
        return 0
    print(result.topology)
    print(impact_json(result.impact), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
