"""Compile semantic owner x resources into an execution profile.

Resource dominance wins over directory convenience: sqlite serializes, process
and lean lanes keep process-group-safe supervision, and setup affinity keys
label shared fixture cost for shard planners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Scheduler = Literal["none", "worksteal", "load"]

# Environment tags declared on lanes in tests/plan_manifest.toml.
_ENV_RESOURCE_MAP: dict[str, str] = {
    "sqlite": "sqlite",
    "process-group": "process-group",
    "mcp": "mcp",
    "provider-readiness": "provider",
    "lean-4.31.0": "lean",
    "mathlib": "lean",
}


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    name: str
    workers: int
    distribution: Scheduler
    timeout_seconds: int
    process_supervision: bool
    sqlite_serial: bool
    setup_affinity: str | None = None


_RESOURCE_DOMINANCE: dict[str, ExecutionProfile] = {
    "process-group": ExecutionProfile(
        name="process-isolated",
        workers=2,
        distribution="worksteal",
        timeout_seconds=120,
        process_supervision=True,
        sqlite_serial=False,
        setup_affinity="process-group",
    ),
    "sqlite": ExecutionProfile(
        name="sqlite-serial",
        workers=0,
        distribution="none",
        timeout_seconds=120,
        process_supervision=False,
        sqlite_serial=True,
        setup_affinity="sqlite",
    ),
    "lean": ExecutionProfile(
        name="lean-serial",
        workers=0,
        distribution="none",
        timeout_seconds=300,
        process_supervision=True,
        sqlite_serial=False,
        setup_affinity="lean",
    ),
    "mcp": ExecutionProfile(
        name="mcp-isolated",
        workers=2,
        distribution="worksteal",
        timeout_seconds=120,
        process_supervision=True,
        sqlite_serial=False,
        setup_affinity="mcp",
    ),
}


def resources_from_environment(tags: tuple[str, ...] | list[str]) -> frozenset[str]:
    return frozenset(
        _ENV_RESOURCE_MAP[tag] for tag in tags if tag in _ENV_RESOURCE_MAP
    )


def compile_execution_profile(
    *,
    semantic_owner: str,
    resources: set[str] | frozenset[str],
    default_workers: int = 2,
    default_timeout: int = 120,
    default_distribution: Scheduler = "worksteal",
) -> ExecutionProfile:
    """Apply resource dominance rules over a semantic lane default."""

    for resource in ("lean", "process-group", "sqlite", "mcp"):
        if resource in resources:
            return _RESOURCE_DOMINANCE[resource]
    return ExecutionProfile(
        name=f"{semantic_owner}-default",
        workers=default_workers,
        distribution=default_distribution,
        timeout_seconds=default_timeout,
        process_supervision="process-group" in resources,
        sqlite_serial="sqlite" in resources,
        setup_affinity=None,
    )


def profile_for_lane(
    *,
    name: str,
    required_environment: tuple[str, ...] | list[str],
    workers: int,
    distribution: str,
    timeout_seconds: int,
) -> ExecutionProfile:
    """Compile the lane's declared environment into an execution profile."""

    resources = resources_from_environment(tuple(required_environment))
    if not resources:
        scheduler: Scheduler
        if distribution == "none":
            scheduler = "none"
        elif distribution == "load":
            scheduler = "load"
        else:
            scheduler = "worksteal"
        return ExecutionProfile(
            name=f"{name}-default",
            workers=workers,
            distribution=scheduler,
            timeout_seconds=timeout_seconds,
            process_supervision=False,
            sqlite_serial=False,
            setup_affinity=None,
        )
    return compile_execution_profile(
        semantic_owner=name,
        resources=resources,
        default_workers=workers,
        default_timeout=timeout_seconds,
        default_distribution=(
            "none"
            if distribution == "none"
            else "load"
            if distribution == "load"
            else "worksteal"
        ),
    )


def validate_lane_against_profile(
    *,
    name: str,
    required_environment: tuple[str, ...] | list[str],
    workers: int,
    distribution: str,
    timeout_seconds: int,
) -> list[str]:
    """Return errors when a dominant resource lane disagrees with its profile."""

    resources = resources_from_environment(tuple(required_environment))
    dominant = next(
        (
            resource
            for resource in ("lean", "process-group", "sqlite", "mcp")
            if resource in resources
        ),
        None,
    )
    if dominant is None:
        return []
    profile = _RESOURCE_DOMINANCE[dominant]
    errors: list[str] = []
    if workers != profile.workers:
        errors.append(
            f"lane {name}: workers={workers} conflicts with execution profile "
            f"{profile.name} (workers={profile.workers})"
        )
    if distribution != profile.distribution:
        errors.append(
            f"lane {name}: distribution={distribution!r} conflicts with "
            f"execution profile {profile.name} "
            f"(distribution={profile.distribution!r})"
        )
    if timeout_seconds < profile.timeout_seconds:
        errors.append(
            f"lane {name}: timeout_seconds={timeout_seconds} is below execution "
            f"profile {profile.name} minimum ({profile.timeout_seconds})"
        )
    return errors
