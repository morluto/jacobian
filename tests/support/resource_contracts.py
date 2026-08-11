"""Typed resource contracts for pytest fixtures.

Fixtures that own isolation-sensitive resources declare them here so collection
and planners can validate transitive closures without stringly-typed guesses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_RESOURCE_REGISTRY: dict[str, ResourceFixtureContract] = {}


class ResourceKind(StrEnum):
    COMPLETE_RUNTIME = "complete-runtime"
    AUTHORIZED_CHECKERS = "authorized-checkers"
    SQLITE = "sqlite"
    PROCESS_GROUP = "process-group"
    MCP = "mcp"
    PROVIDER = "provider"
    LEAN = "lean"
    NETWORK = "network"


class IsolationClass(StrEnum):
    READ_ONLY = "read-only"
    PRIVATE_MUTABLE = "private-mutable"
    LIFECYCLE_OWNER = "lifecycle-owner"


@dataclass(frozen=True, slots=True)
class ResourceFixtureContract:
    name: str
    resources: frozenset[ResourceKind]
    isolation: IsolationClass
    share_scope: str = "test"
    profile_key: str | None = None
    setup_affinity: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


def resource_fixture(
    *,
    resources: set[ResourceKind] | frozenset[ResourceKind],
    isolation: IsolationClass,
    share_scope: str = "test",
    profile_key: str | None = None,
    setup_affinity: str | None = None,
    **extras: str,
) -> Callable[[F], F]:
    """Record a resource contract on a fixture function."""

    def decorator(func: F) -> F:
        contract = ResourceFixtureContract(
            name=func.__name__,
            resources=frozenset(resources),
            isolation=isolation,
            share_scope=share_scope,
            profile_key=profile_key,
            setup_affinity=setup_affinity or profile_key,
            extras=dict(extras),
        )
        _RESOURCE_REGISTRY[func.__name__] = contract
        func.__jacobian_resource_contract__ = contract
        return func

    return decorator


def resource_contract(name: str) -> ResourceFixtureContract | None:
    return _RESOURCE_REGISTRY.get(name)


def registered_resource_fixtures() -> dict[str, ResourceFixtureContract]:
    return dict(_RESOURCE_REGISTRY)
