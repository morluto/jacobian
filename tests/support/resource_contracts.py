"""Typed resource contracts for pytest fixtures.

Fixtures that own isolation-sensitive resources declare them here so collection
and planners can validate transitive closures without stringly-typed guesses.

Contracts bind to the fixture **function identity** (module + qualname + code
object). Name lookup remains a convenience for inventory; collection enforcement
prefers the resolved FixtureDef function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


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
    module: str | None = None
    qualname: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


_RESOURCE_BY_NAME: dict[str, ResourceFixtureContract] = {}
_RESOURCE_BY_IDENTITY: dict[tuple[str, str, int], ResourceFixtureContract] = {}


def _function_identity(func: Callable[..., Any]) -> tuple[str, str, int]:
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or getattr(func, "__name__", "")
    code = getattr(func, "__code__", None)
    code_id = id(code) if code is not None else id(func)
    return (module, qualname, code_id)


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
        identity = _function_identity(func)
        contract = ResourceFixtureContract(
            name=func.__name__,
            resources=frozenset(resources),
            isolation=isolation,
            share_scope=share_scope,
            profile_key=profile_key,
            setup_affinity=setup_affinity or profile_key,
            module=identity[0],
            qualname=identity[1],
            extras=dict(extras),
        )
        _RESOURCE_BY_IDENTITY[identity] = contract
        _RESOURCE_BY_NAME[func.__name__] = contract
        func.__jacobian_resource_contract__ = contract  # type: ignore[attr-defined]
        return func

    return decorator


def resource_contract(name: str) -> ResourceFixtureContract | None:
    """Look up a contract by fixture name (last registration wins)."""

    return _RESOURCE_BY_NAME.get(name)


def resource_contract_for_function(
    func: Callable[..., Any],
) -> ResourceFixtureContract | None:
    """Look up a contract by fixture function identity."""

    attached = getattr(func, "__jacobian_resource_contract__", None)
    if isinstance(attached, ResourceFixtureContract):
        return attached
    return _RESOURCE_BY_IDENTITY.get(_function_identity(func))


def registered_resource_fixtures() -> dict[str, ResourceFixtureContract]:
    return dict(_RESOURCE_BY_NAME)


def registered_resource_identities() -> dict[
    tuple[str, str, int], ResourceFixtureContract
]:
    return dict(_RESOURCE_BY_IDENTITY)
