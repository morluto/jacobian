"""Typed locators for compiled catalog entries.

Declaration-module rows may still use a bare module path. Family rows are JSON
objects recorded at compile time. Legacy ``family:`` prefixes fail closed until
``jacobian update``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from jacobian.operation_catalog import OperationCatalogError


@dataclass(frozen=True, slots=True)
class ModuleLocator:
    """Python module that reconstructs one declaration-owned operation."""

    module: str
    symbol: str | None = None
    kind: Literal["module"] = "module"


@dataclass(frozen=True, slots=True)
class FamilyLocator:
    """Family table that binds one resource-backed operation."""

    family: str
    kind: Literal["family"] = "family"


type OperationLocator = ModuleLocator | FamilyLocator


def encode_locator(locator: OperationLocator) -> str:
    """Serialize one locator for the compiled catalog overlay."""

    if isinstance(locator, FamilyLocator):
        payload: dict[str, str] = {"kind": "family", "family": locator.family}
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload = {"kind": "module", "module": locator.module}
    if locator.symbol is not None:
        payload["symbol"] = locator.symbol
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def decode_locator(raw: str) -> OperationLocator:
    """Parse one persisted locator. Legacy ``family:`` prefixes are stale."""

    if raw.startswith("family:"):
        raise OperationCatalogError(
            "operation catalog locator is stale; run `jacobian update`"
        )
    if not raw.startswith("{"):
        return ModuleLocator(module=raw)
    try:
        payload = cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError as exc:
        raise OperationCatalogError(
            "operation catalog locator is stale; run `jacobian update`"
        ) from exc
    kind = payload.get("kind")
    if kind == "family":
        family = payload.get("family")
        if not isinstance(family, str) or not family:
            raise OperationCatalogError(
                "operation catalog locator is stale; run `jacobian update`"
            )
        return FamilyLocator(family=family)
    if kind == "module":
        module = payload.get("module")
        if not isinstance(module, str) or not module:
            raise OperationCatalogError(
                "operation catalog locator is stale; run `jacobian update`"
            )
        symbol = payload.get("symbol")
        if symbol is not None and not isinstance(symbol, str):
            raise OperationCatalogError(
                "operation catalog locator is stale; run `jacobian update`"
            )
        return ModuleLocator(module=module, symbol=symbol)
    raise OperationCatalogError(
        "operation catalog locator is stale; run `jacobian update`"
    )


__all__ = [
    "FamilyLocator",
    "ModuleLocator",
    "OperationLocator",
    "decode_locator",
    "encode_locator",
]
