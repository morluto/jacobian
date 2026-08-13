"""Validated configuration for one Jacobian runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operation_service import OperationPolicy


class CheckerAuthorityMode(StrEnum):
    """How a runtime obtains operator-owned checker authority."""

    NONE = "NONE"
    INSTALL_BUNDLED = "INSTALL_BUNDLED"
    HYDRATE_EXISTING = "HYDRATE_EXISTING"


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Immutable inputs that determine runtime composition."""

    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE
    operation_exclusions: frozenset[str] = frozenset()
    operation_policy: OperationPolicy | None = None
