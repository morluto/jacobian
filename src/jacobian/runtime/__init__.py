"""Construction and ownership of a Jacobian application runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions

if TYPE_CHECKING:
    from jacobian.capability_service import CapabilityPolicy
    from jacobian.runtime.model import JacobianRuntime


def create_runtime(
    root: str | Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
    capability_exclusions: frozenset[str] = frozenset(),
    capability_policy: CapabilityPolicy | None = None,
) -> JacobianRuntime:
    """Create the single owned runtime for ``root``."""

    from jacobian.runtime.model import JacobianRuntime

    return JacobianRuntime(
        root,
        RuntimeOptions(
            checker_authority=checker_authority,
            capability_exclusions=capability_exclusions,
            capability_policy=capability_policy,
        ),
    )


__all__ = [
    "CheckerAuthorityMode",
    "RuntimeOptions",
    "create_runtime",
]
