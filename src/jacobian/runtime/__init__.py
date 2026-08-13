"""Construction and ownership of a Jacobian application runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jacobian.runtime.config import CheckerAuthorityMode, RuntimeOptions

if TYPE_CHECKING:
    from jacobian.operation_service import OperationPolicy
    from jacobian.runtime.model import JacobianRuntime


def create_runtime(
    root: str | Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
    operation_exclusions: frozenset[str] = frozenset(),
    operation_policy: OperationPolicy | None = None,
) -> JacobianRuntime:
    """Create the single owned runtime for ``root``."""

    from jacobian.composition import compose_runtime

    return compose_runtime(
        root,
        RuntimeOptions(
            checker_authority=checker_authority,
            operation_exclusions=operation_exclusions,
            operation_policy=operation_policy,
        ),
    )


__all__ = [
    "CheckerAuthorityMode",
    "RuntimeOptions",
    "create_runtime",
]
