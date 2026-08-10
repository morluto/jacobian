"""Typed runtime profiles for complete-runtime tests.

Tests declare the minimum install/authority/mutability they need. Named
fixtures such as ``attached_complete_runtime`` remain thin wrappers over
``runtime_for``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime
from tests.support.state import copy_template

InstallationMode = Literal["FRESH", "ATTACH_TEMPLATE"]
StateAccess = Literal["READ_ONLY", "PRIVATE_MUTABLE", "LIFECYCLE_OWNER"]


@dataclass(frozen=True, slots=True)
class RuntimeTestProfile:
    """Minimum complete-runtime facts a test requires."""

    installation: InstallationMode
    checker_authority: CheckerAuthorityMode
    state_access: StateAccess
    bundles: frozenset[str] = frozenset()
    reference_sets: frozenset[str] = frozenset()
    providers: frozenset[str] = frozenset()
    background_work: bool = False

    def requires_authorized_checkers(self) -> bool:
        return self.checker_authority is not CheckerAuthorityMode.NONE


ATTACHED_COMPUTE = RuntimeTestProfile(
    installation="ATTACH_TEMPLATE",
    checker_authority=CheckerAuthorityMode.NONE,
    state_access="PRIVATE_MUTABLE",
)
ATTACHED_COMPUTE_READ_ONLY = RuntimeTestProfile(
    installation="ATTACH_TEMPLATE",
    checker_authority=CheckerAuthorityMode.NONE,
    state_access="READ_ONLY",
)
AUTHORIZED_VERIFY = RuntimeTestProfile(
    installation="ATTACH_TEMPLATE",
    checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    state_access="PRIVATE_MUTABLE",
)
AUTHORIZED_VERIFY_READ_ONLY = RuntimeTestProfile(
    installation="ATTACH_TEMPLATE",
    checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    state_access="READ_ONLY",
)
FRESH_LIFECYCLE = RuntimeTestProfile(
    installation="FRESH",
    checker_authority=CheckerAuthorityMode.NONE,
    state_access="LIFECYCLE_OWNER",
)


def open_runtime_for(
    profile: RuntimeTestProfile,
    *,
    tmp_path: Path,
    complete_portfolio_template: Path | None = None,
    authorized_portfolio_template: Path | None = None,
) -> Iterator[JacobianRuntime]:
    """Materialize a runtime matching ``profile``.

    ``ATTACH_TEMPLATE`` always copies into ``tmp_path``: opening a store mutates
    SQLite/layout, so the session template must stay immutable. ``READ_ONLY`` is
    a sharing contract for module-scoped fixtures—one private copy, tests must
    not write artifacts or durable store state through that runtime.
    """

    if profile.installation == "FRESH":
        if profile.state_access != "LIFECYCLE_OWNER":
            raise ValueError("FRESH installation requires LIFECYCLE_OWNER state access")
        runtime = create_runtime(
            tmp_path / "state",
            checker_authority=profile.checker_authority,
        )
    elif profile.checker_authority is CheckerAuthorityMode.NONE:
        if complete_portfolio_template is None:
            raise ValueError(
                "ATTACH_TEMPLATE without authority needs complete template"
            )
        state = copy_template(complete_portfolio_template, tmp_path / "state")
        runtime = create_runtime(state)
    else:
        if authorized_portfolio_template is None:
            raise ValueError("authorized profile needs authorized template")
        state = copy_template(authorized_portfolio_template, tmp_path / "state")
        runtime = create_runtime(
            state,
            checker_authority=profile.checker_authority,
        )
    try:
        yield runtime
    finally:
        runtime.close()
