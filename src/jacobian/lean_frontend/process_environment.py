"""Sanitized subprocess environments for pinned Lean tooling."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from jacobian.worker_environment import worker_environment


def lean_elan_worker_environment(
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a sanitized Lean worker environment with an explicit elan home."""

    variables = dict(overrides or {})
    variables["ELAN_HOME"] = os.environ.get(
        "ELAN_HOME",
        str(Path.home() / ".elan"),
    )
    return worker_environment(
        extra_variables=("PATH",),
        overrides=variables,
    )
