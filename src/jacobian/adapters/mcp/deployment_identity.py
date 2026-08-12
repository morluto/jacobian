"""Immutable deployment identity advertised by managed MCP releases."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jacobian import __version__

DEPLOYMENT_REVISION_FILE_ENV = "JACOBIAN_DEPLOYMENT_REVISION_FILE"
_FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


class DeploymentIdentity(BaseModel):
    """Revision identity captured once from an immutable managed release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_version: str = Field(min_length=1)
    evidence: Literal["release-marker"] = "release-marker"


def load_deployment_identity(
    *,
    environment: Mapping[str, str] | None = None,
    implementation_path: Path | None = None,
) -> DeploymentIdentity | None:
    """Load a root-owned release marker and bind it to the running package tree."""

    source_environment = os.environ if environment is None else environment
    configured = source_environment.get(DEPLOYMENT_REVISION_FILE_ENV)
    if configured is None:
        return None
    marker = Path(configured)
    if not marker.is_absolute():
        raise RuntimeError(
            f"{DEPLOYMENT_REVISION_FILE_ENV} must name an absolute release marker"
        )
    try:
        resolved_marker = marker.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "the configured deployment revision marker is unavailable"
        ) from exc
    package_path = (implementation_path or Path(__file__)).resolve(strict=True)
    release_root = resolved_marker.parent
    if not package_path.is_relative_to(release_root):
        raise RuntimeError(
            "the deployment revision marker does not contain the running package"
        )
    try:
        with resolved_marker.open("rb") as stream:
            marker_stat = os.fstat(stream.fileno())
            payload = stream.read(42)
    except OSError as exc:
        raise RuntimeError("the deployment revision marker could not be read") from exc
    if (
        not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_size != 41
        or len(payload) != 41
    ):
        raise RuntimeError("the deployment revision marker is not canonical")
    try:
        revision = payload.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError as exc:
        raise RuntimeError("the deployment revision marker is not canonical") from exc
    if (
        payload != f"{revision}\n".encode("ascii")
        or _FULL_GIT_REVISION.fullmatch(revision) is None
    ):
        raise RuntimeError("the deployment revision marker is not canonical")
    return DeploymentIdentity(revision=revision, package_version=__version__)
