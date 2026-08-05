"""Value types shared by the storage components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.contracts.artifacts import ArtifactManifest


@dataclass(frozen=True, slots=True)
class StorageLimits:
    """Local artifact and aggregate blob-size limits."""

    max_artifact_bytes: int = 10 * 1024 * 1024
    max_parents: int = 4096
    max_summary_chars: int = 512
    max_total_blob_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_parents < 3:
            raise ValueError("max_parents must be at least 3")


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """A verified-on-read manifest, payload, and canonical byte sequence."""

    artifact_uri: str
    manifest: ArtifactManifest
    payload: Any
    canonical_bytes: bytes


__all__ = ["StorageLimits", "StoredArtifact"]
