"""Canonical identities used by blob and metadata storage components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final

from jacobian.canonical import CanonicalLimits, canonicalize_json, sha256_digest
from jacobian.contracts.artifacts import ArtifactManifest
from jacobian.storage.errors import ArtifactNotFoundError

OBJECT_FORMAT_VERSION: Final = b"jacobian.object.v1"
_CANONICALIZER_NAME: Final = b"jacobian.rfc8785+nfc+exact-rational.v1"
CANONICALIZER_DIGEST: Final = (
    "sha256:" + hashlib.sha256(_CANONICALIZER_NAME).hexdigest()
)
BOOTSTRAP_SCHEMA_URI: Final = (
    "artifact://sha256/" + hashlib.sha256(b"jacobian.bootstrap.schema.v1").hexdigest()
)
BOOTSTRAP_SEMANTICS_URI: Final = (
    "artifact://sha256/"
    + hashlib.sha256(b"jacobian.bootstrap.semantics.v1").hexdigest()
)


@dataclass(frozen=True, slots=True)
class _ArtifactIdentity:
    """Canonical artifact identity plus the manifest bytes that define it."""

    artifact_uri: str
    object_digest: str
    manifest_digest: str
    manifest: ArtifactManifest
    manifest_bytes: bytes


def uri_from_digest(digest: str) -> str:
    return "artifact://sha256/" + digest.removeprefix("sha256:")


def digest_from_uri(uri: str) -> str:
    prefix = "artifact://sha256/"
    if not uri.startswith(prefix):
        raise ArtifactNotFoundError(f"invalid artifact URI: {uri!r}")
    value = uri.removeprefix(prefix)
    if len(value) != 64 or any(
        char not in "0123456789abcdef"  # pragma: allowlist secret
        for char in value
    ):
        raise ArtifactNotFoundError(f"invalid artifact URI: {uri!r}")
    return "sha256:" + value


def framed_digest(tag: bytes, parts: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"jacobian\x00")
    digest.update(len(tag).to_bytes(8, "big"))
    digest.update(tag)
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return "sha256:" + digest.hexdigest()


def _prepare_artifact_identity(
    *,
    canonical_limits: CanonicalLimits,
    schema_uri: str,
    semantics_uri: str,
    canonical_bytes: bytes,
    parents: tuple[str, ...],
    summary: str,
) -> _ArtifactIdentity:
    """Prepare an artifact identity and its canonical manifest projection."""

    object_digest = framed_digest(
        OBJECT_FORMAT_VERSION,
        (
            schema_uri.encode(),
            semantics_uri.encode(),
            CANONICALIZER_DIGEST.encode(),
            canonical_bytes,
        ),
    )
    manifest = ArtifactManifest(
        object_digest=object_digest,
        payload_digest=sha256_digest(canonical_bytes),
        schema_uri=schema_uri,
        semantics_uri=semantics_uri,
        canonicalizer_digest=CANONICALIZER_DIGEST,
        parents=tuple(sorted(parents)),
        summary=summary,
    )
    manifest_bytes = canonicalize_json(
        manifest.model_dump(mode="json"), limits=canonical_limits
    )
    manifest_digest = sha256_digest(manifest_bytes)
    return _ArtifactIdentity(
        artifact_uri=uri_from_digest(manifest_digest),
        object_digest=object_digest,
        manifest_digest=manifest_digest,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
    )


def descriptor_identity_uri(
    *,
    kind: str,
    name: str,
    version: str,
    definition: Any,
    canonical_limits: CanonicalLimits | None = None,
) -> str:
    """Return the content-addressed URI for an infrastructure descriptor.

    This is the store-free identity that ``ArtifactMetadataStore.descriptor_uri``
    and ``register_descriptor`` assign to the same name, version, and definition.
    """

    if kind not in {"schema", "semantics", "canonicalizer", "implementation"}:
        raise ValueError(f"unsupported descriptor kind: {kind!r}")
    limits = canonical_limits or CanonicalLimits()
    canonical_bytes = canonicalize_json(
        {
            "descriptor_version": "1",
            "kind": kind,
            "name": name,
            "version": version,
            "definition": definition,
        },
        limits=limits,
    )
    uri, _object_digest, _manifest_digest = artifact_identity(
        canonical_limits=limits,
        schema_uri=BOOTSTRAP_SCHEMA_URI,
        semantics_uri=BOOTSTRAP_SEMANTICS_URI,
        canonical_bytes=canonical_bytes,
        parents=(),
        summary=f"{kind}: {name}@{version}",
    )
    return uri


def artifact_identity(
    *,
    canonical_limits: CanonicalLimits,
    schema_uri: str,
    semantics_uri: str,
    canonical_bytes: bytes,
    parents: tuple[str, ...],
    summary: str,
) -> tuple[str, str, str]:
    """Calculate an artifact identity without touching storage."""

    identity = _prepare_artifact_identity(
        canonical_limits=canonical_limits,
        schema_uri=schema_uri,
        semantics_uri=semantics_uri,
        canonical_bytes=canonical_bytes,
        parents=parents,
        summary=summary,
    )
    return (
        identity.artifact_uri,
        identity.object_digest,
        identity.manifest_digest,
    )


__all__ = [
    "BOOTSTRAP_SCHEMA_URI",
    "BOOTSTRAP_SEMANTICS_URI",
    "CANONICALIZER_DIGEST",
    "OBJECT_FORMAT_VERSION",
    "artifact_identity",
    "descriptor_identity_uri",
    "digest_from_uri",
    "framed_digest",
    "uri_from_digest",
]
