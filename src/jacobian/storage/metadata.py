"""SQLite artifact metadata, identities, and descriptor lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jacobian.canonical import CanonicalLimits, canonicalize_json, loads_strict_json
from jacobian.contracts.artifacts import ArtifactManifest, ArtifactPutResult
from jacobian.persistence import PersistenceCorruptionError, decode_persisted_model
from jacobian.storage.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    StorageCorruptionError,
    StorageError,
    StorageLimitError,
)
from jacobian.storage.identity import (
    BOOTSTRAP_SCHEMA_URI,
    BOOTSTRAP_SEMANTICS_URI,
    CANONICALIZER_DIGEST,
    OBJECT_FORMAT_VERSION,
    _ArtifactIdentity,
    _prepare_artifact_identity,
    digest_from_uri,
    framed_digest,
)
from jacobian.storage.models import StoredArtifact

if TYPE_CHECKING:
    from jacobian.storage.blobs import FilesystemBlobStore
    from jacobian.storage.models import StorageLimits
    from jacobian.storage.transactions import ArtifactTransactions


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    canonical_bytes: bytes
    identity: _ArtifactIdentity


class ArtifactMetadataStore:
    """Register, identify, persist, and validate immutable artifact metadata."""

    def __init__(
        self,
        *,
        limits: StorageLimits,
        canonical_limits: CanonicalLimits,
        transactions: ArtifactTransactions,
        blobs: FilesystemBlobStore,
    ) -> None:
        self._limits = limits
        self._canonical_limits = canonical_limits
        self._transactions = transactions
        self._blobs = blobs

    def _artifact_exists(self, artifact_uri: str) -> bool:
        with self._transactions.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM artifacts WHERE artifact_uri = ?",
                (artifact_uri,),
            ).fetchone()
        return row is not None

    def register_descriptor(
        self,
        *,
        kind: str,
        name: str,
        version: str,
        definition: Any,
    ) -> str:
        """Register an operator-owned infrastructure descriptor."""

        summary = self._descriptor_summary(kind=kind, name=name, version=version)
        self._validate_put_request(
            schema_uri=BOOTSTRAP_SCHEMA_URI,
            semantics_uri=BOOTSTRAP_SEMANTICS_URI,
            parents=(),
            summary=summary,
            allow_bootstrap_references=True,
        )
        canonical_bytes = self._canonicalize_descriptor(
            kind=kind,
            name=name,
            version=version,
            definition=definition,
        )
        self._validate_artifact_size(canonical_bytes)
        prepared = self._prepare_identity(
            schema_uri=BOOTSTRAP_SCHEMA_URI,
            semantics_uri=BOOTSTRAP_SEMANTICS_URI,
            canonical_bytes=canonical_bytes,
            parents=(),
            summary=summary,
        )
        return self._commit_prepared(prepared).artifact_uri

    def descriptor_uri(
        self,
        *,
        kind: str,
        name: str,
        version: str,
        definition: Any,
    ) -> str:
        """Return the deterministic URI for an infrastructure descriptor.

        This read-only calculation shares canonical preparation with descriptor
        registration but does not apply repository commit limits.
        """

        summary = self._descriptor_summary(kind=kind, name=name, version=version)
        canonical_bytes = self._canonicalize_descriptor(
            kind=kind,
            name=name,
            version=version,
            definition=definition,
        )
        prepared = self._prepare_identity(
            schema_uri=BOOTSTRAP_SCHEMA_URI,
            semantics_uri=BOOTSTRAP_SEMANTICS_URI,
            canonical_bytes=canonical_bytes,
            parents=(),
            summary=summary,
        )
        return prepared.identity.artifact_uri

    @staticmethod
    def _descriptor_summary(*, kind: str, name: str, version: str) -> str:
        if kind not in {"schema", "semantics", "canonicalizer", "implementation"}:
            raise ValueError(f"unsupported descriptor kind: {kind!r}")
        return f"{kind}: {name}@{version}"

    def _canonicalize_descriptor(
        self,
        *,
        kind: str,
        name: str,
        version: str,
        definition: Any,
    ) -> bytes:
        return canonicalize_json(
            {
                "descriptor_version": "1",
                "kind": kind,
                "name": name,
                "version": version,
                "definition": definition,
            },
            limits=self._canonical_limits,
        )

    def get_descriptor(
        self,
        artifact_uri: str,
        *,
        expected_kind: str | None = None,
    ) -> dict[str, Any]:
        """Load an infrastructure descriptor and optionally require its kind."""

        artifact = self.get(artifact_uri)
        if (
            artifact.manifest.schema_uri != BOOTSTRAP_SCHEMA_URI
            or artifact.manifest.semantics_uri != BOOTSTRAP_SEMANTICS_URI
            or not isinstance(artifact.payload, dict)
            or artifact.payload.get("descriptor_version") != "1"
        ):
            raise StorageError(f"artifact is not a system descriptor: {artifact_uri}")
        kind = artifact.payload.get("kind")
        if expected_kind is not None and kind != expected_kind:
            raise StorageError(
                f"descriptor kind {kind!r} does not match {expected_kind!r}"
            )
        return artifact.payload

    def put(
        self,
        *,
        schema_uri: str,
        semantics_uri: str,
        payload: Any,
        parents: tuple[str, ...] | list[str] = (),
        summary: str = "",
    ) -> ArtifactPutResult:
        """Commit canonical content whose identity binds schema and semantics."""

        return self._put(
            schema_uri=schema_uri,
            semantics_uri=semantics_uri,
            payload=payload,
            parents=tuple(parents),
            summary=summary,
            allow_bootstrap_references=False,
        )

    def _put(
        self,
        *,
        schema_uri: str,
        semantics_uri: str,
        payload: Any,
        parents: tuple[str, ...],
        summary: str,
        allow_bootstrap_references: bool,
    ) -> ArtifactPutResult:
        self._validate_put_request(
            schema_uri=schema_uri,
            semantics_uri=semantics_uri,
            parents=parents,
            summary=summary,
            allow_bootstrap_references=allow_bootstrap_references,
        )
        canonical_bytes = canonicalize_json(payload, limits=self._canonical_limits)
        self._validate_artifact_size(canonical_bytes)
        prepared = self._prepare_identity(
            schema_uri=schema_uri,
            semantics_uri=semantics_uri,
            canonical_bytes=canonical_bytes,
            parents=parents,
            summary=summary,
        )
        return self._commit_prepared(prepared)

    def _validate_put_request(
        self,
        *,
        schema_uri: str,
        semantics_uri: str,
        parents: tuple[str, ...],
        summary: str,
        allow_bootstrap_references: bool,
    ) -> None:
        if len(summary) > self._limits.max_summary_chars:
            raise StorageLimitError("artifact summary exceeds the configured limit")
        if len(parents) > self._limits.max_parents:
            raise StorageLimitError(
                "artifact parent count exceeds the configured limit"
            )
        if len(set(parents)) != len(parents):
            raise StorageError("artifact parents must be unique")
        if not allow_bootstrap_references:
            for reference in (schema_uri, semantics_uri, *parents):
                digest_from_uri(reference)
                if not self._artifact_exists(reference):
                    raise ArtifactNotFoundError(
                        f"referenced artifact is not committed: {reference}"
                    )

    def _validate_artifact_size(self, canonical_bytes: bytes) -> None:
        if len(canonical_bytes) > self._limits.max_artifact_bytes:
            raise StorageLimitError("artifact exceeds the configured size limit")

    def _prepare_identity(
        self,
        *,
        schema_uri: str,
        semantics_uri: str,
        canonical_bytes: bytes,
        parents: tuple[str, ...],
        summary: str,
    ) -> _PreparedArtifact:
        """Calculate an identity from already canonical artifact bytes."""

        return _PreparedArtifact(
            canonical_bytes=canonical_bytes,
            identity=_prepare_artifact_identity(
                canonical_limits=self._canonical_limits,
                schema_uri=schema_uri,
                semantics_uri=semantics_uri,
                canonical_bytes=canonical_bytes,
                parents=parents,
                summary=summary,
            ),
        )

    def _commit_prepared(self, prepared: _PreparedArtifact) -> ArtifactPutResult:
        """Persist one fully validated artifact without recomputing its identity."""

        canonical_bytes = prepared.canonical_bytes
        identity = prepared.identity
        manifest = identity.manifest

        # Re-registering identical content is common while assembling built-in
        # portfolios. Validate the committed artifact before returning so the
        # idempotent path avoids both blob publication and metadata writes
        # without allowing missing or corrupted content to be silently healed.
        if self._artifact_exists(identity.artifact_uri):
            self.get(identity.artifact_uri)
            return ArtifactPutResult(
                artifact_uri=identity.artifact_uri,
                object_digest=identity.object_digest,
                manifest_digest=identity.manifest_digest,
                canonicalizer_digest=CANONICALIZER_DIGEST,
            )

        self._blobs.write(canonical_bytes)
        self._blobs.write(identity.manifest_bytes)

        with self._transactions.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO artifacts (
                    artifact_uri,
                    manifest_digest,
                    object_digest,
                    payload_digest,
                    schema_uri,
                    semantics_uri,
                    canonicalizer_digest,
                    summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity.artifact_uri,
                    identity.manifest_digest,
                    identity.object_digest,
                    manifest.payload_digest,
                    manifest.schema_uri,
                    manifest.semantics_uri,
                    CANONICALIZER_DIGEST,
                    manifest.summary,
                ),
            )
            for position, parent_uri in enumerate(manifest.parents):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO artifact_parents (
                        artifact_uri, position, parent_uri
                    ) VALUES (?, ?, ?)
                    """,
                    (identity.artifact_uri, position, parent_uri),
                )

        return ArtifactPutResult(
            artifact_uri=identity.artifact_uri,
            object_digest=identity.object_digest,
            manifest_digest=identity.manifest_digest,
            canonicalizer_digest=CANONICALIZER_DIGEST,
        )

    def get(self, artifact_uri: str) -> StoredArtifact:
        """Load an artifact after replaying its content and manifest digests."""

        manifest_digest = digest_from_uri(artifact_uri)
        committed_references: set[str] = set()
        with self._transactions.connection() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_uri = ?",
                (artifact_uri,),
            ).fetchone()
            parent_rows = connection.execute(
                """
                SELECT
                    parent.parent_uri,
                    committed.artifact_uri AS committed_parent_uri
                FROM artifact_parents AS parent
                LEFT JOIN artifacts AS committed
                    ON committed.artifact_uri = parent.parent_uri
                WHERE parent.artifact_uri = ?
                ORDER BY parent.position
                """,
                (artifact_uri,),
            ).fetchall()
            if row is not None:
                committed_references = {
                    str(reference["artifact_uri"])
                    for reference in connection.execute(
                        """
                        SELECT artifact_uri
                        FROM artifacts
                        WHERE artifact_uri IN (?, ?)
                        """,
                        (row["schema_uri"], row["semantics_uri"]),
                    ).fetchall()
                }
        if row is None:
            raise ArtifactNotFoundError(f"artifact is not committed: {artifact_uri}")

        manifest_bytes = self._blobs.read(manifest_digest)
        try:
            manifest = decode_persisted_model(
                ArtifactManifest,
                manifest_bytes,
                record_kind="artifact_manifest",
                record_id=artifact_uri,
                field="manifest_json",
            )
        except PersistenceCorruptionError as exc:
            raise StorageCorruptionError(exc) from exc
        database_parents = tuple(parent["parent_uri"] for parent in parent_rows)
        if any(parent["committed_parent_uri"] is None for parent in parent_rows):
            raise ArtifactIntegrityError("manifest parent is not committed")
        if manifest.parents != database_parents:
            raise ArtifactIntegrityError("manifest parents differ from metadata")
        if (
            manifest_digest != row["manifest_digest"]
            or manifest.object_digest != row["object_digest"]
            or manifest.payload_digest != row["payload_digest"]
            or manifest.schema_uri != row["schema_uri"]
            or manifest.semantics_uri != row["semantics_uri"]
            or manifest.canonicalizer_digest != row["canonicalizer_digest"]
            or manifest.summary != row["summary"]
        ):
            raise ArtifactIntegrityError("manifest differs from committed metadata")
        if (
            manifest.schema_uri,
            manifest.semantics_uri,
        ) != (BOOTSTRAP_SCHEMA_URI, BOOTSTRAP_SEMANTICS_URI) and {
            manifest.schema_uri,
            manifest.semantics_uri,
        } != committed_references:
            raise ArtifactIntegrityError(
                "manifest schema or semantics is not committed"
            )

        canonical_bytes = self._blobs.read(manifest.payload_digest)
        recomputed_object_digest = framed_digest(
            OBJECT_FORMAT_VERSION,
            (
                manifest.schema_uri.encode(),
                manifest.semantics_uri.encode(),
                manifest.canonicalizer_digest.encode(),
                canonical_bytes,
            ),
        )
        if recomputed_object_digest != manifest.object_digest:
            raise ArtifactIntegrityError("mathematical object digest mismatch")
        payload = loads_strict_json(canonical_bytes, limits=self._canonical_limits)
        return StoredArtifact(
            artifact_uri=artifact_uri,
            manifest=manifest,
            payload=payload,
            canonical_bytes=canonical_bytes,
        )

    def find_by_object_digest(self, object_digest: str) -> tuple[str, ...]:
        """Return every artifact URI carrying a mathematical object digest."""

        with self._transactions.connection() as connection:
            rows = connection.execute(
                """
                SELECT artifact_uri
                FROM artifacts
                WHERE object_digest = ?
                ORDER BY artifact_uri
                """,
                (object_digest,),
            ).fetchall()
        return tuple(row["artifact_uri"] for row in rows)
