"""Operator-controlled checker authorization and revocation."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.checkers import (
    CheckerAuditEvent,
    CheckerRegistration,
    EvidenceKind,
)
from jacobian.implementation import ImplementationError, checker_source_digest
from jacobian.persistence import (
    PersistenceCorruptionError,
    PersistenceLock,
    decode_persisted_model,
)
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    require_provider_runtime_unchanged,
)
from jacobian.storage.repository import ArtifactRepository
from jacobian.storage.transactions import transaction_active_for

_LOGGER = logging.getLogger(__name__)


class CheckerRegistryError(RuntimeError):
    """Base checker registry failure."""


class CheckerNotFoundError(CheckerRegistryError):
    """No checker with this identifier is registered."""


class CheckerRevokedError(CheckerRegistryError):
    """The checker cannot originate a new verification record."""


class CheckerExecutableChangedError(CheckerRegistryError):
    """The installed checker bytes differ from the authorized bytes."""


class CheckerCompatibilityError(CheckerRegistryError):
    """The checker is not authorized for the requested evidence bindings."""


class _PolicyLockState(threading.local):
    """Per-thread nesting state for the cross-process policy lock."""

    def __init__(self) -> None:
        self.depth = 0


def compute_entrypoint_digest(entrypoint: str) -> str:
    try:
        return checker_source_digest(entrypoint)
    except ImplementationError as exc:
        _LOGGER.warning(
            "could not read checker implementation %s",
            entrypoint,
            exc_info=exc,
        )
        raise CheckerRegistryError(
            "The checker entrypoint could not be loaded. Check that the "
            "module:function entrypoint is installed, then retry."
        ) from exc


def _require_runtime_unchanged(runtime: CapabilityProviderRuntime | None) -> None:
    if runtime is None:
        return
    try:
        require_provider_runtime_unchanged(runtime)
    except (OSError, ProviderRuntimeError, ValueError) as exc:
        raise CheckerExecutableChangedError(
            "The checker runtime changed after authorization or is unavailable. "
            "Authorize the current runtime, then retry."
        ) from exc


class CheckerRegistry:
    """Persist operator authorization, compatibility, audit, and revocation."""

    def __init__(self, store: ArtifactRepository) -> None:
        self.store = store
        self.database_path = store.db_path
        self.policy_lock_path = self.database_path.with_name(
            self.database_path.name + ".checker-policy.lock"
        )
        self._policy_lock = PersistenceLock(self.policy_lock_path)
        self._policy_lock_state = _PolicyLockState()
        self.bind_existing_when_omitted = False

    @contextmanager
    def policy_transaction(self) -> Iterator[None]:
        """Hold checker policy authority across related durable writes."""

        if self._policy_lock_state.depth:
            self._policy_lock_state.depth += 1
            try:
                yield
            finally:
                self._policy_lock_state.depth -= 1
            return
        if transaction_active_for(self.database_path):
            raise CheckerRegistryError(
                "checker policy must be locked before the store transaction"
            )

        with self._policy_lock.hold():
            self._policy_lock_state.depth = 1
            try:
                yield
            finally:
                self._policy_lock_state.depth = 0

    @contextmanager
    def _policy_write_lock(self) -> Iterator[None]:
        """Acquire policy before SQLite or reuse an existing outer lock."""

        with self.policy_transaction():
            yield

    @contextmanager
    def verification_guard(
        self,
        checker_id: str,
        *,
        expected_digest: str,
    ) -> Iterator[CheckerRegistration]:
        """Prevent revocation while a verified record is committed."""

        with self._policy_write_lock():
            registration = self.require_active(checker_id)
            if registration.executable_digest != expected_digest:
                raise CheckerExecutableChangedError(
                    "The checker changed before verification was saved. Authorize the "
                    "current checker version, then run verification again."
                )
            yield registration

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self.store.connection() as connection:
            yield connection

    def authorize(
        self,
        *,
        name: str,
        entrypoint: str,
        evidence_kind: EvidenceKind | str,
        format_id: str,
        format_version: str,
        claim_schema_uris: tuple[str, ...],
        semantics_uris: tuple[str, ...],
        candidate_schema_uris: tuple[str, ...],
        target_schema_uris: tuple[str, ...] = (),
        target_semantics_uris: tuple[str, ...] = (),
        provider_runtime: CapabilityProviderRuntime | None = None,
        reason: str = "operator authorization",
    ) -> CheckerRegistration:
        """Authorize one measured checker for explicit evidence compatibility."""

        selected_kind = EvidenceKind(evidence_kind)
        scope_error = _compatibility_scope_error(
            evidence_kind=selected_kind,
            claim_schema_uris=claim_schema_uris,
            semantics_uris=semantics_uris,
            candidate_schema_uris=candidate_schema_uris,
            target_schema_uris=target_schema_uris,
            target_semantics_uris=target_semantics_uris,
        )
        if scope_error is not None:
            raise CheckerRegistryError(scope_error)
        _require_runtime_unchanged(provider_runtime)
        executable_digest = compute_entrypoint_digest(entrypoint)
        unbound_registration = CheckerRegistration(
            checker_id="checker://sha256/" + "0" * 64,
            name=name,
            entrypoint=entrypoint,
            executable_digest=executable_digest,
            provider_runtime=provider_runtime,
            evidence_kind=selected_kind,
            format_id=format_id,
            format_version=format_version,
            claim_schema_uris=tuple(sorted(claim_schema_uris)),
            semantics_uris=tuple(sorted(semantics_uris)),
            candidate_schema_uris=tuple(sorted(candidate_schema_uris)),
            target_schema_uris=tuple(sorted(target_schema_uris)),
            target_semantics_uris=tuple(sorted(target_semantics_uris)),
            authorized=True,
        )
        registration = unbound_registration.model_copy(
            update={"checker_id": _checker_identifier(unbound_registration)}
        )
        encoded = canonicalize_json(
            registration.model_dump(mode="json", exclude_none=True)
        )

        with self._policy_write_lock(), self._connection() as connection:
            existing = connection.execute(
                """
                SELECT registration_json, authorized, executable_digest
                FROM checkers
                WHERE checker_id = ?
                """,
                (registration.checker_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO checkers (
                        checker_id,
                        registration_json,
                        authorized,
                        executable_digest
                    ) VALUES (?, ?, 1, ?)
                    """,
                    (
                        registration.checker_id,
                        encoded,
                        executable_digest,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO checker_audit (checker_id, action, reason)
                    VALUES (?, 'AUTHORIZED', ?)
                    """,
                    (registration.checker_id, reason),
                )
            elif (
                bytes(existing["registration_json"]) != encoded
                or existing["executable_digest"] != executable_digest
            ):
                raise CheckerRegistryError(
                    "This checker conflicts with an existing registration. Register "
                    "the changed implementation as a new checker version."
                )
            elif not bool(existing["authorized"]):
                raise CheckerRevokedError(
                    "This checker version is revoked and cannot be reauthorized. "
                    "Authorize the current implementation as a new checker version."
                )
        return registration

    def bind_existing(
        self,
        *,
        name: str,
        entrypoint: str,
        evidence_kind: EvidenceKind | str,
        format_id: str,
        format_version: str,
        claim_schema_uris: tuple[str, ...],
        semantics_uris: tuple[str, ...],
        candidate_schema_uris: tuple[str, ...],
        target_schema_uris: tuple[str, ...] = (),
        target_semantics_uris: tuple[str, ...] = (),
        provider_runtime: CapabilityProviderRuntime | None = None,
    ) -> str | None:
        """Return an already-authorized checker_id without writing authorization.

        Used to reconstitute process-local adapters from a store that already
        contains operator-authorized checkers. Missing or revoked checkers fail
        closed as ``None`` rather than authorizing.
        """

        selected_kind = EvidenceKind(evidence_kind)
        scope_error = _compatibility_scope_error(
            evidence_kind=selected_kind,
            claim_schema_uris=claim_schema_uris,
            semantics_uris=semantics_uris,
            candidate_schema_uris=candidate_schema_uris,
            target_schema_uris=target_schema_uris,
            target_semantics_uris=target_semantics_uris,
        )
        if scope_error is not None:
            raise CheckerRegistryError(scope_error)
        if provider_runtime is not None and (
            provider_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
            or provider_runtime.digest is None
            or provider_runtime.digest_kind is None
        ):
            # Match authorization gates that omit unavailable providers.
            return None
        try:
            _require_runtime_unchanged(provider_runtime)
        except CheckerExecutableChangedError:
            return None
        executable_digest = compute_entrypoint_digest(entrypoint)
        unbound_registration = CheckerRegistration(
            checker_id="checker://sha256/" + "0" * 64,
            name=name,
            entrypoint=entrypoint,
            executable_digest=executable_digest,
            provider_runtime=provider_runtime,
            evidence_kind=selected_kind,
            format_id=format_id,
            format_version=format_version,
            claim_schema_uris=tuple(sorted(claim_schema_uris)),
            semantics_uris=tuple(sorted(semantics_uris)),
            candidate_schema_uris=tuple(sorted(candidate_schema_uris)),
            target_schema_uris=tuple(sorted(target_schema_uris)),
            target_semantics_uris=tuple(sorted(target_semantics_uris)),
            authorized=True,
        )
        checker_id = _checker_identifier(unbound_registration)
        try:
            registration = self.get(checker_id)
        except CheckerNotFoundError:
            return None
        if not registration.authorized:
            return None
        return registration.checker_id

    def get(self, checker_id: str) -> CheckerRegistration:
        """Return a checker registration, including its revocation state."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT registration_json, authorized, executable_digest
                FROM checkers
                WHERE checker_id = ?
                """,
                (checker_id,),
            ).fetchone()
        if row is None:
            raise CheckerNotFoundError(
                "The checker is not registered. Review the reference contract, choose "
                "an authorized checker_id, and retry."
            )
        try:
            registration = decode_persisted_model(
                CheckerRegistration,
                bytes(row["registration_json"]),
                record_kind="checker_registration",
                record_id=checker_id,
                field="registration_json",
            ).model_copy(update={"authorized": bool(row["authorized"])})
        except PersistenceCorruptionError as exc:
            raise CheckerRegistryError(
                "Checker registry data is invalid. Restore the Jacobian state "
                "directory from a trusted copy before retrying."
            ) from exc
        if (
            registration.checker_id != checker_id
            or _checker_identifier(registration) != checker_id
            or registration.executable_digest != row["executable_digest"]
        ):
            raise CheckerRegistryError(
                "Checker registry data is inconsistent. Restore the Jacobian state "
                "directory from a trusted copy before retrying."
            )
        return registration

    def require_active(self, checker_id: str) -> CheckerRegistration:
        """Return a checker only while it may create new verified records."""

        registration = self.get(checker_id)
        if not registration.authorized:
            raise CheckerRevokedError(
                "This checker is revoked. Select an active checker from the reference "
                "contract, then retry."
            )
        installed_digest = compute_entrypoint_digest(registration.entrypoint)
        if installed_digest != registration.executable_digest:
            raise CheckerExecutableChangedError(
                "The checker changed after authorization. Authorize the current "
                "checker version, then retry."
            )
        _require_runtime_unchanged(registration.provider_runtime)
        return registration

    def require_compatible(
        self,
        checker_id: str,
        *,
        evidence_kind: EvidenceKind | str,
        format_id: str,
        format_version: str,
        claim_schema_uri: str,
        semantics_uri: str,
        candidate_schema_uri: str,
        target_schema_uri: str | None = None,
        target_semantics_uri: str | None = None,
    ) -> CheckerRegistration:
        """Require an active checker matching every declared evidence binding."""

        registration = self.require_active(checker_id)
        expected_kind = EvidenceKind(evidence_kind)
        scope_error = _compatibility_scope_error(
            evidence_kind=registration.evidence_kind,
            claim_schema_uris=registration.claim_schema_uris,
            semantics_uris=registration.semantics_uris,
            candidate_schema_uris=registration.candidate_schema_uris,
            target_schema_uris=registration.target_schema_uris,
            target_semantics_uris=registration.target_semantics_uris,
        )
        if scope_error is not None:
            raise CheckerCompatibilityError(scope_error)
        if registration.evidence_kind is not expected_kind:
            raise CheckerCompatibilityError(
                "This checker does not support the supplied evidence type. Select a "
                "checker from the same reference contract, then retry."
            )
        if (
            registration.format_id != format_id
            or registration.format_version != format_version
        ):
            raise CheckerCompatibilityError(
                "This checker does not support the supplied evidence format. Select "
                "a compatible checker from the reference contract, then retry."
            )
        compatibility_sets = (
            (registration.claim_schema_uris, claim_schema_uri, "claim schema"),
            (registration.semantics_uris, semantics_uri, "semantics"),
            (
                registration.candidate_schema_uris,
                candidate_schema_uri,
                "candidate schema",
            ),
        )
        for supported, actual, label in compatibility_sets:
            if supported and actual not in supported:
                raise CheckerCompatibilityError(
                    f"This checker does not support the requested {label}. Select a "
                    "compatible checker from the reference contract, then retry."
                )
        target_compatibility = (
            (
                registration.target_schema_uris,
                target_schema_uri,
                "target schema",
            ),
            (
                registration.target_semantics_uris,
                target_semantics_uri,
                "target semantics",
            ),
        )
        for target_supported, target_actual, target_label in target_compatibility:
            if target_supported and (
                target_actual is None or target_actual not in target_supported
            ):
                raise CheckerCompatibilityError(
                    f"This checker does not support the requested {target_label}. "
                    "Select a compatible transformation checker, then retry."
                )
        return registration

    def select_compatible(
        self,
        *,
        evidence_kind: EvidenceKind | str,
        format_id: str,
        format_version: str,
        claim_schema_uri: str,
        semantics_uri: str,
        candidate_schema_uri: str,
        target_schema_uri: str | None = None,
        target_semantics_uri: str | None = None,
    ) -> CheckerRegistration:
        """Select the unique active checker compatible with an evidence format."""

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT checker_id FROM checkers WHERE authorized = 1"
            ).fetchall()
        compatible: list[CheckerRegistration] = []
        for row in rows:
            try:
                compatible.append(
                    self.require_compatible(
                        row["checker_id"],
                        evidence_kind=evidence_kind,
                        format_id=format_id,
                        format_version=format_version,
                        claim_schema_uri=claim_schema_uri,
                        semantics_uri=semantics_uri,
                        candidate_schema_uri=candidate_schema_uri,
                        target_schema_uri=target_schema_uri,
                        target_semantics_uri=target_semantics_uri,
                    )
                )
            except CheckerRegistryError:
                continue
        if not compatible:
            raise CheckerNotFoundError(
                "No authorized checker matches this evidence and its semantics. "
                "Review the reference contract and authorize a compatible checker "
                "before retrying."
            )
        if len(compatible) > 1:
            raise CheckerCompatibilityError(
                "Multiple authorized checkers match this evidence. Configure checker "
                "policy to select exactly one, then retry."
            )
        return compatible[0]

    def revoke(self, checker_id: str, *, reason: str) -> None:
        """Block new verification while preserving historical records."""

        with self._policy_write_lock(), self._connection() as connection:
            row = connection.execute(
                "SELECT authorized FROM checkers WHERE checker_id = ?",
                (checker_id,),
            ).fetchone()
            if row is None:
                raise CheckerNotFoundError(
                    "The checker is not registered. Review the reference contract, "
                    "choose an authorized checker_id, and retry."
                )
            if not bool(row["authorized"]):
                raise CheckerRevokedError(
                    "This checker is already revoked. No registry change is needed."
                )
            connection.execute(
                "UPDATE checkers SET authorized = 0 WHERE checker_id = ?",
                (checker_id,),
            )
            connection.execute(
                """
                INSERT INTO checker_audit (checker_id, action, reason)
                VALUES (?, 'REVOKED', ?)
                """,
                (checker_id, reason),
            )

    def audit_log(self, checker_id: str) -> tuple[CheckerAuditEvent, ...]:
        """Return ordered authorization and revocation events."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, checker_id, action, reason, recorded_at
                FROM checker_audit
                WHERE checker_id = ?
                ORDER BY sequence
                """,
                (checker_id,),
            ).fetchall()
        return tuple(CheckerAuditEvent.model_validate(dict(row)) for row in rows)


def _checker_identifier(registration: CheckerRegistration) -> str:
    identity_payload = registration.model_dump(mode="json", exclude_none=True)
    del identity_payload["checker_id"]
    del identity_payload["authorized"]
    identifier = hashlib.sha256(
        b"jacobian.checker.v1\x00" + canonicalize_json(identity_payload)
    ).hexdigest()
    return f"checker://sha256/{identifier}"


def _compatibility_scope_error(
    *,
    evidence_kind: EvidenceKind,
    claim_schema_uris: tuple[str, ...],
    semantics_uris: tuple[str, ...],
    candidate_schema_uris: tuple[str, ...],
    target_schema_uris: tuple[str, ...],
    target_semantics_uris: tuple[str, ...],
) -> str | None:
    if not claim_schema_uris or not semantics_uris or not candidate_schema_uris:
        return (
            "Checker authorization requires claim schema, semantics, and candidate "
            "schema allowlists. Supply all three, then retry."
        )
    if bool(target_schema_uris) != bool(target_semantics_uris):
        return (
            "Target schema and target semantics allowlists must be supplied together. "
            "Supply both or omit both, then retry."
        )
    return None
