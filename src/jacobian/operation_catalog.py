"""Revisioned, read-optimized catalog for built-in mathematical operations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationInputKind,
)
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operation_discovery import (
    discovery_relevance,
    input_acceptance,
    matches_domain,
    normalize_domain,
)
from jacobian.operation_errors import OperationDiscoveryCursorError
from jacobian.schema_registry import model_schema

if TYPE_CHECKING:
    from jacobian.checker_operations import AuthorizedChecker


class OperationCatalogError(RuntimeError):
    """Persisted catalog state is missing, malformed, or inconsistent."""


class VisibilityPolicy(Protocol):
    @property
    def profile(self) -> str: ...

    @property
    def digest(self) -> str: ...

    def project(
        self, descriptor: OperationDescriptor
    ) -> OperationDescriptor | None: ...

    def allows(self, operation_id: str, tags: tuple[str, ...]) -> bool: ...


@dataclass(frozen=True, slots=True)
class CatalogHeader:
    revision: int
    package_version: str
    format_version: int
    checker_binding_digest: str
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class OperationSearchCard:
    operation_id: str
    title: str
    description: str
    tags: tuple[str, ...]
    accepted_input_kinds: tuple[OperationInputKind, ...]
    accepted_artifact_types: tuple[str, ...]

    @classmethod
    def from_descriptor(cls, descriptor: OperationDescriptor) -> OperationSearchCard:
        return cls(
            operation_id=descriptor.operation_id,
            title=descriptor.title,
            description=descriptor.description,
            tags=descriptor.tags,
            accepted_input_kinds=descriptor.accepted_input_kinds,
            accepted_artifact_types=descriptor.accepted_artifact_types,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "accepted_input_kinds": [kind.value for kind in self.accepted_input_kinds],
            "accepted_artifact_types": list(self.accepted_artifact_types),
        }


@dataclass(frozen=True, slots=True)
class CompiledCatalogEntry:
    descriptor: OperationDescriptor
    declaration_module: str
    declaration_digest: str


@dataclass(frozen=True, slots=True)
class OperationDeclarationRecord:
    """Persisted locator and digest for one exact operation declaration."""

    operation_id: str
    module: str
    declaration_digest: str


@dataclass(frozen=True, slots=True)
class OperationCheckerBinding:
    """Trusted checker identity selected with one immutable catalog revision."""

    operation_id: str
    binding_index: int
    checker_id: str
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class CatalogBuildResult:
    revision: int
    operation_count: int
    omitted_operations: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]


def public_operation_descriptor(
    descriptor: OperationDescriptor,
) -> OperationDescriptor:
    """Remove execution-only identity from catalog and wire descriptors."""

    return descriptor.model_copy(
        update={"provider": "built-in", "provider_runtime": None},
        deep=True,
    )


class OperationCatalogStore:
    """Own catalog writes performed only by the operator lifecycle."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def commit(
        self,
        *,
        package_version: str,
        checker_binding_digest: str,
        entries: tuple[CompiledCatalogEntry, ...],
        checker_bindings: dict[str, tuple[tuple[str, str], ...]],
        diagnostics: tuple[dict[str, Any], ...] = (),
        omitted_operations: tuple[str, ...] = (),
    ) -> CatalogBuildResult:
        operation_ids = tuple(entry.descriptor.operation_id for entry in entries)
        if operation_ids != tuple(sorted(set(operation_ids))):
            raise OperationCatalogError(
                "compiled catalog entries must be unique and sorted"
            )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO operation_catalog_snapshots(
                    package_version, format_version, checker_binding_digest,
                    diagnostics_json
                ) VALUES (?, 1, ?, ?)
                """,
                (
                    package_version,
                    checker_binding_digest,
                    canonicalize_json(list(diagnostics)),
                ),
            )
            if cursor.lastrowid is None:
                raise OperationCatalogError(
                    "catalog snapshot revision was not allocated"
                )
            revision = cursor.lastrowid
            for entry in entries:
                descriptor = entry.descriptor
                card = OperationSearchCard.from_descriptor(descriptor)
                connection.execute(
                    """
                    INSERT INTO operation_catalog_entries(
                        snapshot_revision, operation_id, search_card_json,
                        descriptor_json, input_schema_json, output_schema_json,
                        declaration_module, declaration_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision,
                        descriptor.operation_id,
                        canonicalize_json(card.as_json()),
                        canonicalize_json(descriptor.model_dump(mode="json")),
                        canonicalize_json(descriptor.input_schema),
                        canonicalize_json(descriptor.output_schema),
                        entry.declaration_module,
                        entry.declaration_digest,
                    ),
                )
            for operation_id, operation_bindings in sorted(checker_bindings.items()):
                for binding_index, (checker_id, manifest_digest) in enumerate(
                    operation_bindings
                ):
                    connection.execute(
                        """
                        INSERT INTO operation_checker_bindings(
                            snapshot_revision, operation_id, binding_index,
                            checker_id, manifest_digest
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            revision,
                            operation_id,
                            binding_index,
                            checker_id,
                            manifest_digest,
                        ),
                    )
            connection.execute(
                """
                INSERT INTO active_operation_catalog(id, snapshot_revision)
                VALUES (0, ?)
                ON CONFLICT(id) DO UPDATE
                SET snapshot_revision = excluded.snapshot_revision
                """,
                (revision,),
            )
            connection.commit()
        return CatalogBuildResult(
            revision=revision,
            operation_count=len(entries),
            omitted_operations=omitted_operations,
            diagnostics=diagnostics,
        )


class OperationCatalog:
    """Immutable active header/cards with indexed descriptor inspection."""

    def __init__(
        self,
        database_path: Path,
        policy: VisibilityPolicy,
        *,
        expected_package_version: str,
    ) -> None:
        self.database_path = database_path
        self.policy = policy
        self.header, self._cards, self._checker_bindings = self._load_active(
            expected_package_version
        )

    def _connect_read_only(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def declaration_record(
        self, operation_id: str
    ) -> OperationDeclarationRecord | None:
        """Read one declaration locator without materializing the catalog."""

        try:
            with self._connect_read_only() as connection:
                row = connection.execute(
                    """
                    SELECT operation_id, declaration_module, declaration_digest
                    FROM operation_catalog_entries
                    WHERE snapshot_revision = ? AND operation_id = ?
                    """,
                    (self.header.revision, operation_id),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise OperationCatalogError(
                "active operation declaration locator is unreadable"
            ) from exc
        if row is None:
            return None
        return OperationDeclarationRecord(
            operation_id=str(row["operation_id"]),
            module=str(row["declaration_module"]),
            declaration_digest=str(row["declaration_digest"]),
        )

    def checker_binding(self, operation_id: str) -> OperationCheckerBinding | None:
        """Return the persisted checker authority for one operation, if any."""

        bindings = self._checker_bindings.get(operation_id, ())
        return bindings[0] if bindings else None

    def checker_bindings(
        self, operation_id: str
    ) -> tuple[OperationCheckerBinding, ...]:
        """Return every ordered checker authority for one operation."""

        return self._checker_bindings.get(operation_id, ())

    def _load_active(
        self, expected_package_version: str
    ) -> tuple[
        CatalogHeader,
        tuple[OperationSearchCard, ...],
        dict[str, tuple[OperationCheckerBinding, ...]],
    ]:
        if not self.database_path.exists():
            raise OperationCatalogError(
                "STATE_INITIALIZATION_REQUIRED: run `jacobian init`"
            )
        if not self.database_path.is_file():
            raise OperationCatalogError(
                "STATE_UPDATE_REQUIRED: catalog state is unreadable; "
                "run `jacobian update`"
            )
        try:
            with self._connect_read_only() as connection:
                row = connection.execute(
                    """
                    SELECT s.*
                    FROM active_operation_catalog AS a
                    JOIN operation_catalog_snapshots AS s
                      ON s.revision = a.snapshot_revision
                    WHERE a.id = 0
                    """
                ).fetchone()
                if row is None:
                    raise OperationCatalogError(
                        "STATE_INITIALIZATION_REQUIRED: run `jacobian init`"
                    )
                if str(row["package_version"]) != expected_package_version:
                    raise OperationCatalogError(
                        "STATE_UPDATE_REQUIRED: run `jacobian update`"
                    )
                cards = tuple(
                    _decode_card(item["search_card_json"])
                    for item in connection.execute(
                        """
                        SELECT search_card_json
                        FROM operation_catalog_entries
                        WHERE snapshot_revision = ?
                        ORDER BY operation_id
                        """,
                        (int(row["revision"]),),
                    )
                )
                binding_rows = connection.execute(
                    """
                        SELECT operation_id, binding_index, checker_id,
                               manifest_digest
                        FROM operation_checker_bindings
                        WHERE snapshot_revision = ?
                        ORDER BY operation_id, binding_index
                        """,
                    (int(row["revision"]),),
                )
                grouped_bindings: dict[str, list[OperationCheckerBinding]] = {}
                for item in binding_rows:
                    operation_id = str(item["operation_id"])
                    grouped_bindings.setdefault(operation_id, []).append(
                        OperationCheckerBinding(
                            operation_id=operation_id,
                            binding_index=int(item["binding_index"]),
                            checker_id=str(item["checker_id"]),
                            manifest_digest=str(item["manifest_digest"]),
                        )
                    )
                checker_bindings = {
                    operation_id: tuple(bindings)
                    for operation_id, bindings in grouped_bindings.items()
                }
        except sqlite3.DatabaseError as exc:
            raise OperationCatalogError(
                "STATE_UPDATE_REQUIRED: catalog state is corrupt; run `jacobian update`"
            ) from exc
        return (
            CatalogHeader(
                revision=int(row["revision"]),
                package_version=str(row["package_version"]),
                format_version=int(row["format_version"]),
                checker_binding_digest=str(row["checker_binding_digest"]),
                diagnostics=tuple(
                    cast(
                        list[dict[str, Any]], loads_strict_json(row["diagnostics_json"])
                    )
                ),
            ),
            cards,
            checker_bindings,
        )

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        with self._connect_read_only() as connection:
            row = connection.execute(
                """
                SELECT descriptor_json
                FROM operation_catalog_entries
                WHERE snapshot_revision = ? AND operation_id = ?
                """,
                (self.header.revision, operation_id),
            ).fetchone()
        if row is None:
            return None
        descriptor = OperationDescriptor.model_validate(
            loads_strict_json(row["descriptor_json"])
        )
        return self.policy.project(descriptor)

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        normalized_domain = (
            normalize_domain(request.domain) if request.domain is not None else None
        )
        ranked: list[tuple[int, OperationDiscoveryMatch]] = []
        for card in self._cards:
            if not self.policy.allows(card.operation_id, card.tags):
                continue
            if normalized_domain is not None and not matches_domain(
                card, normalized_domain
            ):
                continue
            applicability, code = input_acceptance(
                card, request.input_kind, request.artifact_type
            )
            score = discovery_relevance(card, request.query)
            if score:
                ranked.append(
                    (
                        score,
                        OperationDiscoveryMatch(
                            operation_id=card.operation_id,
                            title=card.title,
                            description=card.description,
                            tags=card.tags,
                            relevance_score=score,
                            applicability=applicability,
                            applicability_code=code,
                        ),
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1].operation_id))
        start = _cursor_start(ranked, request.cursor)
        page = ranked[start : start + request.limit]
        next_cursor = (
            page[-1][1].operation_id
            if page and start + len(page) < len(ranked)
            else None
        )
        return OperationDiscoveryResult(
            query=request.query,
            domain=normalized_domain,
            input_kind=request.input_kind,
            artifact_type=request.artifact_type,
            matches=tuple(match for _, match in page),
            total_matches=len(ranked),
            truncated=next_cursor is not None,
            next_cursor=next_cursor,
        )

    def snapshot(self) -> OperationCatalogSnapshot:
        descriptors = tuple(
            descriptor
            for card in self._cards
            if (descriptor := self.inspect(card.operation_id)) is not None
        )
        return OperationCatalogSnapshot(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            operations=descriptors,
        )


def declaration_digest(value: dict[str, Any]) -> str:
    return "sha256:" + sha256(canonicalize_json(value)).hexdigest()


def operation_declaration_digest(
    declaration: OperationDeclaration[Any, Any],
) -> str:
    """Digest the stable typed identity loaded again by selected execution."""

    return declaration_digest(
        {
            "operation_id": declaration.operation_id,
            "version": declaration.version,
            "input_schema": model_schema(declaration.request_type),
            "result_schema": model_schema(declaration.result_type),
        }
    )


def exact_checker_declaration_digest(
    declaration: AuthorizedChecker,
    descriptor: OperationDescriptor,
) -> str:
    """Bind one generated verifier descriptor to its pure checker declaration."""

    return declaration_digest(
        {
            "operation_id": descriptor.operation_id,
            "producer_operation_id": declaration.operation_id,
            "version": descriptor.version,
            "input_schema": descriptor.input_schema,
            "output_schema": descriptor.output_schema,
        }
    )


def _decode_card(value: bytes | str) -> OperationSearchCard:
    decoded = cast(dict[str, Any], loads_strict_json(value))
    return OperationSearchCard(
        operation_id=str(decoded["operation_id"]),
        title=str(decoded["title"]),
        description=str(decoded["description"]),
        tags=tuple(str(tag) for tag in cast(list[Any], decoded["tags"])),
        accepted_input_kinds=tuple(
            OperationInputKind(str(kind))
            for kind in cast(list[Any], decoded["accepted_input_kinds"])
        ),
        accepted_artifact_types=tuple(
            str(artifact_type)
            for artifact_type in cast(list[Any], decoded["accepted_artifact_types"])
        ),
    )


def _cursor_start(
    ranked: list[tuple[int, OperationDiscoveryMatch]], cursor: str | None
) -> int:
    if cursor is None:
        return 0
    for index, (_, match) in enumerate(ranked):
        if match.operation_id == cursor:
            return index + 1
    raise OperationDiscoveryCursorError(
        "cursor is not present in the filtered discovery result"
    )


__all__ = [
    "CatalogBuildResult",
    "CatalogHeader",
    "CompiledCatalogEntry",
    "OperationCatalog",
    "OperationCatalogError",
    "OperationCatalogStore",
    "OperationCheckerBinding",
    "OperationDeclarationRecord",
    "OperationSearchCard",
    "declaration_digest",
    "exact_checker_declaration_digest",
    "operation_declaration_digest",
    "public_operation_descriptor",
]
