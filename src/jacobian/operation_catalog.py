"""Revisioned, read-optimized catalog for built-in mathematical operations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
)
from jacobian.operation_discovery import (
    discovery_applicability,
    discovery_relevance,
    matches_domain,
    normalize_domain,
)
from jacobian.operation_errors import OperationDiscoveryCursorError


class OperationCatalogError(RuntimeError):
    """Persisted catalog state is missing, malformed, or inconsistent."""


class VisibilityPolicy(Protocol):
    profile: str
    digest: str

    def project(self, descriptor: OperationDescriptor) -> OperationDescriptor | None: ...


@dataclass(frozen=True, slots=True)
class CatalogHeader:
    revision: int
    package_version: str
    format_version: int
    provider_inventory_digest: str
    checker_binding_digest: str
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class OperationSearchCard:
    operation_id: str
    title: str
    description: str
    tags: tuple[str, ...]

    @classmethod
    def from_descriptor(cls, descriptor: OperationDescriptor) -> OperationSearchCard:
        return cls(
            operation_id=descriptor.operation_id,
            title=descriptor.title,
            description=descriptor.description,
            tags=descriptor.tags,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class CompiledCatalogEntry:
    descriptor: OperationDescriptor
    bundle_module: str
    declaration_digest: str


@dataclass(frozen=True, slots=True)
class CatalogBuildResult:
    revision: int
    operation_count: int
    omitted_operations: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]


class OperationCatalogStore:
    """Own catalog writes performed only by the operator lifecycle."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def commit(
        self,
        *,
        package_version: str,
        provider_inventory_digest: str,
        checker_binding_digest: str,
        entries: tuple[CompiledCatalogEntry, ...],
        checker_bindings: dict[str, tuple[str, str]],
        diagnostics: tuple[dict[str, Any], ...] = (),
        omitted_operations: tuple[str, ...] = (),
    ) -> CatalogBuildResult:
        operation_ids = tuple(entry.descriptor.operation_id for entry in entries)
        if operation_ids != tuple(sorted(set(operation_ids))):
            raise OperationCatalogError("compiled catalog entries must be unique and sorted")
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO operation_catalog_snapshots(
                    package_version, format_version, provider_inventory_digest,
                    checker_binding_digest, diagnostics_json
                ) VALUES (?, 1, ?, ?, ?)
                """,
                (
                    package_version,
                    provider_inventory_digest,
                    checker_binding_digest,
                    canonicalize_json(list(diagnostics)),
                ),
            )
            if cursor.lastrowid is None:
                raise OperationCatalogError("catalog snapshot revision was not allocated")
            revision = cursor.lastrowid
            for entry in entries:
                descriptor = entry.descriptor
                card = OperationSearchCard.from_descriptor(descriptor)
                connection.execute(
                    """
                    INSERT INTO operation_catalog_entries(
                        snapshot_revision, operation_id, search_card_json,
                        descriptor_json, input_schema_json, output_schema_json,
                        bundle_module, declaration_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision,
                        descriptor.operation_id,
                        canonicalize_json(card.as_json()),
                        canonicalize_json(descriptor.model_dump(mode="json")),
                        canonicalize_json(descriptor.input_schema),
                        canonicalize_json(descriptor.output_schema),
                        entry.bundle_module,
                        entry.declaration_digest,
                    ),
                )
            for operation_id, (checker_id, manifest_digest) in sorted(
                checker_bindings.items()
            ):
                connection.execute(
                    """
                    INSERT INTO operation_checker_bindings(
                        snapshot_revision, operation_id, checker_id, manifest_digest
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (revision, operation_id, checker_id, manifest_digest),
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
        self.header, self._cards = self._load_active(expected_package_version)

    def _connect_read_only(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _load_active(
        self, expected_package_version: str
    ) -> tuple[CatalogHeader, tuple[OperationSearchCard, ...]]:
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
                    raise OperationCatalogError("STATE_INITIALIZATION_REQUIRED: run `jacobian init`")
                if str(row["package_version"]) != expected_package_version:
                    raise OperationCatalogError("STATE_UPDATE_REQUIRED: run `jacobian update`")
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
        except sqlite3.DatabaseError as exc:
            raise OperationCatalogError(
                "STATE_UPDATE_REQUIRED: catalog state is corrupt; run `jacobian update`"
            ) from exc
        return (
            CatalogHeader(
                revision=int(row["revision"]),
                package_version=str(row["package_version"]),
                format_version=int(row["format_version"]),
                provider_inventory_digest=str(row["provider_inventory_digest"]),
                checker_binding_digest=str(row["checker_binding_digest"]),
                diagnostics=tuple(
                    cast(list[dict[str, Any]], loads_strict_json(row["diagnostics_json"]))
                ),
            ),
            cards,
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
            descriptor = _card_descriptor(card)
            if self.policy.project(descriptor) is None:
                continue
            if normalized_domain is not None and not matches_domain(
                descriptor, normalized_domain
            ):
                continue
            applicability, code = discovery_applicability(
                descriptor, request.input_kind, request.artifact_type
            )
            score = discovery_relevance(descriptor, request.query)
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


def _decode_card(value: bytes | str) -> OperationSearchCard:
    decoded = cast(dict[str, Any], loads_strict_json(value))
    return OperationSearchCard(
        operation_id=str(decoded["operation_id"]),
        title=str(decoded["title"]),
        description=str(decoded["description"]),
        tags=tuple(str(tag) for tag in cast(list[Any], decoded["tags"])),
    )


def _card_descriptor(card: OperationSearchCard) -> OperationDescriptor:
    """Adapt searchable fields to the existing deterministic ranker only."""

    return OperationDescriptor(
        operation_id=card.operation_id,
        version="catalog-card",
        title=card.title,
        description=card.description,
        provider="built-in",
        input_schema={},
        output_schema={},
        tags=card.tags,
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
    "OperationSearchCard",
    "declaration_digest",
]
