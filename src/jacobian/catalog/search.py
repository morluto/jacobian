"""Deterministic operation discovery and routing projection."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from jacobian.catalog.models import (
    OperationBrowseCard,
    OperationBrowseResult,
    OperationDiscoveryMatch,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
)


class OperationDiscoveryCursorError(ValueError):
    """A continuation cursor does not belong to the filtered result."""


_DISCOVERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DISCOVERY_STOP_WORDS = frozenset(
    {"a", "an", "and", "for", "find", "from", "in", "of", "on", "the", "to", "with"}
)


class SearchableOperation(Protocol):
    @property
    def operation_id(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def tags(self) -> tuple[str, ...]: ...


def discover_operations(
    operations: Sequence[SearchableOperation],
    request: OperationDiscoveryRequest,
) -> OperationDiscoveryResult:
    """Search immutable operation declarations deterministically."""

    normalized_domain = (
        normalize_domain(request.domain) if request.domain is not None else None
    )
    ranked: list[tuple[int, OperationDiscoveryMatch]] = []
    for descriptor in operations:
        if normalized_domain is not None and not matches_domain(
            descriptor, normalized_domain
        ):
            continue
        score = discovery_relevance(descriptor, request.query)
        match = OperationDiscoveryMatch(
            operation_id=descriptor.operation_id,
            title=descriptor.title,
            description=descriptor.description,
            tags=descriptor.tags,
            relevance_score=score,
            applicability="NEEDS_MORE_TYPED_REQUIREMENTS",
            applicability_code="FULL_REQUEST_REQUIRED",
        )
        if score > 0:
            ranked.append((score, match))
    ranked.sort(key=lambda item: (-item[0], item[1].operation_id))
    total_matches = len(ranked)
    start = 0
    if request.cursor is not None:
        try:
            start = (
                next(
                    index
                    for index, (_, match) in enumerate(ranked)
                    if match.operation_id == request.cursor
                )
                + 1
            )
        except StopIteration:
            raise OperationDiscoveryCursorError(
                "cursor is not present in the filtered discovery result"
            ) from None
    page = ranked[start : start + request.limit]
    next_cursor = (
        page[-1][1].operation_id if page and start + len(page) < total_matches else None
    )
    return OperationDiscoveryResult(
        query=request.query,
        domain=normalized_domain,
        matches=tuple(match for _, match in page),
        total_matches=total_matches,
        truncated=next_cursor is not None,
        next_cursor=next_cursor,
    )


def browse_operations(
    searchable_operations: Sequence[SearchableOperation],
    *,
    domain: str | None,
    limit: int,
    cursor: str | None,
) -> OperationBrowseResult:
    """Page a filtered immutable snapshot in operation-ID order without ranking."""

    normalized_domain = normalize_domain(domain) if domain is not None else None
    operations = tuple(
        OperationBrowseCard(
            operation_id=descriptor.operation_id,
            title=descriptor.title,
            description=descriptor.description,
            tags=descriptor.tags,
        )
        for descriptor in sorted(
            searchable_operations, key=lambda operation: operation.operation_id
        )
        if normalized_domain is None or matches_domain(descriptor, normalized_domain)
    )
    start = 0
    if cursor is not None:
        try:
            start = (
                next(
                    index
                    for index, operation in enumerate(operations)
                    if operation.operation_id == cursor
                )
                + 1
            )
        except StopIteration:
            raise OperationDiscoveryCursorError(
                "cursor is not present in the filtered operation result"
            ) from None
    page = operations[start : start + limit]
    next_cursor = (
        page[-1].operation_id if page and start + len(page) < len(operations) else None
    )
    return OperationBrowseResult(
        domain=normalized_domain,
        operations=page,
        total_operations=len(operations),
        truncated=next_cursor is not None,
        next_cursor=next_cursor,
    )


def normalize_discovery_text(value: str) -> str:
    return "-".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def discovery_terms(query: str) -> frozenset[str]:
    return frozenset(
        term
        for term in _DISCOVERY_TOKEN_PATTERN.findall(query.casefold())
        if term not in _DISCOVERY_STOP_WORDS
    )


def token_set(value: str) -> frozenset[str]:
    return frozenset(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def discovery_relevance(
    operation: SearchableOperation,
    query: str,
) -> int:
    query_terms = discovery_terms(query)
    if not query_terms:
        return 0
    identifier_terms = token_set(operation.operation_id)
    tag_terms = frozenset(term for tag in operation.tags for term in token_set(tag))
    title_terms = token_set(operation.title)
    description_terms = token_set(operation.description)
    score = 0
    for terms, weight in (
        (identifier_terms, 12),
        (tag_terms, 10),
        (title_terms, 8),
        (description_terms, 3),
    ):
        overlap = query_terms & terms
        if overlap:
            score += weight * len(overlap)
    normalized_query = normalize_discovery_text(query)
    normalized_text = normalize_discovery_text(
        f"{operation.operation_id} {operation.title} {operation.description}"
    )
    if normalized_query and f"-{normalized_query}-" in f"-{normalized_text}-":
        score += 20
    return score


def normalize_domain(value: str) -> str:
    return "_".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def operation_domain(operation: SearchableOperation) -> str:
    return operation.operation_id.partition(".")[0]


def matches_domain(operation: SearchableOperation, normalized_domain: str) -> bool:
    normalized_tags = {normalize_domain(tag) for tag in operation.tags}
    return (
        normalized_domain == normalize_domain(operation_domain(operation))
        or normalized_domain in normalized_tags
    )


__all__ = [
    "browse_operations",
    "discover_operations",
    "discovery_relevance",
    "matches_domain",
    "normalize_domain",
    "operation_domain",
]
