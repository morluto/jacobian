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
_INFLECTION_EXCLUSIONS = frozenset(
    {"basis", "bases", "class", "classes", "grass", "series"}
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

    @property
    def discovery_terms(self) -> tuple[str, ...]: ...


def discover_operations(
    operations: Sequence[SearchableOperation],
    request: OperationDiscoveryRequest,
) -> OperationDiscoveryResult:
    """Search immutable operation declarations deterministically."""

    normalized_namespace = (
        normalize_namespace(request.namespace)
        if request.namespace is not None
        else None
    )
    ranked: list[tuple[int, OperationDiscoveryMatch]] = []
    for descriptor in operations:
        if normalized_namespace is not None and not matches_namespace(
            descriptor, normalized_namespace
        ):
            continue
        score = discovery_relevance(descriptor, request.query)
        match = OperationDiscoveryMatch(
            operation_id=descriptor.operation_id,
            title=descriptor.title,
            description=descriptor.description,
            tags=descriptor.tags,
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
        namespace=normalized_namespace,
        matches=tuple(match for _, match in page),
        total_matches=total_matches,
        next_cursor=next_cursor,
    )


def browse_operations(
    searchable_operations: Sequence[SearchableOperation],
    *,
    namespace: str | None,
    limit: int,
    cursor: str | None,
) -> OperationBrowseResult:
    """Page a filtered immutable snapshot in operation-ID order without ranking."""

    normalized_namespace = (
        normalize_namespace(namespace) if namespace is not None else None
    )
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
        if normalized_namespace is None
        or matches_namespace(descriptor, normalized_namespace)
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
        namespace=normalized_namespace,
        operations=page,
        total_operations=len(operations),
        next_cursor=next_cursor,
    )


def normalize_discovery_text(value: str) -> str:
    return "-".join(_normalized_tokens(value))


def normalized_discovery_terms(value: str) -> frozenset[str]:
    return frozenset(
        term for term in _normalized_tokens(value) if term not in _DISCOVERY_STOP_WORDS
    )


def token_set(value: str) -> frozenset[str]:
    """Return the legacy exact lexical terms of one declaration field."""

    return frozenset(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        _normalize_inflection(term)
        for term in _DISCOVERY_TOKEN_PATTERN.findall(value.casefold())
    )


def _normalize_inflection(term: str) -> str:
    """Normalize only audited ordinary plural forms for lexical comparison."""

    if term in _INFLECTION_EXCLUSIONS or len(term) < 4:
        return term
    if term.endswith("ies") and term != "series":
        return term[:-3] + "y"
    if term.endswith("s") and not term.endswith(("ss", "is")):
        return term[:-1]
    return term


def discovery_relevance(
    operation: SearchableOperation,
    query: str,
) -> int:
    query_terms = frozenset(
        term for term in token_set(query) if term not in _DISCOVERY_STOP_WORDS
    )
    normalized_query_terms = normalized_discovery_terms(query)
    if not normalized_query_terms:
        return 0
    identifier_terms = token_set(operation.operation_id)
    tag_terms = frozenset(term for tag in operation.tags for term in token_set(tag))
    declared_discovery_terms = frozenset(
        term
        for discovery_term in operation.discovery_terms
        for term in token_set(discovery_term)
    )
    title_terms = token_set(operation.title)
    description_terms = token_set(operation.description)
    score = 0
    normalized_only = normalized_query_terms - query_terms
    for terms, weight in (
        (identifier_terms, 12),
        (tag_terms, 10),
        (declared_discovery_terms, 10),
        (title_terms, 8),
        (description_terms, 3),
    ):
        overlap = query_terms & terms
        if overlap:
            score += weight * len(overlap)
        normalized_overlap = normalized_only & frozenset(
            _normalized_tokens(" ".join(terms))
        )
        if normalized_overlap:
            score += max(1, weight // 2) * len(normalized_overlap)
    normalized_query = normalize_discovery_text(query)
    normalized_text = normalize_discovery_text(
        f"{operation.operation_id} {operation.title} {operation.description}"
    )
    if normalized_query and f"-{normalized_query}-" in f"-{normalized_text}-":
        score += 20
    normalized_declared_terms = tuple(
        normalize_discovery_text(discovery_term)
        for discovery_term in operation.discovery_terms
    )
    matching_term_lengths = (
        len(token_set(discovery_term))
        for discovery_term in normalized_declared_terms
        if discovery_term and f"-{discovery_term}-" in f"-{normalized_query}-"
    )
    score += 12 * max(matching_term_lengths, default=0)
    return score


def normalize_namespace(value: str) -> str:
    return "_".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def operation_namespace(operation: SearchableOperation) -> str:
    return operation.operation_id.partition(".")[0]


def matches_namespace(
    operation: SearchableOperation, normalized_namespace: str
) -> bool:
    """Match one explicit primary operation-ID namespace, never a tag."""

    return normalized_namespace == normalize_namespace(operation_namespace(operation))


__all__ = [
    "browse_operations",
    "discover_operations",
    "discovery_relevance",
    "matches_namespace",
    "normalize_namespace",
    "operation_namespace",
]
