"""Deterministic operation discovery and routing projection."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from math import log
from typing import Protocol

from jacobian.catalog.models import (
    OperationBrowseCard,
    OperationBrowseResult,
    OperationDiscoveryMatch,
    OperationMatchRequest,
    OperationMatchResult,
)


class OperationDiscoveryCursorError(ValueError):
    """A continuation cursor does not belong to the filtered result."""


_DISCOVERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DISCOVERY_STOP_WORDS = frozenset(
    {"a", "an", "and", "for", "find", "from", "in", "of", "on", "the", "to", "with"}
)
# Terms whose final ``s`` is not regular plural morphology, plus irregular
# plurals that the deliberately small suffix rules below would corrupt.
_DISCOVERY_INFLECTION_EXCEPTIONS = frozenset(
    {
        "alias",
        "always",
        "atlas",
        "axes",
        "bases",
        "bias",
        "chaos",
        "does",
        "dynamics",
        "farkas",
        "guigues",
        "indices",
        "lens",
        "lies",
        "macwilliams",
        "matrices",
        "news",
        "series",
        "sims",
        "simplices",
        "species",
        "vertices",
    }
)
# Protected singulars whose ordinary ``-es`` plurals need an explicit map.
_DISCOVERY_PLURAL_INFLECTIONS = {
    "aliases": "alias",
    "atlases": "atlas",
    "biases": "bias",
    "lenses": "lens",
}
_DISCOVERY_SINGULAR_SUFFIXES = ("ics", "is", "ous", "ss", "us")


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


def match_operations(
    operations: Sequence[SearchableOperation],
    request: OperationMatchRequest,
) -> OperationMatchResult:
    """Match a local mathematical need against immutable operation declarations."""

    normalized_namespace = (
        normalize_namespace(request.namespace)
        if request.namespace is not None
        else None
    )
    eligible = tuple(
        descriptor
        for descriptor in operations
        if normalized_namespace is None
        or matches_namespace(descriptor, normalized_namespace)
    )
    operation_fields = tuple(_operation_field_terms(item) for item in eligible)
    document_terms = tuple(frozenset().union(*fields) for fields in operation_fields)
    document_frequency = Counter(term for terms in document_terms for term in terms)
    need_terms = discovery_terms(request.need)
    ranked: list[tuple[float, OperationDiscoveryMatch]] = []
    for descriptor, fields in zip(eligible, operation_fields, strict=True):
        score = need_relevance(
            fields,
            need_terms,
            document_frequency=document_frequency,
            document_count=len(document_terms),
        )
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
    return OperationMatchResult(
        need=request.need,
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


def normalize_discovery_term(term: str) -> str:
    """Return one conservative comparison form for a lexical search token."""

    if term in _DISCOVERY_PLURAL_INFLECTIONS:
        return _DISCOVERY_PLURAL_INFLECTIONS[term]
    if len(term) <= 3 or term in _DISCOVERY_INFLECTION_EXCEPTIONS:
        return term
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if term.endswith(("ches", "shes", "sses", "xes")):
        return term[:-2]
    if term.endswith("s") and not term.endswith(_DISCOVERY_SINGULAR_SUFFIXES):
        return term[:-1]
    return term


def discovery_terms(query: str) -> frozenset[str]:
    return frozenset(
        normalized_term
        for term in _DISCOVERY_TOKEN_PATTERN.findall(query.casefold())
        if (normalized_term := normalize_discovery_term(term))
        not in _DISCOVERY_STOP_WORDS
    )


def token_set(value: str) -> frozenset[str]:
    return frozenset(
        normalize_discovery_term(term)
        for term in _DISCOVERY_TOKEN_PATTERN.findall(value.casefold())
    )


def _operation_field_terms(
    operation: SearchableOperation,
) -> tuple[frozenset[str], ...]:
    """Return normalized identifier, title, prose, tag, and alias fields."""

    return (
        token_set(operation.operation_id),
        token_set(operation.title),
        token_set(operation.description),
        frozenset(term for tag in operation.tags for term in token_set(tag)),
        frozenset(
            term
            for discovery_term in operation.discovery_terms
            for term in token_set(discovery_term)
        ),
    )


def need_relevance(
    document_fields: tuple[frozenset[str], ...],
    need_terms: frozenset[str],
    *,
    document_frequency: Counter[str],
    document_count: int,
) -> float:
    """Score one operation with a deterministic field-weighted BM25-style formula."""

    if not need_terms or not document_fields or document_count == 0:
        return 0
    k1 = 1.2
    field_weights = (2, 3, 1, 2, 4)
    return sum(
        log(
            1.0
            + (document_count - document_frequency[term] + 0.5)
            / (document_frequency[term] + 0.5)
        )
        * (weighted_frequency * (k1 + 1.0))
        / (weighted_frequency + k1)
        for term in need_terms
        if (
            weighted_frequency := sum(
                weight
                for weight, field_terms in zip(
                    field_weights, document_fields, strict=True
                )
                if term in field_terms
            )
        )
    )


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
    "match_operations",
    "matches_namespace",
    "need_relevance",
    "normalize_namespace",
    "operation_namespace",
]
