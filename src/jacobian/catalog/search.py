"""Deterministic operation discovery and routing projection."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from math import log
from typing import Protocol

from jacobian.backends import BackendName
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

    @property
    def runtime_requirements(self) -> tuple[BackendName, ...]: ...


@dataclass(frozen=True, slots=True)
class _SearchCorpus:
    entries: tuple[tuple[SearchableOperation, tuple[frozenset[str], ...]], ...]
    document_frequency: Counter[str]


class OperationSearchIndex:
    """Precomputed lexical projections for one immutable operation sequence."""

    def __init__(self, operations: Sequence[SearchableOperation]) -> None:
        entries = tuple(
            (operation, _operation_field_terms(operation)) for operation in operations
        )
        self._all = self._corpus(entries)
        namespaces = {operation_namespace(operation) for operation in operations}
        self._namespaces = {
            namespace: self._corpus(
                tuple(
                    entry
                    for entry in entries
                    if operation_namespace(entry[0]) == namespace
                )
            )
            for namespace in namespaces
        }

    @staticmethod
    def _corpus(
        entries: tuple[tuple[SearchableOperation, tuple[frozenset[str], ...]], ...],
    ) -> _SearchCorpus:
        document_terms = tuple(frozenset().union(*fields) for _, fields in entries)
        return _SearchCorpus(
            entries=entries,
            document_frequency=Counter(
                term for terms in document_terms for term in terms
            ),
        )

    def match(self, request: OperationMatchRequest) -> OperationMatchResult:
        normalized_namespace = (
            normalize_namespace(request.namespace)
            if request.namespace is not None
            else None
        )
        corpus = (
            self._all
            if normalized_namespace is None
            else self._namespaces.get(
                normalized_namespace, _SearchCorpus((), Counter())
            )
        )
        return _match_corpus(corpus, request, normalized_namespace)


def match_operations(
    operations: Sequence[SearchableOperation],
    request: OperationMatchRequest,
) -> OperationMatchResult:
    """Match a local mathematical need against immutable operation declarations."""

    return OperationSearchIndex(operations).match(request)


def _match_corpus(
    corpus: _SearchCorpus,
    request: OperationMatchRequest,
    normalized_namespace: str | None,
) -> OperationMatchResult:
    need_terms = discovery_terms(request.need)
    ranked: list[tuple[float, OperationDiscoveryMatch]] = []
    for descriptor, fields in corpus.entries:
        if not _explicit_domain_matches(request.need, need_terms, fields, descriptor):
            continue
        score = need_relevance(
            fields,
            need_terms,
            document_frequency=corpus.document_frequency,
            document_count=len(corpus.entries),
        )
        if score > 0:
            score += _phrase_relevance(descriptor, request.need)
        if score > 0:
            ranked.append(
                (
                    score,
                    OperationDiscoveryMatch(
                        operation_id=descriptor.operation_id,
                        title=descriptor.title,
                        description=descriptor.description,
                        tags=descriptor.tags,
                        runtime_requirements=descriptor.runtime_requirements,
                    ),
                )
            )
    ranked.sort(key=lambda item: (-item[0], item[1].operation_id))
    total_matches = len(ranked)
    start = 0
    if request.cursor is not None:
        try:
            start = (
                next(
                    index
                    for index, (_, match) in enumerate(ranked)
                    if _match_cursor(
                        need=request.need,
                        namespace=normalized_namespace,
                        operation_id=match.operation_id,
                    )
                    == request.cursor
                )
                + 1
            )
        except StopIteration:
            raise OperationDiscoveryCursorError(
                "cursor is not present in the filtered discovery result"
            ) from None
    page = ranked[start : start + request.limit]
    next_cursor = (
        _match_cursor(
            need=request.need,
            namespace=normalized_namespace,
            operation_id=page[-1][1].operation_id,
        )
        if page and start + len(page) < total_matches
        else None
    )
    return OperationMatchResult(
        need=request.need,
        namespace=normalized_namespace,
        matches=tuple(match for _, match in page),
        total_matches=total_matches,
        next_cursor=next_cursor,
    )


def _match_cursor(*, need: str, namespace: str | None, operation_id: str) -> str:
    """Bind one opaque stateless cursor to its ranked discovery result."""

    identity = f"{need}\0{namespace or ''}\0{operation_id}".encode()
    return f"cursor.{sha256(identity).hexdigest()}"


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
            runtime_requirements=descriptor.runtime_requirements,
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


def _phrases(value: str) -> frozenset[tuple[str, str]]:
    """Keep adjacent mathematical words together without inventing aliases."""

    words = tuple(
        normalize_discovery_term(term)
        for term in _DISCOVERY_TOKEN_PATTERN.findall(value.casefold())
    )
    return frozenset(
        (left, right)
        for left, right in pairwise(words)
        if left not in _DISCOVERY_STOP_WORDS
        and right not in _DISCOVERY_STOP_WORDS
        and not left.isdigit()
        and not right.isdigit()
    )


def _phrase_relevance(operation: SearchableOperation, need: str) -> float:
    phrases = _phrases(need)
    # A matched phrase in several fields is still one piece of evidence.
    title_phrases = _phrases(operation.title) | _phrases(operation.operation_id)
    contract_phrases = _phrases(operation.description) | frozenset(
        phrase for term in operation.discovery_terms for phrase in _phrases(term)
    )
    return len(phrases & title_phrases) + 0.5 * len(
        phrases & (contract_phrases - title_phrases)
    )


def _explicit_domain_matches(
    need: str,
    need_terms: frozenset[str],
    fields: tuple[frozenset[str], ...],
    operation: SearchableOperation,
) -> bool:
    """Respect an explicitly requested coefficient domain for checking tasks.

    Match the explicitly named domain and mathematical object, not the
    declaration's verb: a complete computation can establish a check.
    Incidental query words remain relevance signals, never hard requirements.
    Do not substitute a quotient-ring check for an integer/rational check.
    """

    domain = re.search(r"\bover\s+(?:the\s+)?(integers?|rationals?)\b", need.casefold())
    if domain is None or not need_terms & {"check", "verify", "test"}:
        return True
    terms = frozenset().union(*fields)
    if normalize_discovery_term(domain.group(1)) not in terms:
        return False
    object_phrases = frozenset(
        phrase
        for phrase in _phrases(need)
        if not set(phrase) & {"check", "verify", "test", "exact", "over", "please"}
    )
    if object_phrases and not object_phrases & (
        _phrases(operation.title) | _phrases(operation.operation_id)
    ):
        return False
    return not (
        {"modular", "modulo"} & terms and not {"modular", "modulo"} & need_terms
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
    "OperationSearchIndex",
    "browse_operations",
    "match_operations",
    "matches_namespace",
    "need_relevance",
    "normalize_namespace",
    "operation_namespace",
]
