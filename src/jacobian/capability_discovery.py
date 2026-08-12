"""Deterministic capability discovery and routing projection."""

from __future__ import annotations

import re
from typing import Literal, Protocol

from jacobian.capability_errors import CapabilityDiscoveryCursorError
from jacobian.contracts.capabilities import (
    CapabilityCatalog,
    CapabilityDescriptor,
    CapabilityDiscoveryMatch,
    CapabilityDiscoveryRequest,
    CapabilityDiscoveryResult,
    CapabilityInputKind,
)

_DISCOVERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DISCOVERY_STOP_WORDS = frozenset(
    {"a", "an", "and", "for", "find", "from", "in", "of", "on", "the", "to", "with"}
)


class DiscoveryOwner(Protocol):
    def catalog(self) -> CapabilityCatalog: ...


class CapabilityDiscoveryMixin:
    """Provide the installed-portfolio discovery state machine."""

    def discover(
        self: DiscoveryOwner,
        request: CapabilityDiscoveryRequest,
    ) -> CapabilityDiscoveryResult:
        descriptors = self.catalog().capabilities
        normalized_domain = (
            normalize_domain(request.domain) if request.domain is not None else None
        )
        ranked: list[tuple[int, CapabilityDiscoveryMatch]] = []
        for descriptor in descriptors:
            if normalized_domain is not None and not matches_domain(
                descriptor,
                normalized_domain,
            ):
                continue
            applicability, applicability_code = discovery_applicability(
                descriptor,
                request.input_kind,
                request.artifact_type,
            )
            score = discovery_relevance(descriptor, request.query)
            match = CapabilityDiscoveryMatch(
                capability_id=descriptor.capability_id,
                title=descriptor.title,
                description=descriptor.description,
                tags=descriptor.tags,
                relevance_score=score,
                applicability=applicability,
                applicability_code=applicability_code,
            )
            if score > 0:
                ranked.append((score, match))
        ranked.sort(key=lambda item: (-item[0], item[1].capability_id))
        total_matches = len(ranked)
        start = 0
        if request.cursor is not None:
            try:
                start = (
                    next(
                        index
                        for index, (_, match) in enumerate(ranked)
                        if match.capability_id == request.cursor
                    )
                    + 1
                )
            except StopIteration:
                raise CapabilityDiscoveryCursorError(
                    "cursor is not present in the filtered discovery result"
                ) from None
        page = ranked[start : start + request.limit]
        next_cursor = (
            page[-1][1].capability_id
            if page and start + len(page) < total_matches
            else None
        )
        return CapabilityDiscoveryResult(
            query=request.query,
            domain=normalized_domain,
            input_kind=request.input_kind,
            artifact_type=request.artifact_type,
            matches=tuple(match for _, match in page),
            total_matches=total_matches,
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


def discovery_applicability(
    descriptor: CapabilityDescriptor,
    input_kind: CapabilityInputKind | None,
    artifact_type: str | None,
) -> tuple[
    Literal["INCOMPATIBLE", "NEEDS_MORE_TYPED_REQUIREMENTS"],
    Literal[
        "FULL_REQUEST_REQUIRED",
        "INPUT_KIND_MISMATCH",
        "ARTIFACT_TYPE_MISMATCH",
    ],
]:
    """Return only compatibility facts established by the supplied filters."""

    if input_kind is not None and input_kind not in descriptor.accepted_input_kinds:
        return "INCOMPATIBLE", "INPUT_KIND_MISMATCH"
    if (
        artifact_type is not None
        and artifact_type not in descriptor.accepted_artifact_types
    ):
        return "INCOMPATIBLE", "ARTIFACT_TYPE_MISMATCH"
    return "NEEDS_MORE_TYPED_REQUIREMENTS", "FULL_REQUEST_REQUIRED"


def token_set(value: str) -> frozenset[str]:
    return frozenset(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def discovery_relevance(
    descriptor: CapabilityDescriptor,
    query: str,
) -> int:
    query_terms = discovery_terms(query)
    if not query_terms:
        return 0
    identifier_terms = token_set(descriptor.capability_id)
    tag_terms = frozenset(term for tag in descriptor.tags for term in token_set(tag))
    title_terms = token_set(descriptor.title)
    description_terms = token_set(descriptor.description)
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
        f"{descriptor.capability_id} {descriptor.title} {descriptor.description}"
    )
    if normalized_query and f"-{normalized_query}-" in f"-{normalized_text}-":
        score += 20
    return score


def normalize_domain(value: str) -> str:
    return "_".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def capability_domain(descriptor: CapabilityDescriptor) -> str:
    return descriptor.capability_id.partition(".")[0]


def matches_domain(descriptor: CapabilityDescriptor, normalized_domain: str) -> bool:
    normalized_tags = {normalize_domain(tag) for tag in descriptor.tags}
    return (
        normalized_domain == normalize_domain(capability_domain(descriptor))
        or normalized_domain in normalized_tags
    )


__all__ = [
    "CapabilityDiscoveryMixin",
    "capability_domain",
    "discovery_applicability",
    "discovery_relevance",
    "matches_domain",
    "normalize_domain",
]
