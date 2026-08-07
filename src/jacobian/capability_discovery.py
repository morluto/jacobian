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
        catalog_descriptors = self.catalog().capabilities
        descriptors = tuple(
            descriptor
            for descriptor in catalog_descriptors
            if descriptor.discovery_visible
        )
        available_domains = tuple(
            sorted({capability_domain(descriptor) for descriptor in descriptors})
        )
        normalized_domain = (
            normalize_domain(request.domain) if request.domain is not None else None
        )
        domain_filter_status, domain_filter_basis = discovery_domain_filter_status(
            catalog_descriptors,
            normalized_domain,
        )
        resolved_input_kind = request.input_kind or infer_discovery_input_kind(
            request.query
        )
        contract_route_count = 0
        lexical_candidates: list[CapabilityDiscoveryMatch] = []
        ranked: list[tuple[int, CapabilityDiscoveryMatch]] = []
        for descriptor in descriptors:
            if request.mode is not None and request.mode not in descriptor.modes:
                continue
            if normalized_domain is not None and not matches_domain(
                descriptor,
                normalized_domain,
            ):
                continue
            input_compatible = accepts_discovery_input(
                descriptor,
                resolved_input_kind,
                request.artifact_type,
            )
            if input_compatible:
                contract_route_count += 1
            (
                score,
                matched_on,
                matched_terms,
                query_term_count,
                query_coverage_milli,
                lexical_fit,
            ) = discovery_relevance(descriptor, request.query)
            match = CapabilityDiscoveryMatch(
                capability_id=descriptor.capability_id,
                title=descriptor.title,
                description=descriptor.description,
                modes=descriptor.modes,
                tags=descriptor.tags,
                matched_on=matched_on,
                matched_terms=matched_terms,
                has_invocation_examples=bool(descriptor.invocation_examples),
                relevance_score=score,
                query_term_count=query_term_count,
                query_coverage_milli=query_coverage_milli,
                lexical_fit=lexical_fit,
            )
            if request.query is None or score > 0:
                lexical_candidates.append(match)
                if input_compatible:
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
        portfolio_fit, portfolio_fit_basis = discovery_portfolio_fit(
            request.query,
            lexical_candidates,
        )
        routing_status, routing_basis = discovery_routing_status(
            resolved_input_kind,
            request.artifact_type,
            contract_route_count,
        )
        if domain_filter_status == "UNKNOWN":
            portfolio_fit = "UNFILTERED"
            portfolio_fit_basis = (
                f"The requested domain filter {normalized_domain!r} matched no "
                "installed capability; lexical fit outside that filter was not "
                "assessed."
            )
            routing_status = "UNFILTERED"
            routing_basis = (
                f"The routing status was not assessed because the requested "
                f"domain filter {normalized_domain!r} matched no installed "
                "capability."
            )
        return CapabilityDiscoveryResult(
            query=request.query,
            domain=normalized_domain,
            domain_filter_status=domain_filter_status,
            domain_filter_basis=domain_filter_basis,
            mode=request.mode,
            resolved_input_kind=resolved_input_kind,
            artifact_type=request.artifact_type,
            routing_status=routing_status,
            routing_basis=routing_basis,
            matches=tuple(match for _, match in page),
            total_matches=total_matches,
            truncated=next_cursor is not None,
            next_cursor=next_cursor,
            available_domains=available_domains,
            portfolio_fit=portfolio_fit,
            portfolio_fit_basis=portfolio_fit_basis,
        )


def discovery_domain_filter_status(
    descriptors: tuple[CapabilityDescriptor, ...],
    normalized_domain: str | None,
) -> tuple[Literal["UNFILTERED", "MATCHED", "UNKNOWN"], str]:
    """Classify a requested domain independently from lexical query fit."""

    if normalized_domain is None:
        return "UNFILTERED", "No domain filter was supplied."
    if any(matches_domain(descriptor, normalized_domain) for descriptor in descriptors):
        return (
            "MATCHED",
            f"The requested domain filter {normalized_domain!r} matches at least "
            "one installed capability domain or tag.",
        )
    return (
        "UNKNOWN",
        f"The requested domain filter {normalized_domain!r} matches no installed "
        "capability domain or tag.",
    )


def normalize_discovery_text(value: str) -> str:
    return "-".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def discovery_terms(query: str) -> frozenset[str]:
    return frozenset(
        term
        for term in _DISCOVERY_TOKEN_PATTERN.findall(query.casefold())
        if term not in _DISCOVERY_STOP_WORDS
    )


def accepts_discovery_input(
    descriptor: CapabilityDescriptor,
    input_kind: CapabilityInputKind | None,
    artifact_type: str | None,
) -> bool:
    return (input_kind is None or input_kind in descriptor.accepted_input_kinds) and (
        artifact_type is None or artifact_type in descriptor.accepted_artifact_types
    )


def discovery_portfolio_fit(
    query: str | None,
    lexical_candidates: list[CapabilityDiscoveryMatch],
) -> tuple[
    Literal[
        "UNFILTERED",
        "STRONG_CANDIDATES_FOUND",
        "ONLY_WEAK_LEXICAL_MATCHES",
        "NO_LEXICAL_MATCHES",
    ],
    str,
]:
    if query is None:
        return (
            "UNFILTERED",
            "No query was supplied; results are an unranked installed-portfolio "
            "listing and make no suitability claim.",
        )
    if not lexical_candidates:
        return (
            "NO_LEXICAL_MATCHES",
            "No installed descriptor shared a meaningful query term. This is "
            "not proof that the mathematical outcome is impossible.",
        )
    if any(match.lexical_fit == "STRONG_CANDIDATE" for match in lexical_candidates):
        return (
            "STRONG_CANDIDATES_FOUND",
            "At least one installed descriptor has substantial lexical query "
            "coverage; inspect its contract before treating it as suitable.",
        )
    return (
        "ONLY_WEAK_LEXICAL_MATCHES",
        "Installed results share only weak lexical evidence with the query. "
        "Do not infer capability fit from top-N ordering alone.",
    )


def discovery_routing_status(
    input_kind: CapabilityInputKind | None,
    artifact_type: str | None,
    route_count: int,
) -> tuple[Literal["UNFILTERED", "ROUTES_FOUND", "NO_ROUTE"], str]:
    if input_kind is None:
        return (
            "UNFILTERED",
            "No input kind was declared or safely inferred; inspect the selected "
            "capability contract before invocation.",
        )
    artifact_basis = (
        f" and artifact type {artifact_type!r}." if artifact_type is not None else "."
    )
    if route_count:
        return (
            "ROUTES_FOUND",
            "Installed routes match the declared or safely inferred input kind"
            + artifact_basis,
        )
    return (
        "NO_ROUTE",
        "No installed capability accepts the declared or safely inferred input kind"
        + artifact_basis,
    )


def infer_discovery_input_kind(query: str | None) -> CapabilityInputKind | None:
    if query is None:
        return None
    normalized = " ".join(_DISCOVERY_TOKEN_PATTERN.findall(query.casefold()))
    if any(
        phrase in normalized
        for phrase in ("natural language proof", "informal proof", "proof prose")
    ):
        return CapabilityInputKind.NATURAL_LANGUAGE_PROOF
    return None


def token_set(value: str) -> frozenset[str]:
    return frozenset(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def discovery_relevance(
    descriptor: CapabilityDescriptor,
    query: str | None,
) -> tuple[
    int,
    tuple[str, ...],
    tuple[str, ...],
    int,
    int,
    Literal["STRONG_CANDIDATE", "WEAK_LEXICAL_MATCH"],
]:
    if query is None:
        return 0, (), (), 0, 0, "WEAK_LEXICAL_MATCH"
    query_terms = discovery_terms(query)
    if not query_terms:
        return 0, (), (), 0, 0, "WEAK_LEXICAL_MATCH"
    identifier_terms = token_set(descriptor.capability_id)
    tag_terms = frozenset(term for tag in descriptor.tags for term in token_set(tag))
    title_terms = token_set(descriptor.title)
    description_terms = token_set(descriptor.description)
    score = 0
    matched_on: list[str] = []
    matched_terms: set[str] = set()
    for label, terms, weight in (
        ("capability_id", identifier_terms, 12),
        ("tags", tag_terms, 10),
        ("title", title_terms, 8),
        ("description", description_terms, 3),
    ):
        overlap = query_terms & terms
        if overlap:
            score += weight * len(overlap)
            matched_on.append(label)
            matched_terms.update(overlap)
    normalized_query = normalize_discovery_text(query)
    normalized_text = normalize_discovery_text(
        f"{descriptor.capability_id} {descriptor.title} {descriptor.description}"
    )
    if normalized_query and f"-{normalized_query}-" in f"-{normalized_text}-":
        score += 20
        matched_on.append("phrase")
    query_term_count = len(query_terms)
    query_coverage_milli = (
        1000 * len(matched_terms) // query_term_count if query_term_count else 0
    )
    strong = "phrase" in matched_on or (
        query_coverage_milli >= 500
        and (len(matched_terms) >= 2 or query_term_count == 1)
        and any(label in matched_on for label in ("capability_id", "tags", "title"))
    )
    return (
        score,
        tuple(matched_on),
        tuple(sorted(matched_terms)),
        query_term_count,
        query_coverage_milli,
        "STRONG_CANDIDATE" if strong else "WEAK_LEXICAL_MATCH",
    )


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
    "accepts_discovery_input",
    "capability_domain",
    "discovery_domain_filter_status",
    "discovery_relevance",
    "infer_discovery_input_kind",
    "matches_domain",
    "normalize_domain",
]
