"""Extensible model-facing capability registry and invocation service."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCatalog,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityDiscoveryMatch,
    CapabilityDiscoveryRequest,
    CapabilityDiscoveryResult,
    CapabilityMode,
    CapabilityObligationStatus,
    CapabilityProviderAvailability,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.memory import ResearchEpisode
from jacobian.contracts.results import Coverage, Execution, ExecutionStatus
from jacobian.contracts.verification import VerificationRecord
from jacobian.memory import ResearchMemory
from jacobian.schema_validation import check_draft202012_schema
from jacobian.store import ArtifactStore, StoreError

if TYPE_CHECKING:
    from jacobian.kernel import JacobianKernel

_ENTRYPOINT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_DISCOVERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LOGGER = logging.getLogger(__name__)
_DISCOVERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "find",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)


class CapabilityError(RuntimeError):
    """A capability descriptor, request, or assurance boundary is invalid."""


class _PayloadValidationError(CapabilityError):
    """Structured descriptor-schema failure safe for public diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        actual_type: str,
        expected: str,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.actual_type = actual_type
        self.expected = expected


class CapabilityDiscoveryCursorError(ValueError):
    """A continuation cursor does not belong to the filtered discovery result."""


class CapabilityInvocationError(RuntimeError):
    """An expected adapter failure that is safe to return to a model."""

    def __init__(self, diagnostic: CapabilityDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


CapabilityPolicyProfile = Literal["DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"]


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    """Operator-controlled capability visibility and invocation policy.

    Policy can restrict evaluation/runtime availability, but it never installs,
    authorizes, or changes the authority of a checker.
    """

    profile: CapabilityPolicyProfile = "DEFAULT"
    allowed_capability_ids: frozenset[str] = frozenset()
    denied_capability_ids: frozenset[str] = frozenset()
    allowed_domains: frozenset[str] = frozenset()
    denied_domains: frozenset[str] = frozenset()
    allowed_tags: frozenset[str] = frozenset()
    denied_tags: frozenset[str] = frozenset()
    allowed_modes: frozenset[CapabilityMode] = frozenset()
    denied_modes: frozenset[CapabilityMode] = frozenset()

    def __post_init__(self) -> None:
        if self.profile not in {"DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"}:
            raise ValueError(f"unknown capability policy profile: {self.profile!r}")
        if any(
            not isinstance(mode, CapabilityMode)
            for mode in self.allowed_modes | self.denied_modes
        ):
            raise ValueError("capability policy modes must be CapabilityMode values")
        if self.profile == "COMPUTE_VERIFY_NO_RETRIEVAL":
            object.__setattr__(
                self,
                "denied_capability_ids",
                self.denied_capability_ids | {"knowledge.search"},
            )
            object.__setattr__(
                self,
                "denied_tags",
                self.denied_tags | {"retrieval"},
            )
        for allowed, denied, label in (
            (
                self.allowed_capability_ids,
                self.denied_capability_ids,
                "capability IDs",
            ),
            (self.allowed_domains, self.denied_domains, "domains"),
            (self.allowed_tags, self.denied_tags, "tags"),
            (self.allowed_modes, self.denied_modes, "modes"),
        ):
            overlap = allowed & denied
            if overlap:
                raise ValueError(
                    f"capability policy allows and denies the same {label}: "
                    + ", ".join(sorted(str(item) for item in overlap))
                )
        for value in (
            *self.allowed_capability_ids,
            *self.denied_capability_ids,
            *self.allowed_domains,
            *self.denied_domains,
            *self.allowed_tags,
            *self.denied_tags,
        ):
            if not value.strip():
                raise ValueError("capability policy values must not be blank")

    @property
    def definition(self) -> dict[str, object]:
        return {
            "policy_version": "1",
            "profile": self.profile,
            "allowed_capability_ids": sorted(self.allowed_capability_ids),
            "denied_capability_ids": sorted(self.denied_capability_ids),
            "allowed_domains": sorted(self.allowed_domains),
            "denied_domains": sorted(self.denied_domains),
            "allowed_tags": sorted(self.allowed_tags),
            "denied_tags": sorted(self.denied_tags),
            "allowed_modes": sorted(mode.value for mode in self.allowed_modes),
            "denied_modes": sorted(mode.value for mode in self.denied_modes),
            "checker_authorization_affected": False,
        }

    @property
    def digest(self) -> str:
        return (
            "sha256:" + hashlib.sha256(canonicalize_json(self.definition)).hexdigest()
        )

    def project(
        self,
        descriptor: CapabilityDescriptor,
    ) -> CapabilityDescriptor | None:
        reasons = self.denial_reasons(descriptor)
        if reasons:
            return None
        visible_modes = tuple(
            mode
            for mode in descriptor.modes
            if (not self.allowed_modes or mode in self.allowed_modes)
            and mode not in self.denied_modes
        )
        if not visible_modes:
            return None
        return descriptor.model_copy(update={"modes": visible_modes})

    def denial_reasons(
        self,
        descriptor: CapabilityDescriptor,
        *,
        mode: CapabilityMode | None = None,
    ) -> tuple[str, ...]:
        capability_id = descriptor.capability_id
        domain = _normalize_domain(_capability_domain(descriptor))
        tags = {_normalize_domain(tag) for tag in descriptor.tags}
        reasons: list[str] = []
        if (
            self.allowed_capability_ids
            and capability_id not in self.allowed_capability_ids
        ):
            reasons.append("capability_id_not_allowed")
        if capability_id in self.denied_capability_ids:
            reasons.append("capability_id_denied")
        if self.allowed_domains and domain not in {
            _normalize_domain(value) for value in self.allowed_domains
        }:
            reasons.append("domain_not_allowed")
        if domain in {_normalize_domain(value) for value in self.denied_domains}:
            reasons.append("domain_denied")
        normalized_allowed_tags = {
            _normalize_domain(value) for value in self.allowed_tags
        }
        if normalized_allowed_tags and not tags & normalized_allowed_tags:
            reasons.append("tag_not_allowed")
        if tags & {_normalize_domain(value) for value in self.denied_tags}:
            reasons.append("tag_denied")
        if mode is not None:
            if self.allowed_modes and mode not in self.allowed_modes:
                reasons.append("mode_not_allowed")
            if mode in self.denied_modes:
                reasons.append("mode_denied")
        return tuple(reasons)


class CapabilityAdapter(Protocol):
    """Operator-installed adapter; registration requires no MCP changes."""

    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def invoke(self, request: CapabilityRequest) -> CapabilityResult: ...


class CapabilityService:
    """Validate, dispatch, trust-check, and remember mathematical operations."""

    def __init__(
        self,
        store: ArtifactStore,
        memory: ResearchMemory,
        *,
        policy: CapabilityPolicy | None = None,
    ) -> None:
        self.store = store
        self.memory = memory
        self.policy = policy or CapabilityPolicy()
        self._adapters: dict[str, CapabilityAdapter] = {}

    def register(self, adapter: CapabilityAdapter) -> None:
        descriptor = adapter.descriptor
        if descriptor.capability_id in self._adapters:
            raise CapabilityError(
                f"duplicate capability ID: {descriptor.capability_id}"
            )
        if descriptor.provider_runtime is None:
            raise CapabilityError(
                f"capability {descriptor.capability_id} has no provider runtime identity"
            )
        if (
            descriptor.provider_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            raise CapabilityError(
                f"capability {descriptor.capability_id} is unavailable: "
                f"{descriptor.provider_runtime.diagnostic}"
            )
        _validator(descriptor.input_schema)
        _validator(descriptor.output_schema)
        for example in descriptor.invocation_examples:
            try:
                _validate_payload(descriptor.input_schema, example.input)
            except CapabilityError as exc:
                raise CapabilityError(
                    f"capability {descriptor.capability_id} invocation example "
                    f"{example.name!r} does not match its input schema"
                ) from exc
        self._adapters[descriptor.capability_id] = adapter

    def catalog(self) -> CapabilityCatalog:
        visible = tuple(
            projected
            for name in sorted(self._adapters)
            if (projected := self.policy.project(self._adapters[name].descriptor))
            is not None
        )
        return CapabilityCatalog(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            capabilities=visible,
        )

    def discover(
        self,
        request: CapabilityDiscoveryRequest,
    ) -> CapabilityDiscoveryResult:
        """Return compact installed outcomes ordered by deterministic relevance."""

        descriptors = self.catalog().capabilities
        available_domains = tuple(
            sorted({_capability_domain(descriptor) for descriptor in descriptors})
        )
        normalized_domain = (
            _normalize_domain(request.domain) if request.domain is not None else None
        )
        ranked: list[tuple[int, CapabilityDiscoveryMatch]] = []
        for descriptor in descriptors:
            if request.mode is not None and request.mode not in descriptor.modes:
                continue
            if normalized_domain is not None and not _matches_domain(
                descriptor,
                normalized_domain,
            ):
                continue
            (
                score,
                matched_on,
                matched_terms,
                query_term_count,
                query_coverage_milli,
                lexical_fit,
            ) = _discovery_relevance(
                descriptor,
                request.query,
            )
            if request.query is not None and score == 0:
                continue
            ranked.append(
                (
                    score,
                    CapabilityDiscoveryMatch(
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
                    ),
                )
            )
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
        portfolio_fit: Literal[
            "UNFILTERED",
            "STRONG_CANDIDATES_FOUND",
            "ONLY_WEAK_LEXICAL_MATCHES",
            "NO_LEXICAL_MATCHES",
        ]
        if request.query is None:
            portfolio_fit = "UNFILTERED"
            portfolio_fit_basis = (
                "No query was supplied; results are an unranked installed-portfolio "
                "listing and make no suitability claim."
            )
        elif not ranked:
            portfolio_fit = "NO_LEXICAL_MATCHES"
            portfolio_fit_basis = (
                "No installed descriptor shared a meaningful query term. This is "
                "not proof that the mathematical outcome is impossible."
            )
        elif any(match.lexical_fit == "STRONG_CANDIDATE" for _, match in ranked):
            portfolio_fit = "STRONG_CANDIDATES_FOUND"
            portfolio_fit_basis = (
                "At least one installed descriptor has substantial lexical query "
                "coverage; inspect its contract before treating it as suitable."
            )
        else:
            portfolio_fit = "ONLY_WEAK_LEXICAL_MATCHES"
            portfolio_fit_basis = (
                "Installed results share only weak lexical evidence with the query. "
                "Do not infer capability fit from top-N ordering alone."
            )
        return CapabilityDiscoveryResult(
            query=request.query,
            domain=normalized_domain,
            mode=request.mode,
            matches=tuple(match for _, match in page),
            total_matches=total_matches,
            truncated=next_cursor is not None,
            next_cursor=next_cursor,
            available_domains=available_domains,
            portfolio_fit=portfolio_fit,
            portfolio_fit_basis=portfolio_fit_basis,
        )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        try:
            adapter = self._adapters[request.capability_id]
        except KeyError:
            result = _resolution_failure(
                request=request,
                capability_version="not-installed",
                diagnostic=CapabilityDiagnostic(
                    code="UNKNOWN_CAPABILITY",
                    stage="capability_resolution",
                    message=(f"Capability {request.capability_id!r} is not installed."),
                    hint=(
                        "Call capability.describe without capability_id to list "
                        "installed capabilities, then retry with one of those IDs."
                    ),
                ),
                context={
                    "available_capability_ids": [
                        descriptor.capability_id
                        for descriptor in self.catalog().capabilities
                    ],
                },
            )
            _log_invocation(result, started)
            return result
        descriptor = adapter.descriptor
        policy_reasons = self.policy.denial_reasons(
            descriptor,
            mode=request.mode,
        )
        if policy_reasons:
            result = _resolution_failure(
                request=request,
                capability_version=descriptor.version,
                diagnostic=CapabilityDiagnostic(
                    code="CAPABILITY_POLICY_DENIED",
                    stage="capability_policy",
                    message=(
                        f"Capability {request.capability_id!r} is denied by the "
                        "operator-controlled capability policy."
                    ),
                    hint=(
                        "Choose a capability visible in capability.describe, or ask "
                        "the operator to change the evaluation/runtime policy."
                    ),
                    details={
                        "policy_profile": self.policy.profile,
                        "policy_digest": self.policy.digest,
                        "reasons": list(policy_reasons),
                        "checker_authorization_affected": False,
                    },
                ),
                context={"capability_policy": self.policy.definition},
            )
            result = result.model_copy(update=_provider_provenance(descriptor))
            _log_invocation(result, started)
            return result
        if request.mode not in descriptor.modes:
            result = _resolution_failure(
                request=request,
                capability_version=descriptor.version,
                diagnostic=CapabilityDiagnostic(
                    code="UNSUPPORTED_MODE",
                    stage="capability_resolution",
                    message=(
                        f"Capability {request.capability_id!r} does not support "
                        f"{request.mode.value} mode."
                    ),
                    hint=(
                        "Call capability.describe for this capability, then retry "
                        "with one of its advertised modes."
                    ),
                ),
                context={
                    "available_modes": [mode.value for mode in descriptor.modes],
                },
            )
            result = result.model_copy(update=_provider_provenance(descriptor))
            _log_invocation(result, started)
            return result
        try:
            normalized_input = _validate_payload(descriptor.input_schema, request.input)
        except CapabilityError as exc:
            path = (
                exc.path
                if isinstance(exc, _PayloadValidationError)
                else _error_path(exc)
            )
            result = _failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=CapabilityDiagnostic(
                    code="INVALID_REQUEST",
                    stage="capability_input_validation",
                    message=(
                        "The capability input does not match its advertised schema"
                        + (f" at {path}." if path else ".")
                    ),
                    path=path,
                    expected=(
                        exc.expected
                        if isinstance(exc, _PayloadValidationError)
                        else "input matching the capability descriptor JSON Schema"
                    ),
                    actual_type=(
                        exc.actual_type
                        if isinstance(exc, _PayloadValidationError)
                        else _json_value_type(request.input)
                    ),
                    hint=(
                        "Correct the reported field. The exact required and missing "
                        "top-level fields are included in diagnostic details."
                    ),
                    details={
                        "required_fields": descriptor.input_schema.get("required", []),
                        "missing_fields": sorted(
                            set(descriptor.input_schema.get("required", []))
                            - set(request.input)
                        ),
                    },
                ),
            )
            _log_invocation(result, started)
            return result
        normalized_request = request.model_copy(update={"input": normalized_input})
        try:
            result = CapabilityResult.model_validate(adapter.invoke(normalized_request))
        except CapabilityInvocationError as exc:
            result = _failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=exc.diagnostic,
            )
        except Exception as exc:
            _LOGGER.warning(
                "capability %s stopped during execution",
                request.capability_id,
                exc_info=exc,
            )
            result = _failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=CapabilityDiagnostic(
                    code="ADAPTER_EXECUTION_FAILED",
                    stage="adapter_execution",
                    message="The capability stopped before returning a result.",
                    hint=(
                        "Retry once. If it fails again, inspect the local Jacobian "
                        "log for this capability."
                    ),
                ),
            )
        if (
            result.capability_id != descriptor.capability_id
            or result.capability_version != descriptor.version
            or result.mode is not request.mode
        ):
            raise CapabilityError("adapter result identity differs from its request")
        if result.provider is not None and result.provider != descriptor.provider:
            raise CapabilityError(
                "adapter result provider runtime differs from its descriptor"
            )
        provenance = _provider_provenance(descriptor)
        if (
            result.provider_digest is not None
            and result.provider_digest != provenance["provider_digest"]
        ):
            raise CapabilityError(
                "adapter result provider runtime differs from its descriptor"
            )
        result = result.model_copy(update=provenance)
        if result.execution.status is ExecutionStatus.COMPLETED:
            normalized_output = _validate_payload(
                descriptor.output_schema, result.output
            )
            result = result.model_copy(update={"output": normalized_output})
        self._validate_artifact_references(result)
        self._validate_verified_result(result)
        if (
            descriptor.records_episode
            and result.execution.status is ExecutionStatus.COMPLETED
        ):
            episode_uri = self.memory.record(
                ResearchEpisode(
                    capability_id=result.capability_id,
                    capability_version=result.capability_version,
                    mode=result.mode,
                    request=normalized_request.input,
                    result=result.model_dump(
                        mode="json",
                        exclude={"episode_uri"},
                    ),
                    assurance_level=result.assurance.level,
                    verification_record_uri=(result.assurance.verification_record_uri),
                    artifact_uris=result.artifact_uris,
                    summary=_episode_summary(result),
                    tags=descriptor.tags,
                )
            )
            result = result.model_copy(update={"episode_uri": episode_uri})
        _log_invocation(result, started)
        return result

    @staticmethod
    def _validate_artifact_references(result: CapabilityResult) -> None:
        exposed = set(result.artifact_uris)
        referenced: set[str] = set()
        if result.scope is not None and result.scope.artifact_uri is not None:
            referenced.add(result.scope.artifact_uri)
        if result.completeness.verification_record_uri is not None:
            referenced.add(result.completeness.verification_record_uri)
        for relationship in result.relationships:
            referenced.update(relationship.source_artifact_uris)
            referenced.update(relationship.target_artifact_uris)
            referenced.update(relationship.obligation_uris)
            if relationship.verification_record_uri is not None:
                referenced.add(relationship.verification_record_uri)
        for obligation in result.obligations:
            referenced.add(obligation.obligation_uri)
            if obligation.verification_record_uri is not None:
                referenced.add(obligation.verification_record_uri)
        missing = referenced - exposed
        if missing:
            raise CapabilityError(
                "capability result has first-class references missing from artifact_uris"
            )

    def _validate_verified_result(self, result: CapabilityResult) -> None:
        if result.assurance.level is not CapabilityAssuranceLevel.VERIFIED:
            return
        record_uri = result.assurance.verification_record_uri
        if record_uri is None:
            raise CapabilityError(
                "verified capability result has no verification record URI"
            )
        try:
            record_artifact = self.store.get(record_uri)
            record = VerificationRecord.model_validate(record_artifact.payload)
        except (StoreError, ValueError) as exc:
            raise CapabilityError(
                "verified capability result has no valid local verification record"
            ) from exc
        if record.evidence_uri not in result.artifact_uris:
            raise CapabilityError(
                "verified capability result does not expose its checked evidence"
            )
        missing_parents = set(record_artifact.manifest.parents) - set(
            result.artifact_uris
        )
        if missing_parents:
            raise CapabilityError(
                "verified capability result omits verification-bound artifacts"
            )
        projected_record = result.output.get("verification_record_uri")
        if projected_record is not None and projected_record != record_uri:
            raise CapabilityError(
                "verified capability output projects a different verification record"
            )
        projected_conclusion = result.output.get("conclusion")
        if (
            projected_conclusion is not None
            and projected_conclusion != record.conclusion.value
        ):
            raise CapabilityError(
                "verified capability output differs from the checked conclusion"
            )
        record_parents = set(record_artifact.manifest.parents)
        for relationship in result.relationships:
            if relationship.status is not CapabilityRelationshipStatus.VERIFIED:
                continue
            bound_artifacts = {
                *relationship.source_artifact_uris,
                *relationship.target_artifact_uris,
                *relationship.obligation_uris,
            }
            if not bound_artifacts.issubset(record_parents):
                raise CapabilityError(
                    "verified relationship record does not bind its artifacts"
                )
            if record_artifact.payload.get("relation_id") != relationship.relation_id:
                raise CapabilityError(
                    "verified relationship differs from the checked relation"
                )
            if (
                record.relationship_source_artifact_uris
                != relationship.source_artifact_uris
                or record.relationship_target_artifact_uris
                != relationship.target_artifact_uris
            ):
                raise CapabilityError(
                    "verified relationship endpoints differ from the checked relation"
                )
            checked_obligations = (
                (record.obligation_uri,) if record.obligation_uri is not None else ()
            )
            if relationship.obligation_uris != checked_obligations:
                raise CapabilityError(
                    "verified relationship obligations differ from the checked relation"
                )
        for obligation in result.obligations:
            if obligation.status is not CapabilityObligationStatus.DISCHARGED:
                continue
            if (
                obligation.obligation_uri not in record_parents
                or record_artifact.payload.get("obligation_uri")
                != obligation.obligation_uri
            ):
                raise CapabilityError(
                    "discharged obligation differs from the checked obligation"
                )
        if (
            result.completeness.status is CapabilityCompletenessStatus.COMPLETE
            and result.completeness.assurance_level is CapabilityAssuranceLevel.VERIFIED
        ):
            if (
                result.scope is None
                or result.scope.artifact_uri is None
                or result.scope.artifact_uri not in record_parents
            ):
                raise CapabilityError(
                    "verified completeness requires a checker-bound scope artifact"
                )
            if record.coverage not in {Coverage.EXHAUSTIVE, Coverage.BOUNDED}:
                raise CapabilityError(
                    "verified completeness differs from checked coverage"
                )


def load_capability_adapter(
    entrypoint: str,
    kernel: JacobianKernel,
) -> CapabilityAdapter:
    """Load one operator-approved ``factory(kernel)`` adapter entrypoint."""

    if not _ENTRYPOINT_PATTERN.fullmatch(entrypoint):
        raise CapabilityError("capability adapter entrypoint has an invalid format")
    module_name, attribute_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
        adapter = factory(kernel)
        descriptor = adapter.descriptor
        invoke = adapter.invoke
    except (AttributeError, ImportError, TypeError) as exc:
        raise CapabilityError(
            f"cannot load capability adapter entrypoint: {entrypoint}"
        ) from exc
    if not isinstance(descriptor, CapabilityDescriptor) or not callable(invoke):
        raise CapabilityError("capability adapter does not implement the protocol")
    return cast(CapabilityAdapter, adapter)


@lru_cache(maxsize=1024)
def _compiled_validator(canonical_schema: bytes) -> Draft202012Validator:
    normalized = loads_strict_json(canonical_schema)
    try:
        check_draft202012_schema(canonical_schema)
    except SchemaError as exc:
        raise CapabilityError("capability JSON Schema is invalid") from exc
    return Draft202012Validator(normalized)


def _validator(schema: dict[str, object]) -> Draft202012Validator:
    return _compiled_validator(canonicalize_json(schema))


def _normalize_discovery_text(value: str) -> str:
    return "-".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def _discovery_terms(query: str) -> frozenset[str]:
    return frozenset(
        term
        for term in _DISCOVERY_TOKEN_PATTERN.findall(query.casefold())
        if term not in _DISCOVERY_STOP_WORDS
    )


def _token_set(value: str) -> frozenset[str]:
    return frozenset(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def _discovery_relevance(
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
    query_terms = _discovery_terms(query)
    if not query_terms:
        return 0, (), (), 0, 0, "WEAK_LEXICAL_MATCH"
    identifier_terms = _token_set(descriptor.capability_id)
    tag_terms = frozenset(term for tag in descriptor.tags for term in _token_set(tag))
    title_terms = _token_set(descriptor.title)
    description_terms = _token_set(descriptor.description)
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
    normalized_query = _normalize_discovery_text(query)
    normalized_text = _normalize_discovery_text(
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


def _normalize_domain(value: str) -> str:
    return "_".join(_DISCOVERY_TOKEN_PATTERN.findall(value.casefold()))


def _capability_domain(descriptor: CapabilityDescriptor) -> str:
    """Project the domain-owned namespace from one installed capability ID."""

    return descriptor.capability_id.partition(".")[0]


def _matches_domain(
    descriptor: CapabilityDescriptor,
    normalized_domain: str,
) -> bool:
    normalized_tags = {_normalize_domain(tag) for tag in descriptor.tags}
    return (
        normalized_domain == _normalize_domain(_capability_domain(descriptor))
        or normalized_domain in normalized_tags
    )


def _validate_payload(
    schema: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    normalized = loads_strict_json(canonicalize_json(payload))
    if not isinstance(normalized, dict):
        raise CapabilityError("capability payload must normalize to an object")
    errors = sorted(
        _validator(schema).iter_errors(normalized),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "$"
        raise _PayloadValidationError(
            f"{location}: {first.message}",
            path=location,
            actual_type=_json_value_type(first.instance),
            expected=_schema_expectation(first),
        )
    return normalized


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _schema_expectation(error: JsonSchemaValidationError) -> str:
    if error.validator == "enum" and isinstance(error.validator_value, list):
        allowed = ", ".join(
            json.dumps(value, ensure_ascii=False) for value in error.validator_value
        )
        return f"one of: {allowed}"
    if error.validator == "const":
        return f"the constant {json.dumps(error.validator_value, ensure_ascii=False)}"
    if error.validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            return "JSON type " + " or ".join(str(item) for item in expected)
        return f"JSON type {expected}"
    return "input matching the capability descriptor JSON Schema"


def _episode_summary(result: CapabilityResult) -> str:
    return (
        f"{result.capability_id} {result.mode.value.lower()} "
        f"{result.execution.status.value.lower()} "
        f"({result.assurance.level.value.lower()})"
    )


def _failed_result(
    *,
    descriptor: CapabilityDescriptor,
    request: CapabilityRequest,
    diagnostic: CapabilityDiagnostic,
) -> CapabilityResult:
    provenance = _provider_provenance(descriptor)
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        mode=request.mode,
        execution=Execution(
            status=ExecutionStatus.ERROR,
            detail=diagnostic.message,
        ),
        output={"error": diagnostic.model_dump(mode="json", exclude_none=True)},
        diagnostics=(diagnostic,),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.HEURISTIC,
            basis="execution or input failure; no mathematical conclusion",
        ),
        provider=provenance["provider"],
        provider_digest=provenance["provider_digest"],
    )


def _provider_provenance(
    descriptor: CapabilityDescriptor,
) -> dict[str, str]:
    if descriptor.provider_runtime is None:
        raise CapabilityError(
            f"capability {descriptor.capability_id} has no provider runtime identity"
        )
    if descriptor.provider_runtime.digest is None:
        raise CapabilityError(
            f"capability {descriptor.capability_id} has no provider runtime digest"
        )
    return {
        "provider": descriptor.provider,
        "provider_digest": descriptor.provider_runtime.digest,
    }


def _resolution_failure(
    *,
    request: CapabilityRequest,
    capability_version: str,
    diagnostic: CapabilityDiagnostic,
    context: dict[str, object],
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=request.capability_id,
        capability_version=capability_version,
        mode=request.mode,
        execution=Execution(
            status=ExecutionStatus.ERROR,
            detail=diagnostic.message,
        ),
        output={
            "error": diagnostic.model_dump(mode="json", exclude_none=True),
            **context,
        },
        diagnostics=(diagnostic,),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.HEURISTIC,
            basis="capability resolution failed; no mathematical conclusion",
        ),
    )


def _error_path(exc: Exception) -> str | None:
    path, separator, _ = str(exc).partition(": ")
    return path if separator else None


def _log_invocation(result: CapabilityResult, started: float) -> None:
    elapsed_ms = round((time.monotonic() - started) * 1000)
    diagnostic_codes = (
        ",".join(diagnostic.code for diagnostic in result.diagnostics) or "-"
    )
    _LOGGER.info(
        (
            "capability invocation capability_id=%s version=%s mode=%s "
            "status=%s assurance=%s elapsed_ms=%d diagnostics=%s episode=%s"
        ),
        result.capability_id,
        result.capability_version,
        result.mode.value,
        result.execution.status.value,
        result.assurance.level.value,
        elapsed_ms,
        diagnostic_codes,
        result.episode_uri or "-",
        extra={
            "jacobian_capability_id": result.capability_id,
            "jacobian_capability_version": result.capability_version,
            "jacobian_mode": result.mode.value,
            "jacobian_execution_status": result.execution.status.value,
            "jacobian_assurance_level": result.assurance.level.value,
            "jacobian_elapsed_ms": elapsed_ms,
            "jacobian_diagnostic_codes": tuple(
                diagnostic.code for diagnostic in result.diagnostics
            ),
            "jacobian_episode_uri": result.episode_uri,
        },
    )
