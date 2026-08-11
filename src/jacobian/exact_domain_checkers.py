"""Operator-controlled declarations for independent exact-operation replay."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.checker_artifacts import put_witness_envelope
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation, ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCatalogRelationship,
    CapabilityCatalogRelationshipKind,
    CapabilityCatalogRelationshipRegistration,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope
from jacobian.contracts.exact_domain_verification import (
    ExactComputedVerificationOutput,
    ExactComputedVerificationRequest,
    ExactDomainResultVerificationRequest,
    inline_exact_value_digest,
)
from jacobian.contracts.results import (
    Conclusion,
    ContractModel,
    Execution,
    ExecutionStatus,
    Verification,
)
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operations import (
    BoundedSearchOperation,
    DomainBundle,
    MaterializedOperation,
)
from jacobian.provider_runtime import source_provider_runtime
from jacobian.providers.flint_runtime import (
    certified_snf_checker_provider_runtime,
    combinatorics_exact_checker_provider_runtime,
    exact_domain_checker_provider_runtime,
    exact_domain_checker_source_provider_runtime,
    graded_syzygy_checker_provider_runtime,
    graph_exact_checker_provider_runtime,
    poset_exact_checker_provider_runtime,
    probability_exact_checker_provider_runtime,
    projective_arrangement_checker_provider_runtime,
    topology_exact_checker_provider_runtime,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

_LOGGER = logging.getLogger(__name__)
_OPTIONAL_EXACT_REPLAY_PROVIDER_KEYS = frozenset({"python-flint"})
_ENTRYPOINT_PROVIDER_RUNTIME_KEYS = {
    "jacobian_checkers.exact_domain_operations": "python-flint",
    "jacobian_checkers.graph_exact_operations": "finite-graph",
    "jacobian_checkers.exact_probability_operations": "finite-probability",
    "jacobian_checkers.recurrence_series": "combinatorics",
    "jacobian_checkers.additive_combinatorics": "combinatorics",
    "jacobian_checkers.jacobian_syzygy": "graded-syzygy",
    "jacobian_checkers.projective_arrangements": "projective-arrangement",
    "jacobian_checkers.simplicial_topology": "topology",
    "jacobian_checkers.certified_snf": "certified-snf",
    "jacobian_checkers.finite_posets": "poset",
    "jacobian_checkers.exact_geometry": "geometry",
    "jacobian_checkers.matrix_normal_forms": "matrix-hnf",
}


@dataclass(frozen=True, slots=True)
class ExactDomainCheckerInstallation:
    """Exact replay identities and non-conclusive installation diagnostics."""

    checker_ids: dict[str, str | None]
    provider_runtimes: dict[str, CapabilityProviderRuntime]
    witness_schema_uri: str | None = None
    diagnostics: tuple[CapabilityDiagnostic, ...] = ()
    catalog_relationships: tuple[CapabilityCatalogRelationshipRegistration, ...] = ()


@dataclass(frozen=True, slots=True)
class _InstalledDeclaration:
    declaration: ExactReplayCheckerDeclaration
    result_model: type[ContractModel]
    input_schema_uri: str
    result_schema_uri: str
    semantics_uri: str
    checker_id: str


def _provider_runtime_key(declaration: ExactReplayCheckerDeclaration) -> str:
    if declaration.entrypoint_module == "jacobian_checkers.linear":
        return {
            "check_rational_solution": "linear-solution",
            "check_rational_inconsistency": "linear-inconsistency",
        }[declaration.function]
    try:
        return _ENTRYPOINT_PROVIDER_RUNTIME_KEYS[declaration.entrypoint_module]
    except KeyError as exc:
        raise ValueError(
            "exact replay checker declaration uses an unsupported provider runtime"
        ) from exc


def install_exact_domain_checkers(
    checkers: CheckerRegistry,
    *,
    bundles: Mapping[str, tuple[DomainBundle, InstalledDomainBundle]],
    authorize: bool,
) -> ExactDomainCheckerInstallation:
    """Install independent exact replay against dynamically registered schemas."""

    installer = CheckerInstaller(checkers)
    provider_runtimes = {
        "python-flint": exact_domain_checker_provider_runtime(),
        "certified-snf": certified_snf_checker_provider_runtime(),
        "finite-graph": graph_exact_checker_provider_runtime(),
        "finite-probability": probability_exact_checker_provider_runtime(),
        "combinatorics": combinatorics_exact_checker_provider_runtime(),
        "poset": poset_exact_checker_provider_runtime(),
        "graded-syzygy": graded_syzygy_checker_provider_runtime(),
        "projective-arrangement": (projective_arrangement_checker_provider_runtime()),
        "topology": topology_exact_checker_provider_runtime(),
        "geometry": source_provider_runtime(
            "jacobian.exact-geometry-checker",
            version="1",
            entrypoint="jacobian_checkers.exact_geometry:check_exact_geometry",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            features=("standard-library-rational-replay", "clean-process-checker"),
        ),
        "matrix-hnf": source_provider_runtime(
            "jacobian.matrix-hnf-checker",
            version="1",
            entrypoint="jacobian_checkers.matrix_normal_forms:check_hermite_normal_form",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            features=("standard-library-integer-replay", "clean-process-checker"),
        ),
        "linear-solution": source_provider_runtime(
            "jacobian.rational-linear-checker",
            version="1",
            entrypoint="jacobian_checkers.linear:check_rational_solution",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            features=("standard-library-rational-replay", "clean-process-checker"),
        ),
        "linear-inconsistency": source_provider_runtime(
            "jacobian.rational-linear-inconsistency-checker",
            version="1",
            entrypoint="jacobian_checkers.linear:check_rational_inconsistency",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            features=("standard-library-rational-replay", "clean-process-checker"),
        ),
    }
    checker_ids: dict[str, str | None] = {}
    declarations_by_id: dict[str, ExactReplayCheckerDeclaration] = {}
    diagnostics: list[CapabilityDiagnostic] = []
    exact_checker_source_available = (
        exact_domain_checker_source_provider_runtime().availability
        is CapabilityProviderAvailability.AVAILABLE
    )
    for installed, declaration in _available_declaration_bundles(bundles):
        declarations_by_id[declaration.capability_id] = declaration
        runtime_key = _provider_runtime_key(declaration)
        provider_runtime = provider_runtimes[runtime_key]
        operation = CheckerOperation(
            name=f"{declaration.capability_id} independent {declaration.replay_method}",
            entrypoint=(f"{declaration.entrypoint_module}:{declaration.function}"),
            evidence_kind=EvidenceKind.WITNESS,
            format_id=declaration.format_id,
            format_version="1",
            claim_schema_uris=(installed.input_schema_uris[declaration.request_model],),
            semantics_uris=(installed.semantics_uri,),
            candidate_schema_uris=(
                installed.result_schema_uris[declaration.capability_id],
            ),
            reason=declaration.reason,
            provider_runtime=provider_runtime,
        )
        if (
            provider_runtime.availability
            is not CapabilityProviderAvailability.AVAILABLE
        ):
            can_omit = (
                runtime_key in _OPTIONAL_EXACT_REPLAY_PROVIDER_KEYS
                and exact_checker_source_available
            )
            if not can_omit:
                checker_ids[declaration.capability_id] = installer.install(
                    operation,
                    authorize=authorize,
                ).checker_id
                continue
            diagnostic = CapabilityDiagnostic(
                code="EXACT_REPLAY_PROVIDER_UNAVAILABLE",
                stage="provider_availability",
                message=(
                    f"Independent replay for {declaration.capability_id!r} is "
                    "not installed: "
                    f"{provider_runtime.diagnostic or 'the provider is unavailable.'}"
                ),
                hint=(
                    "Install or repair the optional python-flint backend, then retry."
                ),
                details={
                    "capability_id": declaration.capability_id,
                    "provider": provider_runtime.provider,
                    "checker_authorization_affected": True,
                },
            )
            diagnostics.append(diagnostic)
            _LOGGER.warning("%s", diagnostic.message)
            checker_ids[declaration.capability_id] = None
            continue
        checker_ids[declaration.capability_id] = installer.install(
            operation,
            authorize=authorize,
        ).checker_id
    authorized_ids = {
        runtime_key: tuple(
            checker_id
            for capability_id, checker_id in checker_ids.items()
            if checker_id is not None
            and _provider_runtime_key(declarations_by_id[capability_id]) == runtime_key
        )
        for runtime_key in provider_runtimes
    }
    return ExactDomainCheckerInstallation(
        checker_ids=checker_ids,
        diagnostics=tuple(diagnostics),
        provider_runtimes={
            "python-flint": exact_domain_checker_provider_runtime(
                checker_ids=authorized_ids["python-flint"]
            ),
            "certified-snf": certified_snf_checker_provider_runtime(
                checker_ids=authorized_ids["certified-snf"]
            ),
            "finite-graph": graph_exact_checker_provider_runtime(
                checker_ids=authorized_ids["finite-graph"]
            ),
            "finite-probability": probability_exact_checker_provider_runtime(
                checker_ids=authorized_ids["finite-probability"]
            ),
            "combinatorics": combinatorics_exact_checker_provider_runtime(
                checker_ids=authorized_ids["combinatorics"]
            ),
            "poset": poset_exact_checker_provider_runtime(
                checker_ids=authorized_ids["poset"]
            ),
            "graded-syzygy": graded_syzygy_checker_provider_runtime(
                checker_ids=authorized_ids["graded-syzygy"]
            ),
            "projective-arrangement": (
                projective_arrangement_checker_provider_runtime(
                    checker_ids=authorized_ids["projective-arrangement"]
                )
            ),
            "topology": topology_exact_checker_provider_runtime(
                checker_ids=authorized_ids["topology"]
            ),
            "geometry": source_provider_runtime(
                "jacobian.exact-geometry-checker",
                version="1",
                entrypoint="jacobian_checkers.exact_geometry:check_exact_geometry",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT",
                features=(
                    "standard-library-rational-replay",
                    "clean-process-checker",
                ),
                checker_ids=authorized_ids["geometry"],
            ),
            "matrix-hnf": source_provider_runtime(
                "jacobian.matrix-hnf-checker",
                version="1",
                entrypoint="jacobian_checkers.matrix_normal_forms:check_hermite_normal_form",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT",
                features=("standard-library-integer-replay", "clean-process-checker"),
                checker_ids=authorized_ids["matrix-hnf"],
            ),
            "linear-solution": source_provider_runtime(
                "jacobian.rational-linear-checker",
                version="1",
                entrypoint="jacobian_checkers.linear:check_rational_solution",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT",
                features=("standard-library-rational-replay", "clean-process-checker"),
                checker_ids=authorized_ids["linear-solution"],
            ),
            "linear-inconsistency": source_provider_runtime(
                "jacobian.rational-linear-inconsistency-checker",
                version="1",
                entrypoint="jacobian_checkers.linear:check_rational_inconsistency",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT",
                features=(
                    "standard-library-rational-replay",
                    "clean-process-checker",
                ),
                checker_ids=authorized_ids["linear-inconsistency"],
            ),
        },
    )


def install_exact_domain_verification(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    bundles: Mapping[str, tuple[DomainBundle, InstalledDomainBundle]],
    authorize: bool,
) -> tuple[tuple[CapabilityAdapter, ...], ExactDomainCheckerInstallation]:
    """Authorize exact replay and expose per-producer verification capabilities.

    Each authorized exact replay declaration becomes one
    :class:`ExactComputedVerificationAdapter` exposing a per-producer typed
    verifier contract for inline results. The verifier capability ID, title,
    description, and tags come from the declaration's verification metadata,
    which is always complete after construction (explicit or strictly derived
    by stripping the producer verb and appending ``.verify``).
    """

    installed = install_exact_domain_checkers(
        checkers,
        bundles=bundles,
        authorize=authorize,
    )
    witness_schema_uri = schemas.register_model(
        name="jacobian.witness-envelope",
        version="1",
        model=WitnessEnvelope,
    )
    installation = ExactDomainCheckerInstallation(
        checker_ids=installed.checker_ids,
        witness_schema_uri=witness_schema_uri,
        diagnostics=installed.diagnostics,
        provider_runtimes=installed.provider_runtimes,
    )
    if not any(
        checker_id is not None for checker_id in installation.checker_ids.values()
    ):
        return (), installation
    adapters: list[CapabilityAdapter] = []
    catalog_relationships: list[CapabilityCatalogRelationshipRegistration] = []
    result_models = {
        operation.capability_id: operation.result_model
        for bundle, _installed_bundle in bundles.values()
        for operation in bundle.capabilities
    }
    stored_producers = {
        operation.capability_id
        for bundle, _installed_bundle in bundles.values()
        for operation in bundle.capabilities
        if isinstance(operation, (MaterializedOperation, BoundedSearchOperation))
    }
    for installed_bundle, declaration in _available_declaration_bundles(bundles):
        if declaration.capability_id not in installed_bundle.result_schema_uris:
            continue
        if installation.checker_ids.get(declaration.capability_id) is None:
            continue
        installed_declaration = _installed_declaration(
            installed_bundle,
            declaration,
            installation,
            result_models[declaration.capability_id],
        )
        adapters.append(
            ExactComputedVerificationAdapter(
                declaration=installed_declaration,
                store=store,
                schemas=schemas,
                artifacts=artifacts,
                verification=verification,
                witness_schema_uri=witness_schema_uri,
                provider_runtime=installation.provider_runtimes[
                    _provider_runtime_key(declaration)
                ],
                stored_result_input=declaration.capability_id in stored_producers,
            )
        )
        verifier_id = declaration.verification_capability_id
        if verifier_id is None:
            raise ValueError("exact replay declaration has no verifier capability ID")
        catalog_relationships.extend(
            (
                CapabilityCatalogRelationshipRegistration(
                    source_capability_id=declaration.capability_id,
                    related_capability=CapabilityCatalogRelationship(
                        capability_id=verifier_id,
                        kind=(CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER),
                        relationship=(
                            "independently verify this exact producer result"
                        ),
                    ),
                ),
                CapabilityCatalogRelationshipRegistration(
                    source_capability_id=verifier_id,
                    related_capability=CapabilityCatalogRelationship(
                        capability_id=declaration.capability_id,
                        kind=(
                            CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER
                        ),
                        relationship=(
                            "produce the exact result accepted by this verifier"
                        ),
                    ),
                ),
            )
        )
    installation = replace(
        installation, catalog_relationships=tuple(catalog_relationships)
    )
    return tuple(adapters), installation


def _available_declaration_bundles(
    bundles: Mapping[str, tuple[DomainBundle, InstalledDomainBundle]],
) -> tuple[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration], ...]:
    """Pair domain-owned declarations with their unique installed producer."""

    available: list[tuple[InstalledDomainBundle, ExactReplayCheckerDeclaration]] = []
    owners: dict[str, str] = {}
    for domain_id, (bundle, installed) in bundles.items():
        producer_capability_ids = {
            operation.capability_id for operation in bundle.capabilities
        }
        for declaration in bundle.checker_declarations:
            if declaration.capability_id not in producer_capability_ids:
                raise ValueError(
                    "exact replay declaration is not backed by a domain producer "
                    f"schema: {domain_id}/{declaration.capability_id}"
                )
            if declaration.capability_id not in installed.result_schema_uris:
                continue
            previous = owners.setdefault(declaration.capability_id, domain_id)
            if previous != domain_id:
                raise ValueError(
                    "exact replay declaration is owned by multiple bundles: "
                    f"{declaration.capability_id}"
                )
            available.append((installed, declaration))
    capability_ids = [declaration.capability_id for _, declaration in available]
    if len(capability_ids) != len(set(capability_ids)):
        duplicates = sorted(
            capability_id
            for capability_id in set(capability_ids)
            if capability_ids.count(capability_id) > 1
        )
        if duplicates:
            raise ValueError(
                "bundle repeats exact replay declarations: " + ", ".join(duplicates)
            )
    return tuple(available)


def _installed_declaration(
    bundle: InstalledDomainBundle,
    declaration: ExactReplayCheckerDeclaration,
    installation: ExactDomainCheckerInstallation,
    result_model: type[ContractModel],
) -> _InstalledDeclaration:
    checker_id = installation.checker_ids[declaration.capability_id]
    if checker_id is None:
        raise ValueError("exact-domain checker is not authorized")
    return _InstalledDeclaration(
        declaration=declaration,
        result_model=result_model,
        input_schema_uri=bundle.input_schema_uris[declaration.request_model],
        result_schema_uri=bundle.result_schema_uris[declaration.capability_id],
        semantics_uri=bundle.semantics_uri,
        checker_id=checker_id,
    )


class ExactComputedVerificationAdapter:
    """Verify one exact producer result from inline input and candidate.

    The verifier validates the inline input and candidate against the
    producer's input and result schemas and checks the authorized checker's
    bounded input scope before any artifact write. Computed operations use
    the v2 inline replay envelope; only materialized and bounded-search
    operations resolve their existing stored lineage.
    """

    def __init__(
        self,
        *,
        declaration: _InstalledDeclaration,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        artifacts: ArtifactService,
        verification: VerificationService,
        witness_schema_uri: str,
        provider_runtime: CapabilityProviderRuntime,
        stored_result_input: bool,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.artifacts = artifacts
        self.verification = verification
        self.declaration = declaration
        self.witness_schema_uri = witness_schema_uri
        self.stored_result_input = stored_result_input
        generic_request_model: Any = ExactComputedVerificationRequest
        self.input_model = generic_request_model[
            declaration.declaration.request_model,
            declaration.result_model,
        ]
        verification_capability_id = declaration.declaration.verification_capability_id
        verification_title = declaration.declaration.verification_title
        verification_description = declaration.declaration.verification_description
        if (
            verification_capability_id is None
            or verification_title is None
            or verification_description is None
        ):
            raise ValueError(
                "exact replay declaration has incomplete verifier metadata"
            )
        self._descriptor = CapabilityDescriptor(
            capability_id=verification_capability_id,
            version="1",
            title=verification_title,
            description=verification_description,
            provider=provider_runtime.provider,
            provider_runtime=provider_runtime,
            input_schema=model_schema(
                ExactDomainResultVerificationRequest
                if stored_result_input
                else self.input_model
            ),
            output_schema=model_schema(ExactComputedVerificationOutput),
            tags=declaration.declaration.verification_tags,
            accepted_input_kinds=(
                (CapabilityInputKind.TYPED_ARTIFACT,)
                if stored_result_input
                else (CapabilityInputKind.STRUCTURED_REQUEST,)
            ),
            accepted_artifact_types=(
                (declaration.result_schema_uri,) if stored_result_input else ()
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        declaration = self.declaration
        source_artifacts: tuple[StoredArtifact, StoredArtifact, StoredArtifact] | None
        normalized_candidate: dict[str, object] | None
        if self.stored_result_input:
            source_artifacts = self._resolve_stored_result(request)
            normalized_input = source_artifacts[0].payload
            normalized_candidate = None
        else:
            normalized_input, normalized_candidate = self._validated_inline_payloads(
                request
            )
            source_artifacts = None
        # Check the authorized checker's bounded input scope before any artifact write.
        if not _checker_supports(
            declaration.declaration.capability_id,
            normalized_input,
        ):
            output = ExactComputedVerificationOutput(
                status="UNSUPPORTED",
                conclusion="UNKNOWN",
                operation_id=declaration.declaration.capability_id,
                input_uri=(
                    source_artifacts[0].artifact_uri
                    if source_artifacts is not None
                    else None
                ),
                result_uri=(
                    source_artifacts[1].artifact_uri
                    if source_artifacts is not None
                    else None
                ),
                checker_id=declaration.checker_id,
                detail=(
                    "The authorized checker does not support this input's bounded "
                    "scope; no mathematical conclusion follows."
                ),
            )
            return CapabilityResult(
                capability_id=self.descriptor.capability_id,
                capability_version=self.descriptor.version,
                execution=Execution(status=ExecutionStatus.COMPLETED),
                output=output.model_dump(mode="json"),
                scope=CapabilityScope(
                    description="the authorized independent checker's bounded scope",
                    parameters={
                        "operation_id": declaration.declaration.capability_id,
                        "scope_supported": False,
                    },
                    artifact_uri=(
                        source_artifacts[0].artifact_uri
                        if source_artifacts is not None
                        else None
                    ),
                ),
                completeness=CapabilityCompleteness(
                    status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                    basis="the input lies outside this checker's declared scope",
                    assurance_level=CapabilityAssuranceLevel.COMPUTED,
                ),
                assurance=CapabilityAssurance(
                    level=CapabilityAssuranceLevel.COMPUTED,
                    basis=(
                        "checker scope was evaluated without making a "
                        "mathematical conclusion"
                    ),
                ),
                artifact_uris=(
                    (source_artifacts[0].artifact_uri, source_artifacts[1].artifact_uri)
                    if source_artifacts is not None
                    else ()
                ),
            )
        if source_artifacts is None:
            if normalized_candidate is None:
                raise AssertionError("inline replay candidate was not validated")
            return self._verify_inline_relation(
                request, normalized_input, normalized_candidate
            )
        input_artifact, result_artifact, semantics_artifact = source_artifacts
        witness = put_witness_envelope(
            self.artifacts,
            witness_schema_uri=self.witness_schema_uri,
            witness_format=declaration.declaration.format_id,
            claim_artifact=input_artifact,
            semantics_artifact=semantics_artifact,
            candidate_artifact=result_artifact,
            payload={
                "operation_id": declaration.declaration.capability_id,
                "input_uri": input_artifact.artifact_uri,
                "result_uri": result_artifact.artifact_uri,
            },
            summary=(
                f"{declaration.declaration.capability_id} independent replay witness"
            ),
        )
        return self._verify_materialized_relation(
            request,
            input_artifact,
            result_artifact,
            self.store.get(witness.artifact_uri),
        )

    def _verify_inline_relation(
        self,
        request: CapabilityRequest,
        normalized_input: dict[str, object],
        normalized_candidate: dict[str, object],
    ) -> CapabilityResult:
        """Replay ordinary values through the checker without storing them."""

        declaration = self.declaration
        semantics = self.store.get(declaration.semantics_uri)
        bindings = EvidenceBindings(
            claim_digest=inline_exact_value_digest(
                schema_uri=declaration.input_schema_uri,
                semantics_uri=declaration.semantics_uri,
                payload=normalized_input,
            ),
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=inline_exact_value_digest(
                schema_uri=declaration.result_schema_uri,
                semantics_uri=declaration.semantics_uri,
                payload=normalized_candidate,
            ),
        )
        checker_request = {
            "request_version": "2",
            "operation_id": declaration.declaration.capability_id,
            "claim": {
                "schema_uri": declaration.input_schema_uri,
                "semantics_uri": declaration.semantics_uri,
                "payload": normalized_input,
            },
            "candidate": {
                "schema_uri": declaration.result_schema_uri,
                "semantics_uri": declaration.semantics_uri,
                "payload": normalized_candidate,
            },
            "semantics": self._checker_artifact(semantics),
            "scope": None,
            "expected_bindings": bindings.model_dump(mode="json"),
        }
        checked = self.verification.verify_inline_exact(
            operation_id=declaration.declaration.capability_id,
            claim_schema_uri=declaration.input_schema_uri,
            candidate_schema_uri=declaration.result_schema_uri,
            semantics_uri=declaration.semantics_uri,
            claim_payload=normalized_input,
            candidate_payload=normalized_candidate,
            checker_id=declaration.checker_id,
            witness_format=declaration.declaration.format_id,
            request=checker_request,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.assurance.verification is Verification.VERIFIED
            and checked.verification_record_uri is not None
        )
        status = self._verification_status(checked.execution.status, verified)
        record_uri = checked.verification_record_uri if verified else None
        detail = checked.execution.detail or (
            checked.input.errors[0]
            if checked.input.errors
            else (
                "the authorized independent checker accepted the exact result"
                if verified
                else "the exact result was not independently accepted"
            )
        )
        output = ExactComputedVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            operation_id=declaration.declaration.capability_id,
            checker_id=declaration.checker_id,
            verification_record_uri=record_uri,
            detail=detail,
        )
        completed = checked.execution.status is ExecutionStatus.COMPLETED
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the exact inline operation input and result",
                parameters={
                    "operation_id": declaration.declaration.capability_id,
                    "claim_digest": bindings.claim_digest,
                    "candidate_digest": bindings.candidate_digest,
                    "semantics_digest": bindings.semantics_digest,
                    "checker_id": declaration.checker_id,
                    "witness_format": declaration.declaration.format_id,
                },
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis="direct exact replay makes no search-completeness claim",
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if completed
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if completed
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                basis=(
                    "accepted in a clean process by the operator-authorized independent exact replay checker"
                    if verified
                    else (
                        "checker replay completed without accepting the candidate; no opposite conclusion follows"
                        if completed
                        else "checker replay did not complete; no conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=(
                (record_uri, declaration.semantics_uri)
                if record_uri is not None
                else ()
            ),
        )

    def _verify_materialized_relation(
        self,
        request: CapabilityRequest,
        input_artifact: StoredArtifact,
        result_artifact: StoredArtifact,
        witness: StoredArtifact,
    ) -> CapabilityResult:
        checked = self.verification.verify_witness(
            claim_uri=input_artifact.artifact_uri,
            candidate_uri=result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=self.declaration.checker_id,
            include_artifact_metadata=True,
            include_semantics_artifact=True,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion is Conclusion.TRUE
            and checked.assurance.verification is Verification.VERIFIED
            and checked.verification_record_uri is not None
        )
        status = self._verification_status(checked.execution.status, verified)
        detail = checked.execution.detail or (
            checked.input.errors[0]
            if checked.input.errors
            else (
                "the authorized independent checker accepted the exact result"
                if verified
                else "the exact result was not independently accepted"
            )
        )
        record_uri = checked.verification_record_uri if verified else None
        artifact_uris = (
            input_artifact.artifact_uri,
            result_artifact.artifact_uri,
            witness.artifact_uri,
            *((record_uri,) if record_uri is not None else ()),
        )
        completed = checked.execution.status is ExecutionStatus.COMPLETED
        output = ExactComputedVerificationOutput(
            status=status,
            conclusion="TRUE" if verified else "UNKNOWN",
            operation_id=self.declaration.declaration.capability_id,
            input_uri=input_artifact.artifact_uri,
            result_uri=result_artifact.artifact_uri,
            witness_uri=witness.artifact_uri,
            checker_id=self.declaration.checker_id,
            verification_record_uri=record_uri,
            detail=detail,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the complete stored exact operation input and result",
                parameters={
                    "operation_id": self.declaration.declaration.capability_id,
                    "input_uri": input_artifact.artifact_uri,
                    "result_uri": result_artifact.artifact_uri,
                },
                artifact_uri=input_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis="direct exact replay makes no search-completeness claim",
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if completed
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if completed
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                basis=(
                    "accepted in a clean process by the operator-authorized independent exact replay checker"
                    if verified
                    else (
                        "checker replay completed without accepting the candidate; no opposite conclusion follows"
                        if completed
                        else "checker replay did not complete; no conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=artifact_uris,
        )

    @staticmethod
    def _verification_status(
        execution_status: ExecutionStatus, verified: bool
    ) -> Literal["VERIFIED", "REJECTED", "TIMEOUT", "CANCELLED", "ERROR"]:
        if verified:
            return "VERIFIED"
        statuses: dict[ExecutionStatus, Literal["REJECTED", "TIMEOUT", "CANCELLED"]] = {
            ExecutionStatus.COMPLETED: "REJECTED",
            ExecutionStatus.TIMEOUT: "TIMEOUT",
            ExecutionStatus.CANCELLED: "CANCELLED",
        }
        return statuses.get(execution_status, "ERROR")

    def _validated_inline_payloads(
        self, request: CapabilityRequest
    ) -> tuple[dict[str, object], dict[str, object]]:
        declaration = self.declaration
        try:
            validated = self.input_model.model_validate(request.input)
            normalized_input = self.schemas.validate(
                declaration.input_schema_uri,
                validated.input.model_dump(mode="json"),
            )
            normalized_candidate = self.schemas.validate(
                declaration.result_schema_uri,
                validated.candidate.model_dump(mode="json"),
            )
        except (SchemaRegistryError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXACT_DOMAIN_INPUT",
                    stage="request_validation",
                    message=str(exc),
                    hint=(
                        "input must satisfy the producer request contract and "
                        "candidate must satisfy its result contract."
                    ),
                )
            ) from exc
        return normalized_input, normalized_candidate

    @staticmethod
    def _checker_artifact(artifact: StoredArtifact) -> dict[str, object]:
        return {
            "artifact_uri": artifact.artifact_uri,
            "object_digest": artifact.manifest.object_digest,
            "payload_digest": artifact.manifest.payload_digest,
            "schema_uri": artifact.manifest.schema_uri,
            "semantics_uri": artifact.manifest.semantics_uri,
            "parents": list(artifact.manifest.parents),
            "payload": artifact.payload,
        }

    def _resolve_stored_result(
        self, request: CapabilityRequest
    ) -> tuple[StoredArtifact, StoredArtifact, StoredArtifact]:
        """Resolve the declared producer's exact materialized lineage."""

        declaration = self.declaration
        try:
            result_uri = ExactDomainResultVerificationRequest.model_validate(
                request.input
            ).result_uri
            result_artifact = self.store.get(result_uri)
            if (
                result_artifact.manifest.schema_uri != declaration.result_schema_uri
                or result_artifact.manifest.semantics_uri != declaration.semantics_uri
                or len(result_artifact.manifest.parents) != 1
            ):
                raise ValueError("result_uri is not this producer's exact result")
            input_artifact = self.store.get(result_artifact.manifest.parents[0])
            if (
                input_artifact.manifest.schema_uri != declaration.input_schema_uri
                or input_artifact.manifest.semantics_uri != declaration.semantics_uri
            ):
                raise ValueError("result lineage does not identify the producer input")
            self.schemas.validate(declaration.input_schema_uri, input_artifact.payload)
            self.schemas.validate(
                declaration.result_schema_uri, result_artifact.payload
            )
            semantics_artifact = self.store.get(declaration.semantics_uri)
        except (SchemaRegistryError, StorageError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXACT_DOMAIN_RESULT",
                    stage="artifact_resolution",
                    message=str(exc),
                    path="result_uri",
                    hint="Pass the result_uri returned by this exact producer.",
                )
            ) from exc
        return input_artifact, result_artifact, semantics_artifact


def _checker_supports(operation_id: str, payload: object) -> bool:
    if operation_id in {
        "graph.hamiltonian_path.decide",
        "graph.induced_tree.maximum.compute",
    }:
        maximum_order = 18 if operation_id == "graph.hamiltonian_path.decide" else 16
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("graph"), dict)
            and isinstance(payload["graph"].get("vertices"), list)
            and len(payload["graph"]["vertices"]) <= maximum_order
        )
    if operation_id == "geometry.projective_line_arrangement.flats.materialize":
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("lines"), list)
            and len(payload["lines"]) <= 64
        )
    if not operation_id.startswith("polynomial."):
        return True
    if not isinstance(payload, dict):
        return False
    if operation_id == "polynomial.jacobian_syzygy.minimum_degree.compute":
        return True
    polynomial_fields = {
        "polynomial.compute.gcd": ("left", "right"),
        "polynomial.compute.resultant": ("left", "right"),
        "polynomial.compute.discriminant": ("polynomial",),
        "polynomial.compute.square_free_decomposition": ("polynomial",),
    }.get(operation_id)
    if polynomial_fields is None:
        return False
    return all(
        isinstance(payload.get(field), dict)
        and payload[field].get("variables")
        and len(payload[field]["variables"]) == 1
        for field in polynomial_fields
    )


__all__ = [
    "ExactComputedVerificationAdapter",
    "ExactDomainCheckerInstallation",
    "install_exact_domain_checkers",
    "install_exact_domain_verification",
]
