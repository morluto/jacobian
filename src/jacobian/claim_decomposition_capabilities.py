"""Deterministic top-level decomposition of validated structured claims."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.claim_decomposition import (
    ClaimDecompositionArtifact,
    ClaimDecompositionOutput,
    ClaimDecompositionRequest,
    DecomposedOccurrence,
    LogicalClaimNode,
    LogicalConnective,
    ReconstructionRecord,
    SourceArtifactBinding,
    StructuredClaimArtifact,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository


def _digest(node: LogicalClaimNode) -> str:
    payload = canonicalize_json(node.model_dump(mode="json"))
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def reconstruct(artifact: ClaimDecompositionArtifact) -> LogicalClaimNode:
    """Rebuild the exact source root from ordered occurrence records."""

    actual_child_digests = tuple(_digest(item.node) for item in artifact.occurrences)
    if actual_child_digests != artifact.reconstruction.ordered_child_digests:
        raise ValueError("ordered child digest binding does not match occurrences")
    children = tuple(item.node for item in artifact.occurrences)
    root = LogicalClaimNode(
        node_id=artifact.reconstruction.root_node_id,
        connective=LogicalConnective(artifact.reconstruction.operator),
        children=children,
        source_span=artifact.reconstruction.root_source_span,
    )
    if _digest(root) != artifact.reconstruction.source_root_digest:
        raise ValueError("reconstructed root does not match the bound source root")
    return root


@dataclass(frozen=True, slots=True)
class ClaimDecompositionInstallation:
    semantics_uri: str
    structured_claim_schema_uri: str
    decomposition_schema_uri: str


@dataclass(frozen=True, slots=True)
class _Resources:
    store: ArtifactRepository
    artifacts: ArtifactService
    semantics_uri: str
    structured_claim_schema_uri: str
    decomposition_schema_uri: str


class ClaimDecompositionAdapter:
    def __init__(self, resources: _Resources, *, connective: LogicalConnective) -> None:
        self.resources = resources
        self.connective = connective
        capability_id = (
            "claim.conjunction.split"
            if connective is LogicalConnective.CONJUNCTION
            else "claim.implication.obligations"
        )
        self._descriptor = CapabilityDescriptor(
            capability_id=capability_id,
            version="1",
            title=(
                "Split one top-level structured conjunction"
                if connective is LogicalConnective.CONJUNCTION
                else "Expose one implication's ordered obligations"
            ),
            description=(
                "Decompose only the requested top-level connective while preserving "
                "the exact ordered subtrees and a deterministic reconstruction record."
            ),
            provider="jacobian.runtime",
            provider_runtime=known_provider_runtime(
                "jacobian.runtime", features=("claim", "structured-decomposition")
            ),
            input_schema=model_schema(ClaimDecompositionRequest),
            output_schema=model_schema(ClaimDecompositionOutput),
            tags=("claim", "decomposition", "deterministic"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = ClaimDecompositionRequest.model_validate(request.input)
            source = self.resources.store.get(validated.source_uri)
            if source.manifest.schema_uri != self.resources.structured_claim_schema_uri:
                raise ValueError("source is not a registered structured-claim artifact")
            if source.manifest.semantics_uri != self.resources.semantics_uri:
                raise ValueError("source does not use structured-claim semantics")
            normalized = self.resources.artifacts.schemas.validate(
                self.resources.structured_claim_schema_uri, source.payload
            )
            claim = StructuredClaimArtifact.model_validate(normalized)
        except (ValidationError, StorageError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_STRUCTURED_CLAIM",
                    stage="request_validation",
                    message="The source is not a valid registered structured claim.",
                    hint="Provide a stored v1 PROPOSITIONAL_STRUCTURE claim artifact.",
                )
            ) from exc
        if claim.root.connective is not self.connective:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="UNSUPPORTED_TOP_LEVEL_CONNECTIVE",
                    stage="claim_decomposition",
                    message=(
                        f"{self.descriptor.capability_id} does not accept top-level "
                        f"{claim.root.connective.value}."
                    ),
                    expected=self.connective.value,
                    actual_type=claim.root.connective.value,
                )
            )

        roles: tuple[
            Literal[
                "CONJUNCT",
                "ASSUME_ANTECEDENT",
                "PROVE_CONSEQUENT_UNDER_ANTECEDENT",
            ],
            ...,
        ]
        if self.connective is LogicalConnective.CONJUNCTION:
            roles = cast(
                tuple[
                    Literal[
                        "CONJUNCT",
                        "ASSUME_ANTECEDENT",
                        "PROVE_CONSEQUENT_UNDER_ANTECEDENT",
                    ],
                    ...,
                ],
                ("CONJUNCT",) * len(claim.root.children),
            )
        else:
            roles = (
                "ASSUME_ANTECEDENT",
                "PROVE_CONSEQUENT_UNDER_ANTECEDENT",
            )
        occurrences = tuple(
            DecomposedOccurrence(
                position=index,
                role=role,
                path=f"{claim.root.node_id}.children[{index}]",
                node=child,
                node_digest=_digest(child),
            )
            for index, (role, child) in enumerate(
                zip(roles, claim.root.children, strict=True)
            )
        )
        root_digest = _digest(claim.root)
        payload = ClaimDecompositionArtifact(
            capability_id=cast(
                Literal[
                    "claim.conjunction.split",
                    "claim.implication.obligations",
                ],
                self.descriptor.capability_id,
            ),
            source_binding=SourceArtifactBinding(
                source_uri=validated.source_uri,
                object_digest=source.manifest.object_digest,
                payload_digest=source.manifest.payload_digest,
                schema_uri=source.manifest.schema_uri,
                semantics_uri=source.manifest.semantics_uri,
                canonicalizer_digest=source.manifest.canonicalizer_digest,
            ),
            occurrences=occurrences,
            reconstruction=ReconstructionRecord(
                operator=cast(
                    Literal["CONJUNCTION", "IMPLICATION"],
                    self.connective.value,
                ),
                root_node_id=claim.root.node_id,
                root_source_span=claim.root.source_span,
                source_root_digest=root_digest,
                ordered_child_digests=tuple(item.node_digest for item in occurrences),
            ),
        )
        saved = self.resources.artifacts.put(
            schema_uri=self.resources.decomposition_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=payload.model_dump(mode="json"),
            parents=(validated.source_uri,),
            summary=f"{self.descriptor.capability_id} exact reconstruction record",
        )
        output = ClaimDecompositionOutput(
            **payload.model_dump(mode="python"),
            decomposition_uri=saved.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one exact top-level structured-claim decomposition",
                parameters={"top_level_connective": self.connective.value},
                artifact_uri=saved.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis="all immediate ordered children of the supported root were returned",
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="claim.reconstruction.proposed",
                    source_artifact_uris=(saved.artifact_uri,),
                    target_artifact_uris=(validated.source_uri,),
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="deterministic structural projection; no truth claim or proof",
            ),
            artifact_uris=(validated.source_uri, saved.artifact_uri),
        )


def install_claim_decomposition_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> tuple[
    tuple[ClaimDecompositionAdapter, ClaimDecompositionAdapter],
    ClaimDecompositionInstallation,
]:
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.structured-logical-claim",
        version="1",
        definition={
            "logic": "PROPOSITIONAL_STRUCTURE",
            "meaning": "ordered, explicitly grouped, opaque-atom logical syntax",
            "verification": "structural computation only; no truth or proof claim",
        },
    )
    claim_schema_uri = schemas.register_model(
        name="jacobian.structured-logical-claim",
        version="1",
        model=StructuredClaimArtifact,
    )
    decomposition_schema_uri = schemas.register_model(
        name="jacobian.claim-decomposition",
        version="1",
        model=ClaimDecompositionArtifact,
    )
    resources = _Resources(
        store=store,
        artifacts=artifacts,
        semantics_uri=semantics_uri,
        structured_claim_schema_uri=claim_schema_uri,
        decomposition_schema_uri=decomposition_schema_uri,
    )
    return (
        (
            ClaimDecompositionAdapter(
                resources, connective=LogicalConnective.CONJUNCTION
            ),
            ClaimDecompositionAdapter(
                resources, connective=LogicalConnective.IMPLICATION
            ),
        ),
        ClaimDecompositionInstallation(
            semantics_uri=semantics_uri,
            structured_claim_schema_uri=claim_schema_uri,
            decomposition_schema_uri=decomposition_schema_uri,
        ),
    )
