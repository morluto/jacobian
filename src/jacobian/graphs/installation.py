"""Installation and composition for the core graph capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.capability_adapters import CapabilityAdapter
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope
from jacobian.contracts.graph_degree_sequence import (
    GraphDegreeSequenceClaim,
    GraphDegreeSequenceResultArtifact,
)
from jacobian.contracts.graph_invariants import (
    GraphInvariantBatchArtifact,
    GraphInvariantResultArtifact,
    GraphNeighborhoodIndependenceArtifact,
    GraphNeighborhoodIndependenceClaim,
)
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.graphs.artifacts import GraphArtifactResources
from jacobian.graphs.atlas_search import (
    GraphAtlasSearchAdapter,
    GraphAtlasSearchResources,
)
from jacobian.graphs.construction import (
    GraphConstructionResources,
    GraphExplicitConstructionAdapter,
)
from jacobian.graphs.degree_sequence import (
    GraphDegreeSequenceAdapter,
    GraphDegreeSequenceResources,
)
from jacobian.graphs.invariants import (
    GraphInvariantResources,
    GraphPropertyAdapter,
)
from jacobian.graphs.neighborhood_independence import (
    GraphNeighborhoodIndependenceAdapter,
    GraphNeighborhoodIndependenceResources,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService
from jacobian.verification_capabilities import certificate_verification_adapter


@dataclass(frozen=True, slots=True)
class GraphInstallation:
    semantics_uri: str
    graph_schema_uri: str
    scope_schema_uri: str
    property_schema_uri: str
    invariant_result_schema_uri: str
    degree_sequence_claim_schema_uri: str
    degree_sequence_result_schema_uri: str
    neighborhood_schema_uri: str
    neighborhood_claim_schema_uri: str
    certificate_schema_uri: str
    degree_sequence_checker_id: str | None
    neighborhood_checker_id: str | None


def install_graph_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[tuple[CapabilityAdapter[Any], ...], GraphInstallation]:
    """Register graph artifact contracts and return the bundled adapters."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.simple-undirected-graph",
        version="1",
        definition={
            "domain": "finite simple undirected graphs",
            "vertices": "distinct string labels",
            "edges": "distinct two-vertex arrays in ascending label order",
            "atlas_scope": (
                "networkx.graph_atlas_g representatives with exactly the "
                "requested order, limited to orders zero through seven"
            ),
        },
    )
    graph_schema_uri = schemas.register_model(
        name="jacobian.simple-undirected-graph",
        version="1",
        model=SimpleUndirectedGraph,
    )
    scope_schema_uri = schemas.register(
        name="jacobian.graph-atlas-scope",
        version="1",
        schema={
            "type": "object",
            "properties": {
                "scope_schema_version": {"const": "1"},
                "source": {"const": "networkx.graph_atlas_g"},
                "backend_version": {"type": "string"},
                "order": {"type": "integer", "minimum": 0, "maximum": 7},
                "enumerated_count": {"type": "integer", "minimum": 0},
            },
            "required": [
                "scope_schema_version",
                "source",
                "backend_version",
                "order",
                "enumerated_count",
            ],
            "additionalProperties": False,
        },
    )
    property_schema_uri = schemas.register(
        name="jacobian.graph-property-batch",
        version="2",
        schema=model_schema(GraphInvariantBatchArtifact),
    )
    invariant_result_schema_uri = schemas.register(
        name="jacobian.graph-invariant-result",
        version="1",
        schema=model_schema(GraphInvariantResultArtifact),
    )
    degree_sequence_claim_schema_uri = schemas.register(
        name="jacobian.graph-degree-sequence-claim",
        version="1",
        schema=GraphDegreeSequenceClaim.model_json_schema(),
    )
    degree_sequence_result_schema_uri = schemas.register(
        name="jacobian.graph-degree-sequence-result",
        version="1",
        schema=GraphDegreeSequenceResultArtifact.model_json_schema(),
    )
    neighborhood_schema_uri = schemas.register(
        name="jacobian.graph-neighborhood-independence",
        version="1",
        schema=model_schema(GraphNeighborhoodIndependenceArtifact),
    )
    neighborhood_claim_schema_uri = schemas.register(
        name="jacobian.graph-neighborhood-independence-claim",
        version="1",
        schema=model_schema(GraphNeighborhoodIndependenceClaim),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    degree_sequence_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact simple-graph degree-sequence replay checker",
                entrypoint=(
                    "jacobian_checkers.graph_degree_sequence:check_degree_sequence"
                ),
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="graph.degree_sequence",
                format_version="1",
                claim_schema_uris=(degree_sequence_claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(degree_sequence_result_schema_uri,),
                reason="bundled independent degree-sequence checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    neighborhood_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact graph neighborhood-independence replay checker",
                entrypoint=(
                    "jacobian_checkers.graph_invariants:check_neighborhood_independence"
                ),
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="graph.neighborhood_independence",
                format_version="1",
                claim_schema_uris=(neighborhood_claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(neighborhood_schema_uri,),
                reason="bundled independent finite-graph invariant checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = GraphInstallation(
        semantics_uri=semantics_uri,
        graph_schema_uri=graph_schema_uri,
        scope_schema_uri=scope_schema_uri,
        property_schema_uri=property_schema_uri,
        invariant_result_schema_uri=invariant_result_schema_uri,
        degree_sequence_claim_schema_uri=degree_sequence_claim_schema_uri,
        degree_sequence_result_schema_uri=degree_sequence_result_schema_uri,
        neighborhood_schema_uri=neighborhood_schema_uri,
        neighborhood_claim_schema_uri=neighborhood_claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        degree_sequence_checker_id=degree_sequence_checker_id,
        neighborhood_checker_id=neighborhood_checker_id,
    )
    common_resources = GraphArtifactResources(
        store=store,
        artifacts=artifacts,
        semantics_uri=semantics_uri,
        graph_schema_uri=graph_schema_uri,
    )
    construction_resources = GraphConstructionResources(
        graph=common_resources,
    )
    atlas_resources = GraphAtlasSearchResources(
        graph=common_resources,
        scope_schema_uri=scope_schema_uri,
    )
    invariant_resources = GraphInvariantResources(
        graph=common_resources,
        property_schema_uri=property_schema_uri,
        invariant_result_schema_uri=invariant_result_schema_uri,
    )
    degree_sequence_resources = GraphDegreeSequenceResources(
        graph=common_resources,
        degree_sequence_claim_schema_uri=degree_sequence_claim_schema_uri,
        degree_sequence_result_schema_uri=degree_sequence_result_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        degree_sequence_checker_id=degree_sequence_checker_id,
    )
    neighborhood_resources = GraphNeighborhoodIndependenceResources(
        graph=common_resources,
        neighborhood_schema_uri=neighborhood_schema_uri,
        neighborhood_claim_schema_uri=neighborhood_claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        neighborhood_checker_id=neighborhood_checker_id,
    )
    adapters: tuple[CapabilityAdapter[Any], ...] = (
        GraphExplicitConstructionAdapter(construction_resources),
        GraphAtlasSearchAdapter(atlas_resources),
        GraphPropertyAdapter(invariant_resources),
        GraphDegreeSequenceAdapter(degree_sequence_resources),
        GraphNeighborhoodIndependenceAdapter(neighborhood_resources),
    )
    for adapter in (
        certificate_verification_adapter(
            capability_id="graph.degree_sequence.verify",
            title="Verify a graph degree-sequence realization",
            description=(
                "Independently replay one exact realization or Erdos-Gallai "
                "obstruction with the installed graph checker."
            ),
            checker_id=degree_sequence_checker_id,
            tags=("graph", "degree-sequence"),
            verification=verification,
        ),
        certificate_verification_adapter(
            capability_id="graph.neighborhood_independence.verify",
            title="Verify graph neighborhood independence values",
            description=(
                "Independently replay one exact neighborhood-independence ledger."
            ),
            checker_id=neighborhood_checker_id,
            tags=("graph", "neighborhood-independence"),
            verification=verification,
        ),
    ):
        if adapter is not None:
            adapters += (adapter,)
    return adapters, installation
