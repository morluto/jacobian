"""Shared resources and operator-only assembly for graph operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.checker_authorization import authorize_checker_operation
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
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService
from jacobian.verification_operations import certificate_verification_adapter


@dataclass(frozen=True, slots=True)
class GraphOperationResources:
    """Schemas, semantics, and checker bindings shared by graph operations."""

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


def register_graph_resources(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
) -> GraphOperationResources:
    """Register passive graph contracts without constructing checker manifests."""

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
    return GraphOperationResources(
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
        degree_sequence_checker_id=None,
        neighborhood_checker_id=None,
    )


def bind_selected_graph_operation(
    operation_id: str,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> OperationAdapter[Any] | None:
    """Bind one ordinary graph operation without checker or portfolio setup."""

    if operation_id not in {
        "graph.construct.explicit",
        "graph.search.atlas",
        "graph.compute.properties",
        "graph.construct.compose",
        "graph.enumerate.nonisomorphic",
        "graph.realize.degree_sequence",
        "graph.compute.neighborhood_independence",
        "graph.degree_sequence.verify",
        "graph.neighborhood_independence.verify",
        "graph.isomorphism.verify",
    }:
        return None
    resources = register_graph_resources(store, schemas)
    if operation_id == "graph.isomorphism.verify":
        from jacobian.graphs.isomorphism import bind_selected_graph_isomorphism

        return bind_selected_graph_isomorphism(
            store,
            schemas,
            artifacts,
            verification,
            checkers,
            resources,
            catalog,
        )
    if operation_id in {
        "graph.construct.compose",
        "graph.enumerate.nonisomorphic",
    }:
        from jacobian.graphs.composition import build_graph_composition_operations

        composition_adapters = build_graph_composition_operations(
            store,
            schemas,
            artifacts,
            semantics_uri=resources.semantics_uri,
            graph_schema_uri=resources.graph_schema_uri,
        )
        return next(
            adapter
            for adapter in composition_adapters
            if adapter.descriptor.operation_id == operation_id
        )
    if operation_id in {
        "graph.realize.degree_sequence",
        "graph.degree_sequence.verify",
    }:
        resources = replace(
            resources,
            degree_sequence_checker_id=_active_checker_id(
                "graph.degree_sequence.verify", catalog, checkers
            ),
        )
    if operation_id in {
        "graph.compute.neighborhood_independence",
        "graph.neighborhood_independence.verify",
    }:
        resources = replace(
            resources,
            neighborhood_checker_id=_active_checker_id(
                "graph.neighborhood_independence.verify", catalog, checkers
            ),
        )
    if operation_id == "graph.degree_sequence.verify":
        return certificate_verification_adapter(
            operation_id=operation_id,
            title="Verify a graph degree-sequence realization",
            description=(
                "Independently replay one exact realization or Erdos-Gallai "
                "obstruction with the installed graph checker."
            ),
            checker_id=resources.degree_sequence_checker_id,
            tags=("graph", "degree-sequence"),
            verification=verification,
        )
    if operation_id == "graph.neighborhood_independence.verify":
        return certificate_verification_adapter(
            operation_id=operation_id,
            title="Verify graph neighborhood independence values",
            description=(
                "Independently replay one exact neighborhood-independence ledger."
            ),
            checker_id=resources.neighborhood_checker_id,
            tags=("graph", "neighborhood-independence"),
            verification=verification,
        )
    adapters = _graph_operation_adapters(store, artifacts, resources)
    return next(
        adapter
        for adapter in adapters
        if adapter.descriptor.operation_id == operation_id
    )


def _active_checker_id(
    operation_id: str,
    catalog: OperationCatalog,
    checkers: CheckerRegistry,
) -> str:
    binding = catalog.checker_binding(operation_id)
    if binding is None:
        raise OperationCatalogError(
            f"checker binding is missing; run `jacobian update`: {operation_id}"
        )
    checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    return binding.checker_id


def build_graph_operations(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[tuple[OperationAdapter[Any], ...], GraphOperationResources]:
    """Register graph artifact contracts and return the bundled adapters."""

    resources = register_graph_resources(store, schemas)
    semantics_uri = resources.semantics_uri
    graph_schema_uri = resources.graph_schema_uri
    scope_schema_uri = resources.scope_schema_uri
    property_schema_uri = resources.property_schema_uri
    invariant_result_schema_uri = resources.invariant_result_schema_uri
    degree_sequence_claim_schema_uri = resources.degree_sequence_claim_schema_uri
    degree_sequence_result_schema_uri = resources.degree_sequence_result_schema_uri
    neighborhood_schema_uri = resources.neighborhood_schema_uri
    neighborhood_claim_schema_uri = resources.neighborhood_claim_schema_uri
    certificate_schema_uri = resources.certificate_schema_uri
    degree_sequence_checker_id = authorize_checker_operation(
        checkers,
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
    ).checker_id
    neighborhood_checker_id = authorize_checker_operation(
        checkers,
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
    ).checker_id
    resources = GraphOperationResources(
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
    adapters = _graph_operation_adapters(store, artifacts, resources)
    for adapter in (
        certificate_verification_adapter(
            operation_id="graph.degree_sequence.verify",
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
            operation_id="graph.neighborhood_independence.verify",
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
    return adapters, resources


def _graph_operation_adapters(
    store: ArtifactRepository,
    artifacts: ArtifactService,
    resources: GraphOperationResources,
) -> tuple[OperationAdapter[Any], ...]:
    common_resources = GraphArtifactResources(
        store=store,
        artifacts=artifacts,
        semantics_uri=resources.semantics_uri,
        graph_schema_uri=resources.graph_schema_uri,
    )
    construction_resources = GraphConstructionResources(
        graph=common_resources,
    )
    atlas_resources = GraphAtlasSearchResources(
        graph=common_resources,
        scope_schema_uri=resources.scope_schema_uri,
    )
    invariant_resources = GraphInvariantResources(
        graph=common_resources,
        property_schema_uri=resources.property_schema_uri,
        invariant_result_schema_uri=resources.invariant_result_schema_uri,
    )
    degree_sequence_resources = GraphDegreeSequenceResources(
        graph=common_resources,
        degree_sequence_claim_schema_uri=resources.degree_sequence_claim_schema_uri,
        degree_sequence_result_schema_uri=resources.degree_sequence_result_schema_uri,
        certificate_schema_uri=resources.certificate_schema_uri,
        degree_sequence_checker_id=resources.degree_sequence_checker_id,
    )
    neighborhood_resources = GraphNeighborhoodIndependenceResources(
        graph=common_resources,
        neighborhood_schema_uri=resources.neighborhood_schema_uri,
        neighborhood_claim_schema_uri=resources.neighborhood_claim_schema_uri,
        certificate_schema_uri=resources.certificate_schema_uri,
        neighborhood_checker_id=resources.neighborhood_checker_id,
    )
    return (
        GraphExplicitConstructionAdapter(construction_resources),
        GraphAtlasSearchAdapter(atlas_resources),
        GraphPropertyAdapter(invariant_resources),
        GraphDegreeSequenceAdapter(degree_sequence_resources),
        GraphNeighborhoodIndependenceAdapter(neighborhood_resources),
    )
