"""Graph-owned adapter over the generic preservation-checked shrink engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.claims import ClaimSpec, CorrespondenceStatus, PredicateSpec
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.graph_shrinking import (
    GraphCounterexampleShrinkOutput,
    GraphCounterexampleShrinkRequest,
    GraphLocalMinimalityScope,
    GraphReduction,
    GraphReductionAttempt,
    GraphReductionOutcome,
    GraphShrinkTraceArtifact,
)
from jacobian.contracts.plugins import (
    CapabilityDescriptor as PluginCapabilityDescriptor,
)
from jacobian.contracts.plugins import CapabilityName, PluginManifest
from jacobian.contracts.results import ExecutionStatus, InputStatus, Verification
from jacobian.contracts.shrinking import ShrinkStep
from jacobian.graphs.installation import GraphInstallation
from jacobian.plugins.registry import PluginRegistry
from jacobian.provider_runtime import known_provider_runtime
from jacobian.references import ReferenceInstaller
from jacobian.registry import CheckerRegistry, CheckerRegistryError
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.shrinking import ShrinkService
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository

_DOMAIN_ID = "jacobian.graph-shrinking"
_PROPERTY_ID = "graph.property.non_bipartite"
_PRESERVATION_FORMAT = "graph.property.non_bipartite.preservation"


@dataclass(frozen=True, slots=True)
class GraphShrinkingInstallation:
    plugin_id: str
    claim_schema_uri: str
    trace_schema_uri: str
    property_checker_id: str | None


@dataclass(frozen=True, slots=True)
class _LoadedGraphArtifact:
    stored: StoredArtifact
    graph: SimpleUndirectedGraph


def install_graph_shrinking(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    plugins: PluginRegistry,
    checkers: CheckerRegistry,
    shrinking: ShrinkService,
    graph: GraphInstallation,
    reference_installer: ReferenceInstaller,
    *,
    authorize_checker: bool,
) -> tuple[GraphCounterexampleShrinkAdapter, GraphShrinkingInstallation]:
    """Install the graph reducer plugin, contracts, optional checker, and adapter."""

    claim_schema_uri = schemas.register(
        name="jacobian.graph-counterexample-property-claim",
        version="1",
        schema=model_schema(ClaimSpec),
    )
    trace_schema_uri = schemas.register(
        name="jacobian.graph-counterexample-shrink-trace",
        version="1",
        schema=model_schema(GraphShrinkTraceArtifact),
    )
    entrypoint = "jacobian.plugins.graph_shrinking:reduce_simple_graph"
    implementation_uri = plugins.register_implementation(entrypoint)
    manifest = artifacts.put(
        schema_uri=reference_installer.manifest_schema_uri,
        semantics_uri=reference_installer.manifest_semantics_uri,
        payload=PluginManifest(
            domain_id=_DOMAIN_ID,
            domain_version="1",
            semantics_uri=graph.semantics_uri,
            claim_schema_uri=claim_schema_uri,
            candidate_schema_uri=graph.graph_schema_uri,
            capabilities={
                CapabilityName.REDUCER: PluginCapabilityDescriptor(
                    implementation_uri=implementation_uri,
                    entrypoint=entrypoint,
                    version="1",
                )
            },
        ).model_dump(mode="json"),
        summary="untrusted simple-graph deletion reducer plugin",
    )
    plugins.install(manifest.artifact_uri)
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name=("exact non-bipartite graph property preservation checker"),
                entrypoint=(
                    "jacobian_checkers.graph_shrinking:check_non_bipartite_preservation"
                ),
                evidence_kind=EvidenceKind.PRESERVATION,
                format_id=_PRESERVATION_FORMAT,
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(graph.semantics_uri,),
                candidate_schema_uris=(graph.graph_schema_uri,),
                reason=("bundled independent finite simple-graph property checker"),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = GraphShrinkingInstallation(
        plugin_id=manifest.artifact_uri,
        claim_schema_uri=claim_schema_uri,
        trace_schema_uri=trace_schema_uri,
        property_checker_id=checker_id,
    )
    return (
        GraphCounterexampleShrinkAdapter(
            store=store,
            artifacts=artifacts,
            checkers=checkers,
            shrinking=shrinking,
            graph=graph,
            installation=installation,
        ),
        installation,
    )


class GraphCounterexampleShrinkAdapter:
    """Shrink one non-bipartite graph over an explicitly tested deletion scope."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        checkers: CheckerRegistry,
        shrinking: ShrinkService,
        graph: GraphInstallation,
        installation: GraphShrinkingInstallation,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.checkers = checkers
        self.shrinking = shrinking
        self.graph = graph
        self.installation = installation
        checker_ids = (
            (installation.property_checker_id,)
            if installation.property_checker_id is not None
            else ()
        )
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.counterexample.shrink",
            version="1",
            title="Shrink a simple-graph counterexample",
            description=(
                "Greedily test deterministic single vertex and edge deletions, "
                "independently checking every accepted non-bipartite reduction."
            ),
            provider="jacobian.graph-shrinking",
            provider_runtime=known_provider_runtime(
                "jacobian.graph-shrinking",
                features=("simple-undirected-graph", "single-deletion", "shrink.run"),
                checker_ids=checker_ids,
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(GraphCounterexampleShrinkRequest),
            output_schema=model_schema(GraphCounterexampleShrinkOutput),
            tags=("graph", "counterexample", "shrinking", "local-minimality"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = GraphCounterexampleShrinkRequest.model_validate(request.input)
        self._load_graph(validated.graph_uri)
        try:
            self.checkers.require_compatible(
                validated.property_checker_id,
                evidence_kind=EvidenceKind.PRESERVATION,
                format_id=_PRESERVATION_FORMAT,
                format_version="1",
                claim_schema_uri=self.installation.claim_schema_uri,
                semantics_uri=self.graph.semantics_uri,
                candidate_schema_uri=self.graph.graph_schema_uri,
            )
        except CheckerRegistryError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="GRAPH_PROPERTY_CHECKER_INVALID",
                    stage="checker_selection",
                    message=(
                        "The selected checker is not an active compatible graph "
                        "property preservation checker."
                    ),
                    hint=(
                        "Select an operator-registered checker advertised for "
                        "graph.property.non_bipartite."
                    ),
                )
            ) from exc
        claim = self.artifacts.put(
            schema_uri=self.installation.claim_schema_uri,
            semantics_uri=self.graph.semantics_uri,
            payload=ClaimSpec(
                domain_id=_DOMAIN_ID,
                domain_version="1",
                semantics_uri=self.graph.semantics_uri,
                predicate=PredicateSpec(name=validated.property_id),
                required_capabilities=(CapabilityName.REDUCER,),
                correspondence_status=CorrespondenceStatus.FORMALLY_LINKED,
            ).model_dump(mode="json"),
            parents=(validated.graph_uri,),
            summary="registered simple-graph counterexample property",
        )
        reducers = tuple(item.value for item in validated.reducers)
        shrunk = self.shrinking.run(
            target_kind="candidate",
            target_uri=validated.graph_uri,
            claim_uri=claim.artifact_uri,
            plugin_id=self.installation.plugin_id,
            preservation_checker_id=validated.property_checker_id,
            reducers=reducers,
            objectives=("vertices", "edges"),
            evaluation_budget=validated.evaluation_budget,
            reducer_timeout_seconds=validated.reducer_timeout_seconds,
            proposal_validator=_validate_exact_graph_reduction,
        )
        attempts = tuple(self._attempt(step) for step in shrunk.steps)
        final = self._load_graph(shrunk.final_target_uri)
        local_scope = _local_scope(
            graph=final,
            final_uri=shrunk.final_target_uri,
            reducers=validated.reducers,
            attempts=attempts,
            final_property_verified=(
                shrunk.result.assurance.verification is Verification.VERIFIED
            ),
            execution_status=shrunk.execution.status,
        )
        parent_uris = tuple(
            dict.fromkeys(
                (
                    validated.graph_uri,
                    shrunk.final_target_uri,
                    claim.artifact_uri,
                    *(
                        attempt.proposed_graph_uri
                        for attempt in attempts
                        if attempt.proposed_graph_uri is not None
                    ),
                    *(
                        attempt.verification_record_uri
                        for attempt in attempts
                        if attempt.verification_record_uri is not None
                    ),
                )
            )
        )
        trace = self.artifacts.put(
            schema_uri=self.installation.trace_schema_uri,
            semantics_uri=self.graph.semantics_uri,
            payload=GraphShrinkTraceArtifact(
                property_id=validated.property_id,
                property_checker_id=validated.property_checker_id,
                claim_uri=claim.artifact_uri,
                initial_graph_uri=validated.graph_uri,
                final_graph_uri=shrunk.final_target_uri,
                reducers=validated.reducers,
                evaluation_budget=validated.evaluation_budget,
                execution_status=shrunk.execution.status,
                attempts=attempts,
                local_minimality_scope=local_scope,
            ).model_dump(mode="json"),
            parents=parent_uris,
            summary="tested simple-graph counterexample shrinking trace",
        )
        output = GraphCounterexampleShrinkOutput(
            property_id=validated.property_id,
            property_checker_id=validated.property_checker_id,
            claim_uri=claim.artifact_uri,
            initial_graph_uri=validated.graph_uri,
            final_graph_uri=shrunk.final_target_uri,
            trace_uri=trace.artifact_uri,
            attempts=attempts,
            local_minimality_scope=local_scope,
        )
        completed = shrunk.execution.status is ExecutionStatus.COMPLETED
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=shrunk.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "only the recorded requested single deletions from each "
                    "encountered finite simple undirected graph"
                ),
                parameters={
                    "property_id": validated.property_id,
                    "reducers": list(reducers),
                    "evaluation_budget": validated.evaluation_budget,
                    "global_minimality": False,
                },
                artifact_uri=trace.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if local_scope.complete_for_requested_reducers
                    else CapabilityCompletenessStatus.PARTIAL
                ),
                basis=local_scope.basis,
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if completed
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "accepted reductions have independent verification records; "
                    "minimality is limited to the explicitly tested deletion scope"
                    if completed
                    else "shrinking did not complete; the incumbent and trace remain"
                ),
            ),
            artifact_uris=(*parent_uris, trace.artifact_uri),
        )

    def _load_graph_artifact(self, graph_uri: str) -> _LoadedGraphArtifact:
        try:
            artifact = self.store.get(graph_uri)
            if (
                artifact.manifest.schema_uri != self.graph.graph_schema_uri
                or artifact.manifest.semantics_uri != self.graph.semantics_uri
            ):
                raise ValueError
            graph = SimpleUndirectedGraph.model_validate(artifact.payload)
            if graph.vertices != tuple(sorted(graph.vertices)) or graph.edges != tuple(
                sorted(graph.edges)
            ):
                raise ValueError
            return _LoadedGraphArtifact(stored=artifact, graph=graph)
        except (StorageError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="GRAPH_SHRINK_INPUT_INVALID",
                    stage="input_validation",
                    message=(
                        "The graph artifact is unavailable or is not a canonical "
                        "finite simple undirected graph."
                    ),
                    hint="Use a graph URI produced by a Jacobian graph capability.",
                )
            ) from exc

    def _load_graph(self, graph_uri: str) -> SimpleUndirectedGraph:
        return self._load_graph_artifact(graph_uri).graph

    def _attempt(self, step: ShrinkStep) -> GraphReductionAttempt:
        deleted_vertex = None
        deleted_edge = None
        proposed_artifact = None
        if step.proposed_uri is not None:
            before = self._load_graph(step.from_uri)
            proposed_artifact = self._load_graph_artifact(step.proposed_uri)
            after = proposed_artifact.graph
            try:
                deleted_vertex, deleted_edge = _exact_graph_reduction(
                    GraphReduction(step.reducer),
                    before,
                    after,
                )
            except ValueError as exc:
                return _invalid_attempt(step, str(exc))
        if step.accepted:
            outcome = GraphReductionOutcome.ACCEPTED_VERIFIED
        elif step.input_status is InputStatus.REJECTED and step.proposed_uri is None:
            outcome = GraphReductionOutcome.INVALID_REDUCTION
        elif step.execution_status is not ExecutionStatus.COMPLETED:
            outcome = GraphReductionOutcome.CHECKER_ERROR
        elif (
            step.input_status is InputStatus.REJECTED and step.proposed_uri is not None
        ):
            outcome = GraphReductionOutcome.PROPERTY_REJECTED
        else:
            outcome = GraphReductionOutcome.INVALID_REDUCTION
        return GraphReductionAttempt(
            index=step.index,
            reducer=GraphReduction(step.reducer),
            from_graph_uri=step.from_uri,
            proposed_graph_uri=step.proposed_uri,
            deleted_vertex=deleted_vertex,
            deleted_edge=deleted_edge,
            outcome=outcome,
            verification_record_uri=step.verification_record_uri,
            detail=step.detail,
            candidate_digest=(
                proposed_artifact.stored.manifest.object_digest
                if proposed_artifact is not None
                else None
            ),
        )


def _validate_exact_graph_reduction(
    reducer: str,
    before: Any,
    after: Any,
) -> None:
    _exact_graph_reduction(
        GraphReduction(reducer),
        SimpleUndirectedGraph.model_validate(before),
        SimpleUndirectedGraph.model_validate(after),
    )


def _exact_graph_reduction(
    reducer: GraphReduction,
    before: SimpleUndirectedGraph,
    after: SimpleUndirectedGraph,
) -> tuple[str | None, tuple[str, str] | None]:
    if reducer is GraphReduction.DELETE_VERTEX:
        removed_vertices = sorted(set(before.vertices) - set(after.vertices))
        if len(removed_vertices) != 1:
            raise ValueError("proposal is not an exact single-vertex deletion")
        deleted_vertex = removed_vertices[0]
        expected = SimpleUndirectedGraph(
            vertices=tuple(
                vertex for vertex in before.vertices if vertex != deleted_vertex
            ),
            edges=tuple(edge for edge in before.edges if deleted_vertex not in edge),
        )
        if after != expected:
            raise ValueError("proposal is not an exact single-vertex deletion")
        return deleted_vertex, None

    if reducer is GraphReduction.DELETE_EDGE:
        removed_edges = sorted(set(before.edges) - set(after.edges))
        if len(removed_edges) != 1:
            raise ValueError("proposal is not an exact single-edge deletion")
        deleted_edge = removed_edges[0]
        expected = SimpleUndirectedGraph(
            vertices=before.vertices,
            edges=tuple(edge for edge in before.edges if edge != deleted_edge),
        )
        if after != expected:
            raise ValueError("proposal is not an exact single-edge deletion")
        return None, deleted_edge

    raise ValueError("proposal uses an unsupported graph reducer")


def _invalid_attempt(step: ShrinkStep, detail: str) -> GraphReductionAttempt:
    return GraphReductionAttempt(
        index=step.index,
        reducer=GraphReduction(step.reducer),
        from_graph_uri=step.from_uri,
        proposed_graph_uri=None,
        outcome=GraphReductionOutcome.INVALID_REDUCTION,
        detail=detail,
    )


def _local_scope(
    *,
    graph: SimpleUndirectedGraph,
    final_uri: str,
    reducers: tuple[GraphReduction, ...],
    attempts: tuple[GraphReductionAttempt, ...],
    final_property_verified: bool,
    execution_status: ExecutionStatus,
) -> GraphLocalMinimalityScope:
    expected_vertices = (
        graph.vertices if GraphReduction.DELETE_VERTEX in reducers else ()
    )
    expected_edges = graph.edges if GraphReduction.DELETE_EDGE in reducers else ()
    final_attempts = tuple(
        attempt for attempt in attempts if attempt.from_graph_uri == final_uri
    )
    tested_vertices = tuple(
        sorted(
            attempt.deleted_vertex
            for attempt in final_attempts
            if attempt.deleted_vertex is not None
        )
    )
    tested_edges = tuple(
        sorted(
            attempt.deleted_edge
            for attempt in final_attempts
            if attempt.deleted_edge is not None
        )
    )
    untested_vertices = tuple(sorted(set(expected_vertices) - set(tested_vertices)))
    untested_edges = tuple(sorted(set(expected_edges) - set(tested_edges)))
    all_rejected = all(
        attempt.outcome is GraphReductionOutcome.PROPERTY_REJECTED
        for attempt in final_attempts
    )
    complete = (
        execution_status is ExecutionStatus.COMPLETED
        and final_property_verified
        and not untested_vertices
        and not untested_edges
        and len(final_attempts) == len(expected_vertices) + len(expected_edges)
        and all_rejected
    )
    expected_attempt_count = len(expected_vertices) + len(expected_edges)
    completeness_status: Literal["COMPLETE", "INCOMPLETE", "UNKNOWN"] = (
        "COMPLETE"
        if complete
        else (
            "INCOMPLETE"
            if untested_vertices
            or untested_edges
            or len(final_attempts) != expected_attempt_count
            else "UNKNOWN"
        )
    )
    obligations = (
        "attempt every supported immediate vertex deletion"
        if untested_vertices
        else "",
        "attempt every supported immediate edge deletion" if untested_edges else "",
        "obtain acceptable rejection evidence for every attempted deletion"
        if not all_rejected
        else "",
        "complete the final property check before claiming local minimality"
        if not final_property_verified
        else "",
    )
    remaining_obligations: tuple[str, ...] = tuple(item for item in obligations if item)
    return GraphLocalMinimalityScope(
        requested_reducers=reducers,
        tested_vertex_deletions=tested_vertices,
        tested_edge_deletions=tested_edges,
        untested_vertex_deletions=untested_vertices,
        untested_edge_deletions=untested_edges,
        complete_for_requested_reducers=complete,
        one_step_locally_minimal=complete,
        expected_attempt_count=expected_attempt_count,
        completed_attempt_count=len(final_attempts),
        completeness_status=completeness_status,
        remaining_obligations=remaining_obligations,
        basis=(
            "every requested single deletion from the final graph was tested "
            "exactly once and mathematically rejected by the registered checker"
            if complete
            else "only the listed single deletions were tested; timeout, budget, "
            "checker, or preservation status left the local boundary incomplete"
        ),
    )
