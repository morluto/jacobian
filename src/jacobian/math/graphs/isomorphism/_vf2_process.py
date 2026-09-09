"""Killable VF2 process boundary for exact graph isomorphism."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    lease_operation_phases,
    request_checkpoint,
    require_execution_deadline,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationRequest,
    ColoredGraphCanonicalizationResult,
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
    SimpleGraph,
    VertexMappingPair,
)
from jacobian.math.graphs.isomorphism.operations import _canonicalize_colored_graph

_VF2_WORKER = Path(__file__).resolve().with_name("_vf2_worker.py")
_VF2_WALL_SECONDS = 60.0
_VF2_STDOUT_LIMIT = 256 * 1024
_VF2_STDERR_LIMIT = 64 * 1024
_VF2_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_VF2_FILE_SIZE_BYTES = 1024 * 1024


def _admit_graph_isomorphism(request: GraphIsomorphismRequest) -> None:
    """Admit the cross-graph domain required by the VF2 kernel."""

    if request.graph_a.directed != request.graph_b.directed:
        raise OperationDomainValidationError(
            location=("graph_a", "directed"),
            code="graph.both_graphs_must_have_the_same_directedness",
            message="both graphs must have the same directedness",
        )
    if request.graph_a.vertex_count != request.graph_b.vertex_count:
        raise OperationDomainValidationError(
            location=("graph_a", "vertex_count"),
            code="graph.both_graphs_must_have_the_same_vertex_count",
            message="both graphs must have the same vertex count",
        )


def _vertex_mapping(
    graph_a: SimpleGraph,
    graph_b: SimpleGraph,
) -> list[VertexMappingPair] | None:
    """Return a worker-derived witness, or ``None`` when absent.

    VF2 is deliberately isolated because its search cannot be interrupted in
    the host process. A stopped or malformed worker raises an operational
    exception; only a completed negative decision returns ``None``.
    """
    from jacobian.process import (
        ProcessResourceLimits,
        check_bounded_process_result,
        decode_checked_worker_output,
        run_bounded_process,
        worker_environment,
    )

    request_checkpoint("before graph isomorphism")
    lease = lease_operation_phases(
        _VF2_WALL_SECONDS,
        admitted_response_bytes=_VF2_STDOUT_LIMIT,
        validation_work=graph_a.vertex_count + len(graph_a.edges) + len(graph_b.edges),
    )
    request = {
        "graph_a": {
            "vertex_count": graph_a.vertex_count,
            "directed": graph_a.directed,
            "edges": graph_a.edges,
        },
        "graph_b": {
            "vertex_count": graph_b.vertex_count,
            "directed": graph_b.directed,
            "edges": graph_b.edges,
        },
    }
    try:
        with TemporaryDirectory(prefix="jacobian-vf2-") as worker_directory:
            remaining_seconds = lease.backend_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError(
                    "graph isomorphism backend lease expired",
                    configured_seconds=_VF2_WALL_SECONDS,
                )
            completed = run_bounded_process(
                [sys.executable, str(_VF2_WORKER)],
                input_bytes=json.dumps(request, separators=(",", ":")).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_VF2_STDOUT_LIMIT,
                stderr_limit=_VF2_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_VF2_WALL_SECONDS),
                    address_space_bytes=_VF2_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_VF2_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    request_checkpoint("after graph isomorphism worker")
    check_bounded_process_result(completed)
    try:
        response = decode_checked_worker_output(
            completed.stdout,
            decode_result=lambda value: value,
            checkpoint=lambda: require_execution_deadline(lease.operation_deadline),
        )
        request_checkpoint("during graph isomorphism response validation")
        mapping = response["mapping"] if response["ok"] is True else None
        if mapping is None:
            if response.get("ok") is True:
                return None
            raise ValueError("worker reported a failure")
        if not isinstance(mapping, list):
            raise ValueError("worker mapping is malformed")
        pairs = [(int(source), int(target)) for source, target in mapping]
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc
    if len(pairs) != graph_a.vertex_count:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    sources = {source for source, _ in pairs}
    targets = {target for _, target in pairs}
    if sources != set(range(graph_a.vertex_count)) or targets != set(
        range(graph_b.vertex_count)
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    forward = dict(pairs)
    edges_b = {
        edge if graph_b.directed else tuple(sorted(edge)) for edge in graph_b.edges
    }
    if {
        (forward[source], forward[target])
        if graph_a.directed
        else tuple(sorted((forward[source], forward[target])))
        for source, target in graph_a.edges
    } != edges_b:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    request_checkpoint("after graph isomorphism response validation")
    return [VertexMappingPair(from_vertex=src, to_vertex=dst) for src, dst in pairs]


def decide_graph_isomorphism(
    request: GraphIsomorphismRequest,
) -> GraphIsomorphismResult:
    """Decide whether two simple graphs are isomorphic."""
    _admit_graph_isomorphism(request)
    mapping = _vertex_mapping(request.graph_a, request.graph_b)
    if mapping is None:
        return GraphIsomorphismResult(
            graph_a=request.graph_a,
            graph_b=request.graph_b,
            status="NOT_ISOMORPHIC",
            vertex_mapping=(),
        )
    return GraphIsomorphismResult(
        graph_a=request.graph_a,
        graph_b=request.graph_b,
        status="ISOMORPHIC",
        vertex_mapping=tuple(mapping),
    )


def verify_graph_isomorphism(claim: GraphIsomorphismResult) -> bool:
    """Check an ISOMORPHIC claim's bijection and adjacency preservation.

    Only ISOMORPHIC claims are verifiable. A NOT_ISOMORPHIC outcome has no
    checkable certificate in this carrier, so it does not verify and
    remains a producer outcome. No VF2 search runs here.
    """
    try:
        _admit_graph_isomorphism(
            GraphIsomorphismRequest(graph_a=claim.graph_a, graph_b=claim.graph_b)
        )
    except OperationDomainValidationError:
        return False
    if claim.status != "ISOMORPHIC":
        return False
    mapping = tuple((item.from_vertex, item.to_vertex) for item in claim.vertex_mapping)
    if len(mapping) != claim.graph_a.vertex_count:
        return False
    if {source for source, _ in mapping} != set(range(claim.graph_a.vertex_count)):
        return False
    if {target for _, target in mapping} != set(range(claim.graph_b.vertex_count)):
        return False
    forward = dict(mapping)
    mapped_edges = {
        (forward[source], forward[target])
        if claim.graph_a.directed
        else tuple(sorted((forward[source], forward[target])))
        for source, target in claim.graph_a.edges
    }
    target_edges = {
        edge if claim.graph_b.directed else tuple(sorted(edge))
        for edge in claim.graph_b.edges
    }
    return mapped_edges == target_edges


def compute_colored_graph_canonicalization(
    request: ColoredGraphCanonicalizationRequest,
) -> ColoredGraphCanonicalizationResult:
    """Return the exact canonical form for one admitted request.

    ``math.run`` parses and admits the typed request before calling this thin
    adapter. Native callers use the same typed kernel after owner-local
    admission without constructing a wire request.
    """

    try:
        return _canonicalize_colored_graph(request.colored_graph)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("colored_graph",),
            code=error.type,
            message=str(error),
        ) from error
