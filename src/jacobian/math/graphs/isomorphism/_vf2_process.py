"""Killable VF2 process boundary for exact graph isomorphism."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_core import PydanticCustomError

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
    the host process.  A stopped or malformed worker has no mathematical
    conclusion and is represented by ``None`` only through the separate
    ``UNKNOWN`` outcome in :func:`decide_graph_isomorphism`.
    """
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
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
            completed = run_bounded_process(
                [sys.executable, str(_VF2_WORKER)],
                input_bytes=json.dumps(request, separators=(",", ":")).encode("utf-8"),
                timeout_seconds=_VF2_WALL_SECONDS,
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
        raise RuntimeError("bounded VF2 worker could not be started") from exc
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded VF2 worker did not establish an outcome")
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
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
        raise RuntimeError("bounded VF2 worker returned malformed output") from exc
    if len(pairs) != graph_a.vertex_count:
        raise ValueError("worker mapping is not complete")
    sources = {source for source, _ in pairs}
    targets = {target for _, target in pairs}
    if sources != set(range(graph_a.vertex_count)) or targets != set(
        range(graph_b.vertex_count)
    ):
        raise ValueError("worker mapping is not bijective")
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
        raise ValueError("worker mapping does not preserve adjacency")
    return [VertexMappingPair(from_vertex=src, to_vertex=dst) for src, dst in pairs]


def decide_graph_isomorphism(
    request: GraphIsomorphismRequest,
) -> GraphIsomorphismResult:
    """Decide whether two simple graphs are isomorphic."""
    _admit_graph_isomorphism(request)
    try:
        mapping = _vertex_mapping(request.graph_a, request.graph_b)
    except RuntimeError:
        return GraphIsomorphismResult(status="UNKNOWN", vertex_mapping=())
    if mapping is None:
        return GraphIsomorphismResult(status="NOT_ISOMORPHIC", vertex_mapping=())
    return GraphIsomorphismResult(
        status="ISOMORPHIC",
        vertex_mapping=tuple(mapping),
    )


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
