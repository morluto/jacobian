"""Monochromatic clique hypergraph constructor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from pydantic_core import PydanticCustomError

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.monochromatic_clique._models import (
    MAX_CLIQUE_ORDER,
    MAX_VERTICES,
    MonochromaticCliqueHypergraphResult,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


@dataclass(frozen=True, slots=True)
class MonochromaticCliqueAdmission:
    """Derived bounds shared by native and catalog execution."""

    clique_count: int
    incidence_count: int
    result_bytes: int
    cliques: tuple[tuple[str, ...], ...]


def _array_size(value_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(value_sizes) - 1, 0) + sum(value_sizes)


def _int_size(value: int) -> int:
    return len(encode_strict_json(value))


def _admit_monochromatic_clique_hypergraph(
    colored_graph: ColoredUndirectedGraph, clique_order: int
) -> MonochromaticCliqueAdmission:
    """Admit the complete finite construction before enumerating subsets."""

    if not isinstance(colored_graph, ColoredUndirectedGraph):
        raise OperationDomainValidationError(
            location=("colored_graph",),
            code="monochromatic_clique.invalid_graph",
            message="colored_graph must be a ColoredUndirectedGraph",
        )
    if type(clique_order) is not int or not 2 <= clique_order <= MAX_CLIQUE_ORDER:
        raise OperationDomainValidationError(
            location=("clique_order",),
            code="monochromatic_clique.invalid_clique_order",
            message=(
                f"clique_order must be an integer between 2 and {MAX_CLIQUE_ORDER}"
            ),
        )
    graph = colored_graph.graph
    vertices = graph.vertices
    vertex_count = len(vertices)
    if vertex_count > MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("colored_graph", "graph", "vertices"),
            code="monochromatic_clique.too_many_vertices",
            message=f"at most {MAX_VERTICES} vertices are supported",
        )
    if not colored_graph.edge_colors:
        raise OperationDomainValidationError(
            location=("colored_graph", "edge_colors"),
            code="monochromatic_clique.no_edge_colors",
            message="edge_colors must be provided (total colouring required)",
        )
    if len(colored_graph.edge_colors) != len(graph.edges):
        raise OperationDomainValidationError(
            location=("colored_graph", "edge_colors"),
            code="monochromatic_clique.edge_colors_not_aligned",
            message="edge_colors must align with every graph edge",
        )
    edge_set = set(graph.edges)
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            if (left, right) not in edge_set and (right, left) not in edge_set:
                raise OperationDomainValidationError(
                    location=("colored_graph", "graph", "edges"),
                    code="monochromatic_clique.graph_not_complete",
                    message="the underlying graph must be complete",
                )
    candidate_count = comb(vertex_count, clique_order)
    work = candidate_count * (clique_order * (clique_order - 1) // 2)
    if work > 2_000_000:
        raise OperationDomainValidationError(
            location=("clique_order",),
            code="monochromatic_clique.work_bound_exceeded",
            message="monochromatic clique enumeration exceeds the work bound",
        )
    edge_to_color = dict(
        zip(graph.edges, colored_graph.edge_colors, strict=True)
    )
    cliques = tuple(
        tuple(sorted(subset))
        for subset in combinations(vertices, clique_order)
        if len(
            {
                edge_to_color.get((left, right), edge_to_color.get((right, left)))
                for left, right in combinations(subset, 2)
            }
        )
        == 1
    )
    clique_count = len(cliques)
    incidence_count = clique_count * clique_order
    if clique_count > MAX_EDGES:
        raise OperationDomainValidationError(
            location=("clique_order",),
            code="monochromatic_clique.edge_bound_exceeded",
            message=f"the construction exceeds the {MAX_EDGES}-edge hypergraph bound",
        )
    if incidence_count > MAX_TOTAL_INCIDENCES:
        raise OperationDomainValidationError(
            location=("clique_order",),
            code="monochromatic_clique.incidence_bound_exceeded",
            message=(
                "the construction exceeds the "
                f"{MAX_TOTAL_INCIDENCES}-incidence hypergraph bound"
            ),
        )
    try:
        colored_graph_bytes = len(
            encode_strict_json(colored_graph.model_dump(mode="json"))
        )
        label_sizes = tuple(len(encode_strict_json(label)) for label in vertices)
        vertex_bytes = _array_size(label_sizes)
        member_bytes = _array_size(
            tuple(sorted(label_sizes, reverse=True)[:clique_order])
        )
        edge_id_bytes = len(encode_strict_json(f"clique_{clique_count - 1}"))
        edge_bytes = _array_size((edge_id_bytes, member_bytes))
        hypergraph_bytes = strict_json_object_size(
            (
                ("vertices", vertex_bytes),
                ("edges", _array_size((edge_bytes,) * clique_count)),
            )
        )
        result_bytes = strict_json_object_size(
            (
                ("colored_graph", colored_graph_bytes),
                ("clique_order", _int_size(clique_order)),
                ("hypergraph", hypergraph_bytes),
            )
        )
    except (ValueError, TypeError, PydanticCustomError) as error:
        raise OperationDomainValidationError(
            location=("colored_graph",),
            code="monochromatic_clique.source_not_canonical",
            message="colored_graph cannot be represented in canonical JSON",
        ) from error
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("colored_graph",),
            code="monochromatic_clique.result_bytes_exceeded",
            message="monochromatic clique hypergraph exceeds the canonical output-byte limit",
        )
    return MonochromaticCliqueAdmission(
        clique_count, incidence_count, result_bytes, cliques
    )


__all__ = ["construct_monochromatic_clique_hypergraph"]


def construct_monochromatic_clique_hypergraph(
    colored_graph: ColoredUndirectedGraph,
    clique_order: int,
) -> MonochromaticCliqueHypergraphResult:
    """Construct the t-uniform monochromatic-clique hypergraph.

    For each t-element vertex subset, check whether all C(t,2) edges
    share the same colour. If so, the subset is a monochromatic t-clique
    and becomes a hyperedge.
    """
    admission = _admit_monochromatic_clique_hypergraph(colored_graph, clique_order)
    hyper_edges = tuple(
        (f"clique_{index}", clique) for index, clique in enumerate(admission.cliques)
    )

    if len(hyper_edges) != admission.clique_count:
        raise AssertionError("kernel produced more clique edges than admitted")
    hypergraph = FiniteHypergraph(
        vertices=colored_graph.graph.vertices,
        edges=hyper_edges,
    )
    return MonochromaticCliqueHypergraphResult(
        colored_graph=colored_graph,
        clique_order=clique_order,
        hypergraph=hypergraph,
    )
