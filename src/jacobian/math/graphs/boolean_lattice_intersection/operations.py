"""Boolean-lattice intersection graph constructor."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.boolean_lattice_intersection._models import (
    BooleanLatticeIntersectionResult,
    IntersectionRelation,
    _validate_intersection_request,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_boolean_lattice_intersection_graph"]


@dataclass(frozen=True, slots=True)
class BooleanLatticeIntersectionAdmission:
    """The exact bounded graph construction shared by admission and execution."""

    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def _admit_boolean_lattice_intersection(
    ground_set_size: int,
    threshold: int,
    relation: IntersectionRelation,
) -> BooleanLatticeIntersectionAdmission:
    try:
        _validate_intersection_request(ground_set_size, threshold, relation)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error

    subsets = _all_subsets(ground_set_size)
    vertices = tuple(_subset_label(subset) for subset in subsets)
    edges: list[tuple[str, str]] = []
    for left_index, left in enumerate(subsets):
        left_label = vertices[left_index]
        for right in subsets[left_index + 1 :]:
            right_label = _subset_label(right)
            intersection_size = len(left & right)
            if relation == "INTERSECTION_EQ":
                adjacent = intersection_size == threshold
            elif relation == "INTERSECTION_LT":
                adjacent = intersection_size < threshold
            else:
                adjacent = intersection_size > threshold
            if adjacent:
                edges.append(
                    (min(left_label, right_label), max(left_label, right_label))
                )
    canonical_edges = tuple(edges)
    return BooleanLatticeIntersectionAdmission(vertices, canonical_edges)


def construct_boolean_lattice_intersection_graph(
    ground_set_size: int,
    threshold: int,
    relation: IntersectionRelation,
) -> BooleanLatticeIntersectionResult:
    """Construct a graph on the Boolean lattice 2^[n].

    Vertices are all subsets of [n], labelled by canonical subset strings.
    An edge joins two distinct subsets when their intersection size
    satisfies the declared relation with the threshold.
    """
    admission = _admit_boolean_lattice_intersection(
        ground_set_size, threshold, relation
    )
    graph = SimpleUndirectedGraph(
        vertices=admission.vertices,
        edges=admission.edges,
    )
    return BooleanLatticeIntersectionResult.model_construct(
        ground_set_size=ground_set_size,
        threshold=threshold,
        relation=relation,
        graph=graph,
    )


def _all_subsets(n: int) -> list[frozenset[int]]:
    if n == 0:
        return [frozenset()]
    result = []
    for mask in range(1 << n):
        result.append(frozenset(i for i in range(n) if mask & (1 << i)))
    return result


def _subset_label(subset: frozenset[int]) -> str:
    if not subset:
        return "{}"
    return "{" + ",".join(str(i) for i in sorted(subset)) + "}"
