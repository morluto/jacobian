"""Domain functions for algebraic topology operations."""

from __future__ import annotations

from collections import deque
from typing import Literal, cast

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.operations import smith_normal_form_result
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
    run_topology_admission,
)
from jacobian.math.topology.edge_paths._models import (
    MAX_EDGES,
    MAX_PRESENTATION_GENERATORS,
    MAX_PRESENTATION_RELATOR_LETTERS,
    MAX_PRESENTATION_RELATORS,
    MAX_WORD,
    AbelianizationResult,
    EdgeGraph,
    EdgePathConcatenateResult,
    EdgePathWordResult,
    EdgeWordEntry,
    FiniteGroupPresentation,
    FiniteGroupWord,
    FreeReductionRequest,
    FreeReductionResult,
    FundamentalGroupPresentationResult,
    OrientedEdge,
    PresentationAbelianizationResult,
    TriangleRelator,
    WordLetter,
)


def _reject(*, location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"topology.edge_path.{code}",
        message=message,
    )


def _admit_edge_path_word(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
    start_vertex: int,
    path: tuple[OrientedEdge, ...],
) -> None:
    if len(edges) > MAX_EDGES or len(path) > MAX_WORD:
        raise OperationResourceAdmissionError(
            location=("path",),
            code="topology.edge_path.source_budget",
            message="edge path exceeds the source edge or word bound",
        )
    for u, v in edges:
        if not (0 <= u < vertex_count and 0 <= v < vertex_count):
            _reject(
                location=("edges",),
                code="edge_vertex_range",
                message="edge vertices must be in 0..vertex_count-1",
            )
    if not 0 <= start_vertex < vertex_count:
        _reject(
            location=("start_vertex",),
            code="start_vertex_range",
            message="start vertex must be in 0..vertex_count-1",
        )
    current = start_vertex
    for step in path:
        if step.edge_index >= len(edges):
            _reject(
                location=("path",),
                code="edge_index_range",
                message="path edge index is outside the graph",
            )
        left, right = edges[step.edge_index]
        source, target = (left, right) if step.orientation == 1 else (right, left)
        if source != current:
            _reject(
                location=("path",),
                code="path_continuity",
                message="oriented edge path is not continuous",
            )
        current = target


def _admit_edge_path_concatenation(
    vertex_count: int,
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
) -> None:
    if not path_a or not path_b:
        _reject(
            location=("path_a", "path_b"),
            code="missing_endpoint",
            message="each path must retain at least its endpoint vertex",
        )
    if len(path_a) > MAX_WORD or len(path_b) > MAX_WORD:
        raise OperationResourceAdmissionError(
            location=("path_a", "path_b"),
            code="topology.edge_path.source_budget",
            message="edge path exceeds the source word bound",
        )
    if any(not 0 <= v < vertex_count for v in path_a):
        _reject(
            location=("path_a",),
            code="vertex_range",
            message="path_a vertices must be valid",
        )
    if any(not 0 <= v < vertex_count for v in path_b):
        _reject(
            location=("path_b",),
            code="vertex_range",
            message="path_b vertices must be valid",
        )
    if path_a[-1] != path_b[0]:
        _reject(
            location=("path_b",),
            code="concatenation_endpoint",
            message="concatenated paths must share their endpoint",
        )


def edge_path_word(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
    start_vertex: int,
    path: tuple[OrientedEdge, ...],
) -> EdgePathWordResult:
    """Compute the free group word for an edge path.

    Each edge in the graph is assigned a generator label e_i.
    Traversing edge i forward adds e_i, backward adds e_i^{-1}.
    """
    _admit_edge_path_word(vertex_count, edges, start_vertex, path)
    word = [
        f"e{step.edge_index + 1}" + ("" if step.orientation == 1 else "^-1")
        for step in path
    ]
    return EdgePathWordResult(
        graph=EdgeGraph.from_request(vertex_count, edges),
        start_vertex=start_vertex,
        path=path,
        word=tuple(word),
        length=len(word),
    )


def concatenate_edge_paths(
    vertex_count: int,
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
) -> EdgePathConcatenateResult:
    """Concatenate two edge paths.

    If the last vertex of path_a equals the first vertex of path_b,
    the concatenation is path_a + path_b[1:], removing the duplicate.
    """
    _admit_edge_path_concatenation(vertex_count, path_a, path_b)
    result = path_a + path_b[1:]
    return EdgePathConcatenateResult(
        vertex_count=vertex_count,
        path_a=path_a,
        path_b=path_b,
        path=tuple(result),
        length=len(result),
    )


def verify_edge_path_word(claim: EdgePathWordResult) -> bool:
    """Verify generator encoding against the retained graph and source path."""
    try:
        return (
            edge_path_word(
                claim.graph.vertex_count,
                claim.graph.edges,
                claim.start_vertex,
                claim.path,
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_edge_path_concatenation(claim: EdgePathConcatenateResult) -> bool:
    """Verify concatenation against the retained vertex axis and input paths."""
    try:
        return (
            concatenate_edge_paths(claim.vertex_count, claim.path_a, claim.path_b)
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def _free_reduce(
    letters: list[tuple[int, int]],
) -> tuple[WordLetter, ...]:
    """Freely reduce signed generator occurrences with one stack pass."""

    stack: list[WordLetter] = []
    for generator, exponent in letters:
        if (
            stack
            and stack[-1].generator == generator
            and (stack[-1].exponent == -exponent)
        ):
            stack.pop()
        else:
            stack.append(
                WordLetter(
                    generator=generator,
                    exponent=cast("Literal[-1, 1]", exponent),
                )
            )
    return tuple(stack)


def free_reduce(request: FreeReductionRequest) -> FreeReductionResult:
    """Return the unique freely reduced word on the supplied generator axis."""
    return FreeReductionResult(
        generator_count=request.generator_count,
        word=FiniteGroupWord(
            letters=_free_reduce(
                [(letter.generator, letter.exponent) for letter in request.letters]
            )
        ),
    )


def _admit_fundamental_complex(
    complex_: FiniteSimplicialComplex,
    base_vertex: str,
) -> None:
    require_canonical_complex_admission(complex_)
    if base_vertex not in set(complex_.vertices):
        _reject(
            location=("base_vertex",),
            code="base_vertex_missing",
            message="the base vertex must be a vertex of the source complex",
        )


def _one_skeleton_edges(
    complex_: FiniteSimplicialComplex,
) -> tuple[tuple[str, str], ...]:
    if complex_.dimension < 1:
        return ()
    return tuple((face[0], face[1]) for face in complex_.faces_by_dimension[1].faces)


def _two_skeleton_triangles(
    complex_: FiniteSimplicialComplex,
) -> tuple[tuple[str, str, str], ...]:
    if complex_.dimension < 2:
        return ()
    return tuple(
        (face[0], face[1], face[2]) for face in complex_.faces_by_dimension[2].faces
    )


def _component_and_tree(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    base_vertex: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Lexicographic breadth-first spanning tree of the base 1-component."""

    adjacency: dict[str, list[str]] = {vertex: [] for vertex in vertices}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbours in adjacency.values():
        neighbours.sort()
    visited: set[str] = {base_vertex}
    queue: deque[str] = deque([base_vertex])
    tree: list[tuple[str, str]] = []
    while queue:
        vertex = queue.popleft()
        for neighbour in adjacency[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                tree.append(
                    (vertex, neighbour) if vertex < neighbour else (neighbour, vertex)
                )
                queue.append(neighbour)
    return tuple(sorted(visited)), tuple(sorted(tree))


def _oriented_edge_word(
    traversal_start: str,
    traversal_end: str,
    tree_edges: frozenset[tuple[str, str]],
    generator_for_edge: dict[tuple[str, str], int],
) -> list[tuple[int, int]]:
    edge = (
        (traversal_start, traversal_end)
        if traversal_start < traversal_end
        else (traversal_end, traversal_start)
    )
    if edge in tree_edges:
        return []
    generator = generator_for_edge[edge]
    sign = 1 if traversal_start == edge[0] else -1
    return [(generator, sign)]


def _abelianization(
    generators: tuple[str, ...],
    relators: tuple[FiniteGroupWord, ...],
) -> AbelianizationResult:
    rows: list[tuple[int, ...]] = []
    for relator in relators:
        row = [0] * len(generators)
        for letter in relator.letters:
            row[letter.generator] += letter.exponent
        rows.append(tuple(row))
    relation_matrix = IntegerMatrix(
        entries=tuple(rows),
        row_count=len(rows),
        column_count=len(generators),
    )
    smith = smith_normal_form_result(relation_matrix)
    invariant_factors = tuple(int(factor) for factor in smith.invariant_factors)
    return AbelianizationResult(
        relation_matrix=relation_matrix,
        rank=smith.rank,
        free_rank=len(generators) - smith.rank,
        torsion_invariant_factors=tuple(
            factor for factor in invariant_factors if factor > 1
        ),
    )


def presentation_abelianization(
    presentation: FiniteGroupPresentation,
) -> PresentationAbelianizationResult:
    """Compute the exact integer abelianization of a finite presentation."""
    total_letters = sum(len(relator.letters) for relator in presentation.relators)
    maximum_letters = MAX_PRESENTATION_RELATORS * MAX_WORD
    if total_letters > maximum_letters:
        raise OperationResourceAdmissionError(
            location=("presentation",),
            code="topology.fundamental_group.relator_length_budget",
            message=f"presentation relators exceed the {maximum_letters}-letter bound",
        )
    return PresentationAbelianizationResult(
        presentation=presentation,
        abelianization=_abelianization(presentation.generators, presentation.relators),
    )


def _fundamental_group_kernel(
    complex_value: FiniteSimplicialComplex,
    base_vertex: str,
) -> FundamentalGroupPresentationResult:
    _admit_fundamental_complex(complex_value, base_vertex)
    edges = _one_skeleton_edges(complex_value)
    component_vertices, tree_edges = _component_and_tree(
        complex_value.vertices, edges, base_vertex
    )
    component_set = set(component_vertices)
    component_edges = tuple(
        sorted(
            edge
            for edge in edges
            if edge[0] in component_set and edge[1] in component_set
        )
    )
    tree_edge_set = frozenset(tree_edges)
    non_tree_edges = tuple(
        edge for edge in component_edges if edge not in tree_edge_set
    )
    triangles = tuple(
        simplex
        for simplex in _two_skeleton_triangles(complex_value)
        if simplex[0] in component_set
        and simplex[1] in component_set
        and simplex[2] in component_set
    )
    if len(non_tree_edges) > MAX_PRESENTATION_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.fundamental_group.generator_budget",
            message=(
                "the basepoint component needs "
                f"{len(non_tree_edges)} generators, above the "
                f"{MAX_PRESENTATION_GENERATORS}-generator presentation bound"
            ),
        )
    if len(triangles) > MAX_PRESENTATION_RELATORS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.fundamental_group.relator_budget",
            message=(
                "the basepoint component has "
                f"{len(triangles)} two-simplices, above the "
                f"{MAX_PRESENTATION_RELATORS}-relator presentation bound"
            ),
        )
    generator_for_edge = {edge: index for index, edge in enumerate(non_tree_edges)}
    edge_words = tuple(
        EdgeWordEntry(
            edge=edge,
            is_tree_edge=edge in tree_edge_set,
            forward=FiniteGroupWord(
                letters=_free_reduce(
                    _oriented_edge_word(
                        edge[0], edge[1], tree_edge_set, generator_for_edge
                    )
                )
            ),
            backward=FiniteGroupWord(
                letters=_free_reduce(
                    _oriented_edge_word(
                        edge[1], edge[0], tree_edge_set, generator_for_edge
                    )
                )
            ),
        )
        for edge in component_edges
    )
    triangle_relators: list[TriangleRelator] = []
    relator_letters = 0
    for first, second, third in triangles:
        letters = _free_reduce(
            _oriented_edge_word(first, second, tree_edge_set, generator_for_edge)
            + _oriented_edge_word(second, third, tree_edge_set, generator_for_edge)
            + _oriented_edge_word(third, first, tree_edge_set, generator_for_edge)
        )
        relator_letters += len(letters)
        triangle_relators.append(
            TriangleRelator(
                simplex=(first, second, third),
                word=FiniteGroupWord(letters=letters),
            )
        )
    if relator_letters > MAX_PRESENTATION_RELATOR_LETTERS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.fundamental_group.relator_length_budget",
            message=(
                "the reduced relators need "
                f"{relator_letters} letters, above the "
                f"{MAX_PRESENTATION_RELATOR_LETTERS}-letter bound"
            ),
        )
    generators = tuple(f"g{index}" for index in range(len(non_tree_edges)))
    presentation = FiniteGroupPresentation(
        generators=generators,
        relators=tuple(entry.word for entry in triangle_relators),
    )
    return FundamentalGroupPresentationResult._from_kernel(
        complex=complex_value,
        base_vertex=base_vertex,
        component_vertices=component_vertices,
        spanning_tree_edges=tree_edges,
        non_tree_edges=non_tree_edges,
        edge_words=edge_words,
        triangle_relators=tuple(triangle_relators),
        presentation=presentation,
        abelianization=_abelianization(generators, presentation.relators),
    )


def fundamental_group_presentation(
    complex_: FiniteSimplicialComplex,
    base_vertex: str,
) -> FundamentalGroupPresentationResult:
    """Compute the finite edge-path presentation of the basepoint component.

    The basepoint's 1-component is collapsed along one lexicographic
    breadth-first spanning tree. Non-tree edges become the generators; each
    oriented two-simplex contributes one freely reduced boundary relator.
    Tree edges are the identity word and edge reversal is the inverse word.
    The result also carries the exact exponent-sum relation matrix and its
    Smith invariants, which equal the integral first homology of the complex.
    """

    return run_topology_admission(
        lambda: _fundamental_group_kernel(complex_, base_vertex),
        location=("complex",),
    )


__all__ = [
    "concatenate_edge_paths",
    "edge_path_word",
    "free_reduce",
    "fundamental_group_presentation",
    "verify_edge_path_concatenation",
    "verify_edge_path_word",
]
