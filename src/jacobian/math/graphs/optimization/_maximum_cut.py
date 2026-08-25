"""Bounded exact maximum cut on canonical simple undirected graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAXIMUM_CUT_CANDIDATE_PARTITIONS = 1_048_576
"""Maximum reduced component assignments admitted by one exact request."""

MAXIMUM_CUT_EDGE_UPDATES = 5_000_000
"""Maximum weighted adjacency updates in one complete result replay."""

MAXIMUM_CUT_RESULT_BYTES = CanonicalLimits().max_output_bytes
"""Maximum projected canonical JSON bytes for one exact result."""

MAXIMUM_CUT_Z3_RLIMIT = 100_000
"""Per-component Z3 resource limit before the exact fallback takes over."""


@dataclass(frozen=True, slots=True)
class _ReducedEdge:
    left: int
    right: int
    weight: int


@dataclass(frozen=True, slots=True)
class _ReducedComponent:
    class_ids: tuple[int, ...]
    edges: tuple[_ReducedEdge, ...]
    bipartite_sides: tuple[bool, ...] | None
    root: int
    variable_order: tuple[int, ...]
    candidate_partitions: int


@dataclass(frozen=True, slots=True)
class _MaximumCutAnalysis:
    twin_classes: tuple[tuple[int, ...], ...]
    class_of_vertex: tuple[int, ...]
    components: tuple[_ReducedComponent, ...]
    candidate_partitions: int
    edge_updates: int


@dataclass(frozen=True, slots=True)
class _ComponentSolution:
    value: int
    sides: tuple[bool, ...]


def _connected_components(
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(len(adjacency)))
    components: list[tuple[int, ...]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor, _weight in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _bipartite_sides(
    class_ids: tuple[int, ...],
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[bool, ...] | None:
    colors: dict[int, bool] = {class_ids[0]: False}
    queue = [class_ids[0]]
    for vertex in queue:
        for neighbor, _weight in adjacency[vertex]:
            expected = not colors[vertex]
            if neighbor not in colors:
                colors[neighbor] = expected
                queue.append(neighbor)
            elif colors[neighbor] != expected:
                return None
    return tuple(colors[class_id] for class_id in class_ids)


def _analyze_graph(graph: SimpleUndirectedGraph) -> _MaximumCutAnalysis:
    """Derive the exact false-twin quotient and exhaustive replay budget.

    Vertices with equal open neighborhoods are nonadjacent false twins. For a
    fixed assignment of every other vertex, their incident cut contributions
    are independent and identical, so homogenizing the class on whichever side
    is no worse preserves an optimum. Aggregating source edges between classes
    as integer weights therefore preserves the exact maximum-cut value.
    """

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    source_adjacency: list[set[int]] = [set() for _vertex in graph.vertices]
    indexed_edges: list[tuple[int, int]] = []
    for left_label, right_label in graph.edges:
        left_index = vertex_index[left_label]
        right_index = vertex_index[right_label]
        indexed_edges.append((left_index, right_index))
        source_adjacency[left_index].add(right_index)
        source_adjacency[right_index].add(left_index)
    source_adjacency_value = tuple(
        tuple(sorted(neighbors)) for neighbors in source_adjacency
    )

    classes_by_neighborhood: dict[tuple[int, ...], list[int]] = {}
    for vertex, neighbors in enumerate(source_adjacency_value):
        classes_by_neighborhood.setdefault(neighbors, []).append(vertex)
    twin_classes = tuple(
        tuple(vertices) for vertices in classes_by_neighborhood.values()
    )
    class_of_vertex = [0] * len(graph.vertices)
    for class_id, vertices in enumerate(twin_classes):
        for vertex in vertices:
            class_of_vertex[vertex] = class_id

    reduced_weights: dict[tuple[int, int], int] = {}
    for left_index, right_index in indexed_edges:
        left_class = class_of_vertex[left_index]
        right_class = class_of_vertex[right_index]
        if left_class == right_class:
            raise RuntimeError("false-twin reduction produced an internal edge")
        endpoints = (
            (left_class, right_class)
            if left_class < right_class
            else (right_class, left_class)
        )
        reduced_weights[endpoints] = reduced_weights.get(endpoints, 0) + 1

    adjacency_lists: list[list[tuple[int, int]]] = [[] for _class in twin_classes]
    reduced_edges = tuple(
        _ReducedEdge(left, right, weight)
        for (left, right), weight in sorted(reduced_weights.items())
    )
    for edge in reduced_edges:
        adjacency_lists[edge.left].append((edge.right, edge.weight))
        adjacency_lists[edge.right].append((edge.left, edge.weight))
    adjacency = tuple(tuple(sorted(neighbors)) for neighbors in adjacency_lists)

    component_ids = _connected_components(adjacency)
    component_of_class = {
        class_id: component_index
        for component_index, class_ids in enumerate(component_ids)
        for class_id in class_ids
    }
    component_edges: list[list[_ReducedEdge]] = [[] for _component in component_ids]
    for edge in reduced_edges:
        component_edges[component_of_class[edge.left]].append(edge)

    components: list[_ReducedComponent] = []
    candidate_partitions = 0
    edge_updates = 0
    for component_index, class_ids in enumerate(component_ids):
        coloring = _bipartite_sides(class_ids, adjacency)
        if coloring is not None:
            components.append(
                _ReducedComponent(
                    class_ids=class_ids,
                    edges=tuple(component_edges[component_index]),
                    bipartite_sides=coloring,
                    root=class_ids[0],
                    variable_order=(),
                    candidate_partitions=0,
                )
            )
            continue

        root = max(
            class_ids, key=lambda class_id: (len(adjacency[class_id]), -class_id)
        )
        variable_order = tuple(
            sorted(
                (class_id for class_id in class_ids if class_id != root),
                key=lambda class_id: (len(adjacency[class_id]), class_id),
            )
        )
        component_candidates = 1 << len(variable_order)
        component_updates = sum(
            len(adjacency[class_id]) * (1 << (len(variable_order) - position - 1))
            for position, class_id in enumerate(variable_order)
        )
        candidate_partitions += component_candidates
        edge_updates += component_updates
        components.append(
            _ReducedComponent(
                class_ids=class_ids,
                edges=tuple(component_edges[component_index]),
                bipartite_sides=None,
                root=root,
                variable_order=variable_order,
                candidate_partitions=component_candidates,
            )
        )

    return _MaximumCutAnalysis(
        twin_classes=twin_classes,
        class_of_vertex=tuple(class_of_vertex),
        components=tuple(components),
        candidate_partitions=candidate_partitions,
        edge_updates=edge_updates,
    )


def _projected_result_bytes(graph: SimpleUndirectedGraph) -> int:
    """Return a conservative exact-result JSON size before search."""

    graph_value = graph.model_dump(mode="json")
    worst_case = {
        "graph": graph_value,
        # A valid partition contains every source vertex exactly once across
        # both sides. Putting every vertex on one side maximizes list separators
        # and is therefore a conservative exact bound for the partition fields.
        "left_vertices": list(graph.vertices),
        "right_vertices": [],
        "crossing_edges": [list(edge) for edge in graph.edges],
        "cut_value": len(graph.edges),
        "lower_bound": len(graph.edges),
        "upper_bound": len(graph.edges),
    }
    return len(
        encode_strict_json(
            worst_case,
            limits=CanonicalLimits(max_output_bytes=4 * MAXIMUM_CUT_RESULT_BYTES),
        )
    )


def _require_graph_envelope(graph: SimpleUndirectedGraph) -> _MaximumCutAnalysis:
    analysis = _analyze_graph(graph)
    if analysis.candidate_partitions > MAXIMUM_CUT_CANDIDATE_PARTITIONS:
        raise ValueError(
            "maximum-cut exact preflight requires "
            f"{analysis.candidate_partitions} candidate partitions; at most "
            f"{MAXIMUM_CUT_CANDIDATE_PARTITIONS} are admitted"
        )
    if analysis.edge_updates > MAXIMUM_CUT_EDGE_UPDATES:
        raise ValueError(
            "maximum-cut exact preflight requires "
            f"{analysis.edge_updates} incremental weighted edge contributions; at most "
            f"{MAXIMUM_CUT_EDGE_UPDATES} are admitted"
        )
    projected_bytes = _projected_result_bytes(graph)
    if projected_bytes > MAXIMUM_CUT_RESULT_BYTES:
        raise ValueError(
            "maximum-cut projected exact result requires at most "
            f"{projected_bytes} bytes; the admitted result bound is "
            f"{MAXIMUM_CUT_RESULT_BYTES} bytes"
        )
    return analysis


def _maximum_cut_graph_schema() -> JsonSchemaValue:
    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "Canonical materialized SimpleUndirectedGraph for an exact-only maximum-cut "
        "request. Admission preflights a complete exact proof with at most "
        f"{MAXIMUM_CUT_CANDIDATE_PARTITIONS} internally derived candidate partitions, "
        f"{MAXIMUM_CUT_EDGE_UPDATES} incremental weighted edge contributions, and a "
        f"projected exact result of at most {MAXIMUM_CUT_RESULT_BYTES} bytes. Requests "
        "outside any bound are rejected before search."
    )
    return schema


MaximumCutGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(_maximum_cut_graph_schema()),
]


class GraphMaximumCutRequest(StrictModel):
    """One materialized graph inside the complete exact search envelope."""

    graph: MaximumCutGraph

    @model_validator(mode="after")
    def require_complete_search_envelope(self) -> Self:
        _require_graph_envelope(self.graph)
        return self


class GraphMaximumCutResult(StrictModel):
    """An exact maximum cut bound to its complete canonical source graph."""

    graph: SimpleUndirectedGraph
    left_vertices: tuple[str, ...] = Field(max_length=256)
    right_vertices: tuple[str, ...] = Field(max_length=256)
    crossing_edges: tuple[tuple[str, str], ...] = Field(max_length=32_640)
    cut_value: StrictInt = Field(ge=0, le=32_640)
    lower_bound: StrictInt = Field(ge=0, le=32_640)
    upper_bound: StrictInt = Field(ge=0, le=32_640)

    @model_validator(mode="after")
    def bind_cut_to_source_and_replay_optimality(self) -> Self:
        analysis = _require_graph_envelope(self.graph)
        graph_vertices = set(self.graph.vertices)
        left = set(self.left_vertices)
        right = set(self.right_vertices)
        if (
            len(left) != len(self.left_vertices)
            or len(right) != len(self.right_vertices)
            or left & right
            or left | right != graph_vertices
            or self.left_vertices
            != tuple(vertex for vertex in self.graph.vertices if vertex in left)
            or self.right_vertices
            != tuple(vertex for vertex in self.graph.vertices if vertex in right)
        ):
            raise ValueError(
                "bipartition sides must partition the source graph vertices in "
                "source-axis order"
            )

        expected_crossing = tuple(
            edge for edge in self.graph.edges if (edge[0] in left) != (edge[1] in left)
        )
        if self.crossing_edges != expected_crossing:
            raise ValueError(
                "crossing-edge ledger must equal the source graph edges crossing "
                "the submitted partition in source-edge order"
            )
        if self.cut_value != len(expected_crossing):
            raise ValueError(
                "cut value must equal the crossing-edge ledger cardinality"
            )
        if self.lower_bound != self.cut_value or self.upper_bound != self.cut_value:
            raise ValueError("an exact result requires exact bounds equal to cut value")

        replayed_value, _sides = _solve_analysis_by_enumeration(analysis)
        if self.cut_value != replayed_value:
            raise ValueError(
                "cut value is feasible but not maximum for the retained source graph"
            )
        return self


def _solve_component_by_enumeration(
    analysis: _MaximumCutAnalysis,
    component_index: int,
) -> _ComponentSolution:
    component = analysis.components[component_index]
    if component.bipartite_sides is not None:
        return _ComponentSolution(
            value=sum(edge.weight for edge in component.edges),
            sides=component.bipartite_sides,
        )

    local_index = {
        class_id: position for position, class_id in enumerate(component.class_ids)
    }
    local_adjacency: list[list[tuple[int, int]]] = [
        [] for _class_id in component.class_ids
    ]
    for edge in component.edges:
        left = local_index[edge.left]
        right = local_index[edge.right]
        local_adjacency[left].append((right, edge.weight))
        local_adjacency[right].append((left, edge.weight))

    sides = [False] * len(component.class_ids)
    best_sides = tuple(sides)
    current_value = 0
    best_value = 0
    for step in range(1, component.candidate_partitions):
        changed_position = (step & -step).bit_length() - 1
        changed_class = component.variable_order[changed_position]
        changed = local_index[changed_class]
        old_side = sides[changed]
        for neighbor, weight in local_adjacency[changed]:
            current_value += weight if sides[neighbor] == old_side else -weight
        sides[changed] = not old_side
        side_value = tuple(sides)
        if current_value > best_value or (
            current_value == best_value and side_value < best_sides
        ):
            best_value = current_value
            best_sides = side_value
    return _ComponentSolution(value=best_value, sides=best_sides)


def _solve_component_with_z3(
    analysis: _MaximumCutAnalysis,
    component_index: int,
) -> _ComponentSolution | None:
    """Return an exact deterministic Z3 optimum, or defer to the fallback."""

    component = analysis.components[component_index]
    if component.bipartite_sides is not None:
        return _ComponentSolution(
            value=sum(edge.weight for edge in component.edges),
            sides=component.bipartite_sides,
        )

    import z3  # type: ignore[import-untyped]

    try:
        optimizer = z3.Optimize()
        optimizer.set(priority="lex", rlimit=MAXIMUM_CUT_Z3_RLIMIT)
        variables = {
            class_id: z3.Bool(f"maximum_cut_{component_index}_{position}")
            for position, class_id in enumerate(component.class_ids)
        }
        optimizer.add(variables[component.root] == z3.BoolVal(False))
        objective = z3.Sum(
            [
                edge.weight * z3.If(variables[edge.left] != variables[edge.right], 1, 0)
                for edge in component.edges
            ]
        )
        objective_handle = optimizer.maximize(objective)
        tie_break = z3.Sum(
            [
                z3.If(
                    variables[class_id],
                    1 << (len(component.class_ids) - position - 1),
                    0,
                )
                for position, class_id in enumerate(component.class_ids)
            ]
        )
        optimizer.minimize(tie_break)
        if optimizer.check() != z3.sat:
            return None
        lower = objective_handle.lower()
        upper = objective_handle.upper()
        if not (z3.is_int_value(lower) and z3.is_int_value(upper)):
            return None
        lower_value = lower.as_long()
        upper_value = upper.as_long()
        if lower_value != upper_value:
            return None
        model = optimizer.model()
        model_value = model.eval(objective, model_completion=True)
        if not z3.is_int_value(model_value) or model_value.as_long() != lower_value:
            return None
        sides = tuple(
            z3.is_true(model.eval(variables[class_id], model_completion=True))
            for class_id in component.class_ids
        )
        return _ComponentSolution(value=lower_value, sides=sides)
    except z3.Z3Exception:
        return None


def _combine_component_solutions(
    analysis: _MaximumCutAnalysis,
    solutions: tuple[_ComponentSolution, ...],
) -> tuple[int, tuple[bool, ...]]:
    class_sides = [False] * len(analysis.twin_classes)
    value = 0
    for component, solution in zip(analysis.components, solutions, strict=True):
        value += solution.value
        for class_id, side in zip(component.class_ids, solution.sides, strict=True):
            class_sides[class_id] = side
    return value, tuple(class_sides)


def _solve_analysis_by_enumeration(
    analysis: _MaximumCutAnalysis,
) -> tuple[int, tuple[bool, ...]]:
    return _combine_component_solutions(
        analysis,
        tuple(
            _solve_component_by_enumeration(analysis, component_index)
            for component_index in range(len(analysis.components))
        ),
    )


def _solve_analysis(
    analysis: _MaximumCutAnalysis,
) -> tuple[int, tuple[bool, ...]]:
    solutions: list[_ComponentSolution] = []
    for component_index in range(len(analysis.components)):
        solution = _solve_component_with_z3(analysis, component_index)
        if solution is None:
            solution = _solve_component_by_enumeration(analysis, component_index)
        solutions.append(solution)
    return _combine_component_solutions(analysis, tuple(solutions))


def _partition_from_class_sides(
    graph: SimpleUndirectedGraph,
    analysis: _MaximumCutAnalysis,
    class_sides: tuple[bool, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    vertex_sides = tuple(class_sides[class_id] for class_id in analysis.class_of_vertex)
    left_vertices = tuple(
        vertex
        for vertex, side in zip(graph.vertices, vertex_sides, strict=True)
        if not side
    )
    right_vertices = tuple(
        vertex
        for vertex, side in zip(graph.vertices, vertex_sides, strict=True)
        if side
    )
    right = set(right_vertices)
    crossing_edges = tuple(
        edge for edge in graph.edges if (edge[0] in right) != (edge[1] in right)
    )
    return left_vertices, right_vertices, crossing_edges


def compute_maximum_cut(request: GraphMaximumCutRequest) -> GraphMaximumCutResult:
    """Compute one deterministic exact maximum cut and source-edge ledger."""

    analysis = _analyze_graph(request.graph)
    cut_value, class_sides = _solve_analysis(analysis)
    left_vertices, right_vertices, crossing_edges = _partition_from_class_sides(
        request.graph,
        analysis,
        class_sides,
    )
    if len(crossing_edges) != cut_value:
        cut_value, class_sides = _solve_analysis_by_enumeration(analysis)
        left_vertices, right_vertices, crossing_edges = _partition_from_class_sides(
            request.graph,
            analysis,
            class_sides,
        )

    # The producer already has coincident exact backend bounds. Avoid paying a
    # second complete search inside this call; independently supplied or
    # deserialized results always execute the exhaustive source-bound replay.
    return GraphMaximumCutResult.model_construct(
        graph=request.graph,
        left_vertices=left_vertices,
        right_vertices=right_vertices,
        crossing_edges=crossing_edges,
        cut_value=cut_value,
        lower_bound=cut_value,
        upper_bound=cut_value,
    )


MAXIMUM_CUT_OPERATION: MathTool[
    GraphMaximumCutRequest,
    GraphMaximumCutResult,
] = MathTool(
    operation_id="graph.cut.maximum.compute",
    title="Exact maximum cut",
    description=(
        "Compute one exact maximum-cardinality bipartition cut of a bounded "
        "canonical simple undirected graph. Return both partition sides, the source-"
        "ordered crossing-edge ledger, and coincident lower and upper bounds; requests "
        "outside the complete exact envelope are rejected before search."
    ),
    request_type=GraphMaximumCutRequest,
    result_type=GraphMaximumCutResult,
    run=compute_maximum_cut,
    tags=(
        "graph",
        "cut",
        "max-cut",
        "maximum-bipartition",
        "exact",
        "bounded",
    ),
    examples=(
        example(
            "cycle_five",
            (
                "Compute an exact maximum cut of the five-cycle; the materialized "
                "simple graph must satisfy the published exact work and result bounds."
            ),
            {
                "graph": {
                    "vertices": ["0", "1", "2", "3", "4"],
                    "edges": [
                        ["0", "1"],
                        ["0", "4"],
                        ["1", "2"],
                        ["2", "3"],
                        ["3", "4"],
                    ],
                }
            },
        ),
    ),
)


__all__ = [
    "MAXIMUM_CUT_CANDIDATE_PARTITIONS",
    "MAXIMUM_CUT_EDGE_UPDATES",
    "MAXIMUM_CUT_OPERATION",
    "MAXIMUM_CUT_RESULT_BYTES",
    "GraphMaximumCutRequest",
    "GraphMaximumCutResult",
    "compute_maximum_cut",
]
