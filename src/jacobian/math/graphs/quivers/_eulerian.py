"""Directed Euler circuits on explicit, source-indexed arrows."""

from typing import Annotated, Literal, Self

import networkx as nx
from pydantic import Field, model_validator

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.quivers._models import MAX_ARROWS, FiniteQuiver


class EulerCircuitRequest(StrictModel):
    quiver: FiniteQuiver


class EulerCircuitFound(StrictModel):
    status: Literal["CIRCUIT"] = "CIRCUIT"
    start_vertex: int = Field(ge=0)
    arrow_indices: tuple[int, ...] = Field(max_length=MAX_ARROWS)


class EulerCircuitAbsent(StrictModel):
    status: Literal["NO_CIRCUIT"] = "NO_CIRCUIT"


class DirectedEulerCircuit(StrictModel):
    """A closed arrow ordering, or nonexistence on the non-isolated support.

    Arrow indices refer to positions in the retained quiver, preserving loops
    and parallel arrows. Empty support has the empty circuit starting at 0.
    """

    quiver: FiniteQuiver
    outcome: Annotated[
        EulerCircuitFound | EulerCircuitAbsent, Field(discriminator="status")
    ]

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if isinstance(self.outcome, EulerCircuitFound):
            if self.outcome.start_vertex >= self.quiver.vertex_count:
                raise ValueError("start vertex must belong to the source quiver")
            indices = self.outcome.arrow_indices
            if len(indices) != len(self.quiver.arrows) or set(indices) != set(
                range(len(indices))
            ):
                raise ValueError("circuit indices must permute all source arrows")
        return self


def directed_euler_circuit(quiver: FiniteQuiver) -> DirectedEulerCircuit:
    """Construct a circuit in O(V+E) work and storage, with E output indices.

    The carrier bounds 128 vertices and 16,384 explicit arrows. All counts
    and indices fit those axes; no multiplicity or dense graph expansion occurs.
    """
    request_checkpoint("before Euler circuit construction")
    if not quiver.arrows:
        return DirectedEulerCircuit(
            quiver=quiver, outcome=EulerCircuitFound(start_vertex=0, arrow_indices=())
        )
    graph: nx.MultiDiGraph[int] = nx.MultiDiGraph()
    for index, (source, target) in enumerate(quiver.arrows):
        graph.add_edge(source, target, key=index)
    if not nx.is_eulerian(graph):
        return DirectedEulerCircuit(quiver=quiver, outcome=EulerCircuitAbsent())
    start = min(graph)
    indices = tuple(
        index for _, _, index in nx.eulerian_circuit(graph, source=start, keys=True)
    )
    request_checkpoint("before Euler result construction")
    return DirectedEulerCircuit(
        quiver=quiver,
        outcome=EulerCircuitFound(start_vertex=start, arrow_indices=indices),
    )


def _run(request: EulerCircuitRequest) -> DirectedEulerCircuit:
    return directed_euler_circuit(request.quiver)


EULER_CIRCUIT = MathTool(
    operation_id="graph.directed.euler_circuit.compute",
    title="Construct a directed Euler circuit preserving arrow identity",
    description="Return a closed ordering using each explicit quiver arrow exactly once, or decide no circuit exists. Loops and parallel arrows retain their source indices. Isolated vertices are ignored; empty support has an empty circuit at vertex 0. Linear work and output in at most 16,384 arrows and 128 vertices.",
    request_type=EulerCircuitRequest,
    result_type=DirectedEulerCircuit,
    run=_run,
    tags=("graph", "directed", "eulerian", "circuit", "multigraph", "quiver"),
    examples=(
        OperationExample(
            name="loop_and_parallel_arrows",
            description="Construct a directed Euler circuit; arrows are explicit ordered pairs and parallel occurrences have distinct indices.",
            input={
                "quiver": {
                    "vertex_count": 2,
                    "arrows": [[0, 0], [0, 1], [0, 1], [1, 0], [1, 0]],
                }
            },
        ),
    ),
)
