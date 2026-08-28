"""Electrical-network operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.electrical_networks._models import (
    EffectiveResistanceRequest,
    EffectiveResistanceResult,
    LaplacianRequest,
    LaplacianResult,
    NodePotentialRequest,
    NodePotentialResult,
)
from jacobian.math.graphs.electrical_networks._operations import (
    compute_effective_resistance,
    compute_laplacian,
    compute_node_potentials,
)


def en_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    en_operation(
        "electrical_network.effective_resistance.compute",
        "Compute the exact effective resistance between two terminals",
        "Compute the exact rational effective resistance between two terminals of an undirected conductance network by solving the reduced Laplacian system over QQ.",
        EffectiveResistanceRequest,
        EffectiveResistanceResult,
        compute_effective_resistance,
        "graph",
        "electrical-network",
        "effective-resistance",
        "exact",
        examples=(
            example(
                "triangle_equal_resistances",
                "Effective resistance of two vertices in a triangle with unit resistances.",
                {
                    "network": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "conductance": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "conductance": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 0,
                                "target": 2,
                                "conductance": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                    "terminal_a": 0,
                    "terminal_b": 1,
                },
            ),
        ),
    ),
    en_operation(
        "electrical_network.node_potentials.compute",
        "Compute exact node potentials for unit current injection",
        "Solve the Dirichlet problem: inject 1 ampere at source and extract 1 ampere at sink, returning exact rational node potentials with the sink gauge fixed at zero.",
        NodePotentialRequest,
        NodePotentialResult,
        compute_node_potentials,
        "graph",
        "electrical-network",
        "node-potential",
        "exact",
        examples=(
            example(
                "path_of_two_edges",
                "Node potentials for a path graph of 3 vertices with unit conductances.",
                {
                    "network": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "conductance": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "conductance": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 2,
                },
            ),
        ),
    ),
    en_operation(
        "electrical_network.laplacian.compute",
        "Compute the exact conductance-weighted Laplacian matrix",
        "Build the exact rational conductance-weighted graph Laplacian of an undirected network, returned as a flat list of (row, col, value) entries.",
        LaplacianRequest,
        LaplacianResult,
        compute_laplacian,
        "graph",
        "electrical-network",
        "laplacian",
        "exact",
        examples=(
            example(
                "single_edge",
                "Laplacian of a two-vertex network with one unit-conductance edge.",
                {
                    "network": {
                        "vertex_count": 2,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "conductance": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
