"""Minimal transversal enumeration declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.finite_structures.hypergraphs._transversal_enumeration import (
    MinimalTransversalEnumerationRequest,
    MinimalTransversalEnumerationResult,
    enumerate_minimal_transversals,
)

MINIMAL_TRANSVERSAL_ENUMERATION_OPERATION = MathTool(
    operation_id="hypergraph.minimal_transversals.bounded_cardinality.enumerate",
    title="Enumerate bounded-cardinality minimal hypergraph transversals",
    description=(
        "Enumerate every inclusion-minimal transversal of cardinality at most k "
        "with a complete cardinality profile. The owner admits exhaustive "
        "candidate-edge and minimality work plus row, incidence, and transport "
        "bounds before search. An edge-free hypergraph returns its unique empty "
        "minimal transversal; a family containing an empty edge returns no "
        "transversals."
    ),
    request_type=MinimalTransversalEnumerationRequest,
    result_type=MinimalTransversalEnumerationResult,
    run=enumerate_minimal_transversals,
    tags=("combinatorics", "hypergraph", "transversal", "enumerate", "complete"),
    examples=(
        OperationExample(
            name="two_crossing_edges",
            description=(
                "Enumerate minimal transversals of {a,b} and {b,c} through size "
                "two; the complete bounded slice must fit the admitted work and "
                "transport envelope."
            ),
            input={
                "hypergraph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["e0", ["a", "b"]], ["e1", ["b", "c"]]],
                },
                "maximum_cardinality": 2,
            },
        ),
    ),
)
