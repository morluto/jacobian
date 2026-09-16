"""Discrete Morse matching operation declarations."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingRequest,
    DiscreteMorseMatchingResult,
)
from jacobian.math.topology.discrete_morse.operations import construct_matching

__all__ = ["TOOLS"]

_CIRCLE_FACETS = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}

TOOLS: MathTools = (
    MathTool(
        operation_id="topology.discrete_morse.matching.construct",
        title="Construct and classify a discrete Morse matching",
        description=(
            "Classify one caller-supplied family of face/coface pairs on the "
            "complete face closure of a bounded finite simplicial complex as an "
            "acyclic discrete Morse matching with its exact critical-cell profile "
            "and Hasse topological order, a valid pairing containing one concrete "
            "closed V-path, or an invalid family carrying its first fault. Each "
            "pair must be a codimension-one cover relation of the face poset and "
            "each cell may occur in at most one pair; acyclicity is decided "
            "exactly on the directed Hasse graph whose unmatched cover edges "
            "point down and matched edges point up."
        ),
        request_type=DiscreteMorseMatchingRequest,
        result_type=DiscreteMorseMatchingResult,
        run=construct_matching,
        tags=(
            "topology",
            "discrete-morse",
            "matching",
            "critical-cells",
            "hasse-diagram",
            "exact",
        ),
        discovery_terms=(
            "discrete Morse matching",
            "critical cells",
            "acyclic matching",
            "closed V-path",
            "Morse matching",
            "Hasse diagram acyclicity",
        ),
        examples=(
            OperationExample(
                name="circle_one_critical_vertex_and_edge",
                description=(
                    "Classify a matching on a triangulated circle leaving one "
                    "vertex and one edge critical; pairs must be cover relations "
                    "with each cell in at most one pair."
                ),
                input={
                    "complex": _CIRCLE_FACETS,
                    "pairs": [
                        {"face": ["a"], "coface": ["a", "b"]},
                        {"face": ["c"], "coface": ["a", "c"]},
                    ],
                },
            ),
            OperationExample(
                name="circle_closed_v_path",
                description=(
                    "Return the concrete closed V-path of a locally valid circle "
                    "matching whose directed Hasse graph contains a cycle; every "
                    "pair is still a disjoint cover relation."
                ),
                input={
                    "complex": _CIRCLE_FACETS,
                    "pairs": [
                        {"face": ["a"], "coface": ["a", "b"]},
                        {"face": ["b"], "coface": ["b", "c"]},
                        {"face": ["c"], "coface": ["a", "c"]},
                    ],
                },
            ),
        ),
    ),
)
