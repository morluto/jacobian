"""Cycle-length profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.cycle_length_profile._models import (
    CycleLengthProfileRequest,
    CycleLengthProfileResult,
    FixedLengthCycleEnumerationRequest,
    FixedLengthCycleEnumerationResult,
)
from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
    enumerate_chordless_fixed_length_cycles,
    enumerate_fixed_length_cycles,
)


def compute_cycle_length_profile_op(
    request: CycleLengthProfileRequest,
) -> CycleLengthProfileResult:
    return compute_cycle_length_profile(request.graph)


def _enumerate_cycles(
    request: FixedLengthCycleEnumerationRequest,
) -> FixedLengthCycleEnumerationResult:
    return enumerate_fixed_length_cycles(request.graph, request.cycle_length)


def _enumerate_chordless_cycles(
    request: FixedLengthCycleEnumerationRequest,
) -> FixedLengthCycleEnumerationResult:
    return enumerate_chordless_fixed_length_cycles(request.graph, request.cycle_length)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.invariant.cycle_length_profile.compute",
        title="Compute the complete cycle-length profile of a graph",
        description=(
            "Given a bounded finite simple graph, return the complete set of "
            "simple-cycle lengths together with one canonical witness cycle "
            "for each occurring length. Admission requires the first-witness "
            "search to fit 10,000,000 work units and bounds retained label "
            "characters independently of transport encoding."
        ),
        request_type=CycleLengthProfileRequest,
        result_type=CycleLengthProfileResult,
        run=compute_cycle_length_profile_op,
        tags=("graph", "invariant", "exact"),
        examples=(
            OperationExample(
                name="c4",
                description="C4 has cycle-length spectrum {4}.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"], ["0", "3"]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.cycle.fixed_length.enumerate",
        title="Enumerate fixed-length simple cycles",
        description=(
            "Return every simple cycle of the requested length in dihedral "
            "canonical form, with the exact count and complete source vertex "
            "and edge incidence indexes; the input must be a canonical finite "
            "simple undirected graph and the complete family must fit the "
            "admitted traversal and result envelope."
        ),
        request_type=FixedLengthCycleEnumerationRequest,
        result_type=FixedLengthCycleEnumerationResult,
        run=_enumerate_cycles,
        tags=("graph", "cycle", "enumeration", "exact"),
        discovery_terms=(
            "enumerate fixed-length simple cycles",
            "complete simple cycle family",
        ),
        examples=(
            OperationExample(
                name="square",
                description="Enumerate the unique four-cycle of a square.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["0", "3"], ["1", "2"], ["2", "3"]],
                    },
                    "cycle_length": 4,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.cycle.chordless_fixed_length.enumerate",
        title="Enumerate fixed-length chordless cycles",
        description=(
            "Return every induced (chordless) cycle of the requested length in "
            "dihedral canonical form, with the exact count and complete source "
            "vertex and edge incidence indexes; the input must be a canonical "
            "finite simple undirected graph and the complete family must fit "
            "the admitted traversal and result envelope."
        ),
        request_type=FixedLengthCycleEnumerationRequest,
        result_type=FixedLengthCycleEnumerationResult,
        run=_enumerate_chordless_cycles,
        tags=("graph", "cycle", "chordless", "enumeration", "exact"),
        discovery_terms=(
            "enumerate fixed-length chordless cycles",
            "induced cycle family",
        ),
        examples=(
            OperationExample(
                name="square",
                description="Enumerate the unique chordless four-cycle of a square.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["0", "3"], ["1", "2"], ["2", "3"]],
                    },
                    "cycle_length": 4,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
