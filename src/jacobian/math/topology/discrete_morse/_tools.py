"""Discrete Morse matching operation declarations."""

from __future__ import annotations

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    FiniteSimplicialComplex,
)
from jacobian.math.topology.discrete_morse._models import (
    MAX_MORSE_CONTRACTION_FACE_CANDIDATES,
    DiscreteMorseMatchingRequest,
    DiscreteMorseMatchingResult,
    GradientPathsRequest,
    GradientPathsResult,
    IntegerMorseComplexRequest,
    IntegerMorseComplexResult,
    MinimumMorseMatchingRequest,
    MinimumMorseMatchingResult,
    MorseChainContractionRequest,
    MorseChainContractionResult,
    MorseComplexRequest,
    MorseComplexResult,
)
from jacobian.math.topology.discrete_morse.extensions import minimum_matching
from jacobian.math.topology.discrete_morse.extensions_tools import (
    TOOLS as EXTENSION_TOOLS,
)
from jacobian.math.topology.discrete_morse.operations import (
    compute_chain_contraction,
    compute_gradient_paths,
    compute_integer_morse_complex,
    compute_morse_complex,
    construct_matching,
)
from jacobian.math.topology.operations import canonicalize

__all__ = ["TOOLS"]


def _run_construct_matching(
    request: DiscreteMorseMatchingRequest,
) -> DiscreteMorseMatchingResult:
    canonical: FiniteSimplicialComplex = canonicalize(
        request.complex.vertices, request.complex.facets
    ).complex
    return construct_matching(canonical, request.pairs)


def _run_compute_gradient_paths(
    request: GradientPathsRequest,
) -> GradientPathsResult:
    canonical = canonicalize(request.complex.vertices, request.complex.facets).complex
    return compute_gradient_paths(
        canonical, request.pairs, request.start, request.target
    )


def _run_compute_morse_complex(request: MorseComplexRequest) -> MorseComplexResult:
    canonical = canonicalize(request.complex.vertices, request.complex.facets).complex
    return compute_morse_complex(canonical, request.pairs)


def _run_compute_integer_morse_complex(
    request: IntegerMorseComplexRequest,
) -> IntegerMorseComplexResult:
    canonical = canonicalize(request.complex.vertices, request.complex.facets).complex
    return compute_integer_morse_complex(canonical, request.pairs)


def _run_minimum_matching(
    request: MinimumMorseMatchingRequest,
) -> MinimumMorseMatchingResult:
    return minimum_matching(request)


def _run_compute_chain_contraction(
    request: MorseChainContractionRequest,
) -> MorseChainContractionResult:
    if any(
        len(facet) > MAX_TOPOLOGY_DIMENSION + 1 for facet in request.complex.facets
    ) or any(
        len(cell) > MAX_TOPOLOGY_DIMENSION + 1
        for pair in request.pairs
        for cell in (pair.face, pair.coface)
    ):
        raise OperationResourceAdmissionError(
            location=("complex", "pairs"),
            code="topology.discrete_morse.contraction.axes",
            message="facets and matching cells exceed the bounded dimension axis",
        )
    possible_faces = sum((1 << len(facet)) - 1 for facet in request.complex.facets)
    if possible_faces > MAX_MORSE_CONTRACTION_FACE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="topology.discrete_morse.contraction.face_candidates",
            message="request facets exceed the 512-candidate closure admission bound",
        )
    canonical = canonicalize(request.complex.vertices, request.complex.facets).complex
    return compute_chain_contraction(canonical, request.pairs)


_CIRCLE_FACETS = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}

_INTERVAL_FACETS = {
    "vertices": ["a", "b"],
    "facets": [["a", "b"]],
}

_TRIANGLE_FACETS = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b", "c"]],
}

_CIRCLE_MATCHING_PAIRS = [
    {"face": ["a"], "coface": ["a", "b"]},
    {"face": ["c"], "coface": ["a", "c"]},
]

TOOLS: MathTools = (
    *EXTENSION_TOOLS,
    MathTool(
        operation_id="topology.discrete_morse.matching.minimum.compute",
        title="Compute a minimum-critical-cell acyclic matching",
        description=(
            "Exhaustively search the admitted finite face poset for an acyclic "
            "matching with the minimum total number of critical cells. Ties use "
            "the first optimum found by canonical lexicographic cover-pair order. "
            "The operation rejects inputs whose complete search bound exceeds "
            "its envelope; it never returns a partial incumbent as a minimum."
        ),
        request_type=MinimumMorseMatchingRequest,
        result_type=MinimumMorseMatchingResult,
        run=_run_minimum_matching,
        tags=("topology", "discrete-morse", "minimum", "exact", "search"),
        discovery_terms=(
            "minimum critical cells",
            "optimal discrete Morse matching",
            "exact acyclic matching optimization",
        ),
        examples=(
            OperationExample(
                name="filled_triangle_minimum_matching",
                description=(
                    "Find an acyclic matching of a filled triangle with one "
                    "critical vertex, the minimum possible total."
                ),
                input={"complex": _TRIANGLE_FACETS},
            ),
        ),
    ),
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
        run=_run_construct_matching,
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
    MathTool(
        operation_id="topology.discrete_morse.gradient_paths.compute",
        title="Enumerate discrete Morse gradient paths from a critical cell",
        description=(
            "Given an acyclic matching on a bounded finite simplicial complex "
            "and a selected critical cell, enumerate the complete bounded family "
            "of gradient paths (V-paths) leaving that cell toward critical cells "
            "of the adjacent lower dimension. Each path is replayed as its "
            "explicit alternating matched/unmatched cover-step sequence, and an "
            "optional critical target restricts the family. A non-acyclic "
            "matching or a non-critical selection is rejected; exceeding the "
            "path, search-state, or path-length envelope is a resource "
            "rejection, never a truncated family."
        ),
        request_type=GradientPathsRequest,
        result_type=GradientPathsResult,
        run=_run_compute_gradient_paths,
        tags=(
            "topology",
            "discrete-morse",
            "gradient-paths",
            "v-path",
            "critical-cells",
            "exact",
        ),
        discovery_terms=(
            "discrete Morse gradient path",
            "V-path",
            "gradient flow path",
            "alternating matched unmatched path",
            "critical cell paths",
        ),
        examples=(
            OperationExample(
                name="circle_edge_to_critical_vertex",
                description=(
                    "Enumerate both gradient paths from the critical circle edge "
                    "to the critical vertex left by the supplied acyclic matching."
                ),
                input={
                    "complex": _CIRCLE_FACETS,
                    "pairs": _CIRCLE_MATCHING_PAIRS,
                    "start": ["b", "c"],
                    "target": ["b"],
                },
            ),
            OperationExample(
                name="interval_single_down_step",
                description=(
                    "With the empty acyclic matching every cell is critical, so "
                    "the only gradient path from an edge to a vertex is its "
                    "single unmatched down step."
                ),
                input={
                    "complex": _INTERVAL_FACETS,
                    "pairs": [],
                    "start": ["a", "b"],
                    "target": ["a"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.discrete_morse.complex.compute",
        title="Compute the graded GF(2) Morse complex of an acyclic matching",
        description=(
            "Given an acyclic matching on a bounded finite simplicial complex, "
            "return the Morse complex over GF(2): the critical cells graded by "
            "dimension, the reduced boundary rows whose coefficients are the "
            "parities of complete gradient-path counts between critical cells of "
            "adjacent dimension, the boundary-square-zero replay, the Morse Euler "
            "identity, and the reduced Betti numbers. Orientation signs are "
            "unnecessary in characteristic two, where each path contributes "
            "parity one."
        ),
        request_type=MorseComplexRequest,
        result_type=MorseComplexResult,
        run=_run_compute_morse_complex,
        tags=(
            "topology",
            "discrete-morse",
            "morse-complex",
            "critical-cells",
            "betti-numbers",
            "gf2",
            "exact",
        ),
        discovery_terms=(
            "discrete Morse complex",
            "Morse boundary",
            "critical cell grading",
            "Morse homology Betti numbers",
            "gradient path incidence",
        ),
        examples=(
            OperationExample(
                name="interval_boundary_over_gf2",
                description=(
                    "The empty acyclic matching keeps every interval cell "
                    "critical; the GF(2) boundary of the edge lists both "
                    "vertices with coefficient one."
                ),
                input={"complex": _INTERVAL_FACETS, "pairs": []},
            ),
            OperationExample(
                name="circle_morse_complex",
                description=(
                    "The circle matching leaving one critical vertex and one "
                    "critical edge gives a two-cell Morse complex whose Betti "
                    "numbers are (1, 1)."
                ),
                input={"complex": _CIRCLE_FACETS, "pairs": _CIRCLE_MATCHING_PAIRS},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.discrete_morse.integer_complex.compute",
        title="Compute the integral signed Morse chain complex",
        description=(
            "For a supplied acyclic matching on a bounded finite simplicial "
            "complex, return the exact Morse differential over ZZ with the "
            "critical simplices as ordered bases. Simplex orientations use "
            "lexicographic vertex order; each gradient path contributes its "
            "initial simplicial boundary incidence followed by Forman's "
            "orientation transport across matched cells."
        ),
        request_type=IntegerMorseComplexRequest,
        result_type=IntegerMorseComplexResult,
        run=_run_compute_integer_morse_complex,
        tags=("topology", "discrete-morse", "morse-complex", "integer", "exact"),
        discovery_terms=(
            "integral discrete Morse complex",
            "signed Morse differential",
            "Morse boundary over integers",
            "ZZ gradient paths",
        ),
        examples=(
            OperationExample(
                name="interval_integral_boundary",
                description=(
                    "The empty matching on one interval has both endpoints and "
                    "the edge critical; its signed boundary uses lexicographic "
                    "orientations [a,b]."
                ),
                input={"complex": _INTERVAL_FACETS, "pairs": []},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.discrete_morse.chain_contraction.compute",
        title="Construct the integral chain contraction of an acyclic Morse matching",
        description=(
            "For a bounded acyclic matching on a finite simplicial complex, "
            "return the exact inclusion and projection between its oriented "
            "simplicial chains and the critical-cell Morse chains, together "
            "with a degree-raising chain homotopy H satisfying "
            "id - I P = d H + H d. The operation admits only small exact "
            "matrix reductions and rejects cyclic or malformed matchings."
        ),
        request_type=MorseChainContractionRequest,
        result_type=MorseChainContractionResult,
        run=_run_compute_chain_contraction,
        tags=("topology", "discrete-morse", "chain-contraction", "homotopy", "exact"),
        discovery_terms=(
            "Morse chain contraction",
            "discrete Morse chain homotopy",
            "inclusion projection homotopy",
            "strong deformation retract of chain complexes",
        ),
        examples=(
            OperationExample(
                name="interval_contraction_to_one_vertex",
                description=(
                    "Collapse the interval chain complex onto vertex b; the "
                    "result includes maps and an exact chain homotopy."
                ),
                input={
                    "complex": _INTERVAL_FACETS,
                    "pairs": [{"face": ["a"], "coface": ["a", "b"]}],
                },
            ),
            OperationExample(
                name="empty_matching_identity_contraction",
                description=(
                    "With no matched pairs every simplex stays critical and "
                    "the contraction is the identity with zero homotopy."
                ),
                input={"complex": _INTERVAL_FACETS, "pairs": []},
            ),
        ),
    ),
)
