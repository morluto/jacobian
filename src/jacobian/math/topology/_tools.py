"""Finite simplicial topology domain."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology._homology import (
    IntegralSimplicialHomologyRequest,
    IntegralSimplicialHomologyResult,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
)
from jacobian.math.topology._models import (
    BarycentricSubdivisionRequest,
    BarycentricSubdivisionResult,
    ChainComplexRequest,
    ChainComplexResult,
    FiniteSimplicialComplex,
    ShellingCheckRequest,
    ShellingCheckResult,
    SimplicialComplexCanonicalizationResult,
    SimplicialComplexRequest,
)
from jacobian.math.topology._pseudomanifold import (
    PseudomanifoldRequest,
    PseudomanifoldResult,
)
from jacobian.math.topology._structural import (
    BoundaryRequest,
    BoundaryResult,
    ConeRequest,
    ConeResult,
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
    FVectorRequest,
    FVectorResult,
    InducedSubcomplexRequest,
    InducedSubcomplexResult,
    JoinRequest,
    JoinResult,
    LinkRequest,
    LinkResult,
    MinimalNonfacesRequest,
    MinimalNonfacesResult,
    SkeletonRequest,
    SkeletonResult,
    StanleyReisnerIdealRequest,
    StanleyReisnerIdealResult,
    StarRequest,
    StarResult,
    VertexDeletionRequest,
    VertexDeletionResult,
    compute_boundary,
    compute_cone,
    compute_elementary_collapse,
    compute_f_vector,
    compute_induced_subcomplex,
    compute_join,
    compute_link,
    compute_minimal_nonfaces,
    compute_skeleton,
    compute_stanley_reisner_ideal,
    compute_star,
    compute_vertex_deletion,
)
from jacobian.math.topology.operations import (
    barycentric_subdivision as _barycentric_subdivision,
)
from jacobian.math.topology.operations import (
    canonicalize as _canonicalize,
)
from jacobian.math.topology.operations import (
    chain_complex as _chain_complex,
)
from jacobian.math.topology.operations import (
    homology as _homology,
)
from jacobian.math.topology.operations import (
    integral_homology as _integral_homology,
)
from jacobian.math.topology.operations import (
    pseudomanifold as _pseudomanifold,
)
from jacobian.math.topology.operations import (
    shelling_check as _shelling_check,
)
from jacobian.math.topology.release import (
    OrderComplexRequest,
    OrderComplexResult,
    order_complex,
)
from jacobian.math.topology.release_tools import TOOLS as RELEASE_TOOLS

__all__ = ["TOOLS"]


def _canonical_complex(request: SimplicialComplexRequest) -> FiniteSimplicialComplex:
    return _canonicalize(request.vertices, request.facets).complex


def compute_canonicalize(
    request: SimplicialComplexRequest,
) -> SimplicialComplexCanonicalizationResult:
    return _canonicalize(request.vertices, request.facets)


def compute_chain_complex(request: ChainComplexRequest) -> ChainComplexResult:
    return _chain_complex(
        request.complex,
        request.coefficient_ring,
        request.prime,
        request.convention,
    )


def compute_homology(request: SimplicialHomologyRequest) -> SimplicialHomologyResult:
    return _homology(request.complex, request.prime, request.convention)


def compute_integral_homology(
    request: IntegralSimplicialHomologyRequest,
) -> IntegralSimplicialHomologyResult:
    return _integral_homology(request.complex, request.convention)


def compute_barycentric_subdivision(
    request: BarycentricSubdivisionRequest,
) -> BarycentricSubdivisionResult:
    return _barycentric_subdivision(_canonical_complex(request.complex))


def compute_pseudomanifold_decision(
    request: PseudomanifoldRequest,
) -> PseudomanifoldResult:
    return _pseudomanifold(_canonical_complex(request.complex))


def compute_shelling_check(request: ShellingCheckRequest) -> ShellingCheckResult:
    return _shelling_check(_canonical_complex(request.complex), request.facet_order)


def compute_order_complex(request: OrderComplexRequest) -> OrderComplexResult:
    return order_complex(request)


TOPOLOGY_OPERATIONS: MathTools = (
    MathTool(
        operation_id="topology.simplicial_complex.canonicalize",
        title="Canonicalize a finite simplicial complex",
        description=(
            "Validate bounded maximal facets, close them under every non-empty "
            "face, and return canonical oriented simplex bases and the exact "
            "f-vector."
        ),
        request_type=SimplicialComplexRequest,
        result_type=SimplicialComplexCanonicalizationResult,
        run=compute_canonicalize,
        tags=(
            "topology",
            "simplicial-complex",
            "facets",
            "face-closure",
            "f-vector",
            "exact",
        ),
        examples=(
            OperationExample(
                name="triangle_boundary",
                description="Canonicalize the three-edge simplicial model of a circle.",
                input={
                    "vertices": ["a", "b", "c"],
                    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.chain_complex.compute",
        title="Compute an oriented simplicial chain complex",
        description=(
            "Construct every oriented sparse boundary matrix for one canonical "
            "finite simplicial complex over the integers or a bounded prime field."
        ),
        request_type=ChainComplexRequest,
        result_type=ChainComplexResult,
        run=compute_chain_complex,
        tags=(
            "topology",
            "simplicial-complex",
            "chain-complex",
            "boundary-matrix",
            "exact",
        ),
        examples=(
            OperationExample(
                name="circle_integer_chain_complex",
                description="Construct the oriented integer boundary matrices of a circle.",
                input={
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
                        "faces_by_dimension": [
                            {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
                            {
                                "dimension": 1,
                                "faces": [["a", "b"], ["a", "c"], ["b", "c"]],
                            },
                        ],
                        "dimension": 1,
                        "f_vector": [3, 3],
                        "closure_size": 6,
                    },
                    "coefficient_ring": "INTEGER",
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_homology.compute",
        title="Compute finite-field simplicial homology",
        description=(
            "Compute every Betti number and inspectable cycle, boundary, and "
            "quotient basis of a bounded finite simplicial complex over F_p."
        ),
        request_type=SimplicialHomologyRequest,
        result_type=SimplicialHomologyResult,
        run=compute_homology,
        tags=(
            "topology",
            "simplicial-homology",
            "betti-number",
            "cycle-basis",
            "prime-field",
            "exact",
        ),
        examples=(
            OperationExample(
                name="circle_homology_mod_two",
                description="Compute H_0 and H_1 over F_2 for a triangle boundary.",
                input={
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
                        "faces_by_dimension": [
                            {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
                            {
                                "dimension": 1,
                                "faces": [["a", "b"], ["a", "c"], ["b", "c"]],
                            },
                        ],
                        "dimension": 1,
                        "f_vector": [3, 3],
                        "closure_size": 6,
                    },
                    "prime": 2,
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_homology.integral.compute",
        title="Compute transformation-certified integral simplicial homology",
        description=(
            "Compute the chain-owned certified ZZ homology of a finite "
            "simplicial complex: free ranks, torsion invariant factors, "
            "simplex-basis cycles, Smith transformations, and torsion "
            "bounding chains. Reduced homology is the ordinary homology of "
            "the augmented complex with a rank-one group in degree -1."
        ),
        request_type=IntegralSimplicialHomologyRequest,
        result_type=IntegralSimplicialHomologyResult,
        run=compute_integral_homology,
        tags=(
            "topology",
            "simplicial-homology",
            "integer-homology",
            "torsion",
            "betti-number",
            "cycle-generator",
            "smith-normal-form",
            "certificate",
            "exact",
        ),
        examples=(
            OperationExample(
                name="integral_circle_homology",
                description="Compute H_0 and H_1 over the integers for a triangle boundary.",
                input={
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
                        "faces_by_dimension": [
                            {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
                            {
                                "dimension": 1,
                                "faces": [["a", "b"], ["a", "c"], ["b", "c"]],
                            },
                        ],
                        "dimension": 1,
                        "f_vector": [3, 3],
                        "closure_size": 6,
                    }
                },
            ),
        ),
    ),
)

_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}

_CANONICAL_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
    "faces_by_dimension": [
        {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
        {"dimension": 1, "faces": [["a", "b"], ["a", "c"], ["b", "c"]]},
    ],
    "dimension": 1,
    "f_vector": [3, 3],
    "closure_size": 6,
}

_f_vector_tool = MathTool(
    operation_id="topology.simplicial_complex.f_vector.compute",
    title="Compute the f-vector and h-vector of a simplicial complex",
    description=(
        "Compute the f-vector (face counts by dimension) and h-vector "
        "of a finite simplicial complex, with Euler characteristic."
    ),
    request_type=FVectorRequest,
    result_type=FVectorResult,
    run=compute_f_vector,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="triangle_f_vector",
            description="Compute f-vector of a triangle (3 vertices, 3 edges, 1 face); "
            "facets must be a list of simplices.",
            input={
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                }
            },
        ),
    ),
)

_link_tool = MathTool(
    operation_id="topology.simplicial_complex.link.compute",
    title="Compute the link of a simplex",
    description=(
        "Compute the link of a simplex in a finite simplicial complex and return "
        "the maximal facets of the resulting link complex."
    ),
    request_type=LinkRequest,
    result_type=LinkResult,
    run=compute_link,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="link_of_vertex_in_triangle",
            description="Compute the link of one vertex in a triangle.",
            input={
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                },
                "simplex": ["v0"],
            },
        ),
    ),
)

_star_tool = MathTool(
    operation_id="topology.simplicial_complex.star.compute",
    title="Compute the closed star of a simplex",
    description=(
        "Compute the closed star of a simplex sigma in a finite simplicial "
        "complex: all facets of the complex that contain sigma."
    ),
    request_type=StarRequest,
    result_type=StarResult,
    run=compute_star,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="star_of_vertex_in_triangle",
            description="Compute the star of one vertex in a triangle.",
            input={
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                },
                "simplex": ["v0"],
            },
        ),
    ),
)

_vertex_deletion_tool = MathTool(
    operation_id="topology.simplicial_complex.deletion.compute",
    title="Compute the deletion of a vertex subset",
    description=(
        "Delete a vertex subset from a finite simplicial complex and return "
        "the induced subcomplex on the remaining vertices: every face "
        "disjoint from the deleted set, given by its maximal facets. The "
        "deletion must leave at least one simplex on the remaining "
        "vertices; deleting every vertex is out of contract."
    ),
    request_type=VertexDeletionRequest,
    result_type=VertexDeletionResult,
    run=compute_vertex_deletion,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="delete_vertex_from_triangle",
            description="Delete one vertex from a triangle, leaving the opposite edge; "
            "the deletion must leave at least one simplex.",
            input={
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                },
                "vertices_to_delete": ["v2"],
            },
        ),
    ),
)

_induced_subcomplex_tool = MathTool(
    operation_id="topology.simplicial_complex.induced_subcomplex.compute",
    title="Compute an induced subcomplex on selected vertices",
    description=(
        "Take the full subcomplex on a nonempty selected vertex subset of a "
        "canonical finite simplicial complex. Return its exact face closure "
        "and the image or deletion status of every source face."
    ),
    request_type=InducedSubcomplexRequest,
    result_type=InducedSubcomplexResult,
    run=compute_induced_subcomplex,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="induce_edge_from_triangle",
            description="Restrict a filled triangle to two of its vertices.",
            input={
                "complex": {
                    "vertices": ["a", "b", "c"],
                    "maximal_simplices": [["a", "b", "c"]],
                    "faces_by_dimension": [
                        {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
                        {
                            "dimension": 1,
                            "faces": [["a", "b"], ["a", "c"], ["b", "c"]],
                        },
                        {"dimension": 2, "faces": [["a", "b", "c"]]},
                    ],
                    "f_vector": [3, 3, 1],
                    "dimension": 2,
                    "closure_size": 7,
                },
                "selected_vertices": ["a", "b"],
            },
        ),
    ),
)

_skeleton_tool = MathTool(
    operation_id="topology.simplicial_complex.skeleton.compute",
    title="Compute the k-skeleton of a simplicial complex",
    description=(
        "Compute the k-skeleton of a finite simplicial complex: the subcomplex "
        "consisting of all faces of dimension at most k."
    ),
    request_type=SkeletonRequest,
    result_type=SkeletonResult,
    run=compute_skeleton,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="one_skeleton_of_triangle",
            description="Compute the 1-skeleton of a triangle (all edges).",
            input={
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                },
                "k": 1,
            },
        ),
    ),
)

_join_tool = MathTool(
    operation_id="topology.simplicial_complex.join.compute",
    title="Compute the join of two simplicial complexes",
    description=(
        "Compute the join of two simplicial complexes on disjoint vertex sets: "
        "the facets are unions of a facet from each complex."
    ),
    request_type=JoinRequest,
    result_type=JoinResult,
    run=compute_join,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="join_of_two_points",
            description="Join two single-vertex complexes (a point join a point is an edge).",
            input={
                "complex_a": {
                    "vertices": ["a"],
                    "facets": [["a"]],
                },
                "complex_b": {
                    "vertices": ["b"],
                    "facets": [["b"]],
                },
            },
        ),
    ),
)

_barycentric_subdivision_tool = MathTool(
    operation_id="topology.simplicial_complex.barycentric_subdivision.compute",
    title="Compute the barycentric subdivision of a simplicial complex",
    description=(
        "Compute the barycentric subdivision (order complex) of a finite "
        "simplicial complex: new vertices are nonempty faces, new simplices "
        "are strict face chains. Each output facet carries its exact source-face "
        "chain. The complete subdivision must fit 128 maximal chains and 2048 "
        "nonempty faces."
    ),
    request_type=BarycentricSubdivisionRequest,
    result_type=BarycentricSubdivisionResult,
    run=compute_barycentric_subdivision,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="barycentric_subdivision_of_edge",
            description="Subdivide an edge into two edges.",
            input={
                "complex": {
                    "vertices": ["a", "b"],
                    "facets": [["a", "b"]],
                },
            },
        ),
    ),
)

_pseudomanifold_tool = MathTool(
    operation_id="topology.simplicial_complex.pseudomanifold.decide",
    title="Decide whether a complex is a pseudomanifold",
    description=(
        "Decide whether a finite simplicial complex is a pseudomanifold: "
        "pure, every codimension-1 face in exactly 1 or 2 facets. "
        "Reports closed vs. with-boundary."
    ),
    request_type=PseudomanifoldRequest,
    result_type=PseudomanifoldResult,
    run=compute_pseudomanifold_decision,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="circle_is_pseudomanifold",
            description="Check that a triangle boundary is a closed pseudomanifold.",
            input={"complex": _CIRCLE},
        ),
    ),
)

_shelling_check_tool = MathTool(
    operation_id="topology.simplicial_complex.shelling.check",
    title="Check a submitted shelling order",
    description=(
        "Check whether a submitted facet order is a valid shelling order "
        "for a pure finite simplicial complex."
    ),
    request_type=ShellingCheckRequest,
    result_type=ShellingCheckResult,
    run=compute_shelling_check,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="valid_shelling_of_edge",
            description="Check a valid shelling of a single edge.",
            input={
                "complex": {
                    "vertices": ["a", "b"],
                    "facets": [["a", "b"]],
                },
                "facet_order": [0],
            },
        ),
    ),
)

_elementary_collapse_tool = MathTool(
    operation_id="topology.simplicial_complex.elementary_collapse.check",
    title="Check and perform an elementary collapse",
    description=(
        "Verify that a free face is contained in exactly one coface facet, "
        "then remove both the free face and the coface from the complex."
    ),
    request_type=ElementaryCollapseRequest,
    result_type=ElementaryCollapseResult,
    run=compute_elementary_collapse,
    tags=("topology", "simplicial", "exact"),
    examples=(
        OperationExample(
            name="collapse_edge_endpoint",
            description="Collapse a free vertex from an edge (the vertex is free, the edge is the coface).",
            input={
                "complex": {
                    "vertices": ["a", "b"],
                    "facets": [["a", "b"]],
                },
                "free_face": ["a"],
                "coface": ["a", "b"],
            },
        ),
    ),
)


def compute_cone_entry(request: ConeRequest) -> ConeResult:
    return compute_cone(request)


_cone_tool = MathTool(
    operation_id="topology.simplicial_complex.cone.compute",
    title="Compute the cone over a simplicial complex",
    description=(
        "Join one bounded finite simplicial complex with a single fresh tagged "
        "apex vertex: every cone facet is a source facet plus the apex, the "
        "cone dimension is one higher, and every nonempty source face carries "
        "exact transport to its cone face. The apex must lie outside the "
        "source vertex domain; deleting it returns the source complex."
    ),
    request_type=ConeRequest,
    result_type=ConeResult,
    run=compute_cone_entry,
    tags=("topology", "simplicial", "cone", "exact"),
    discovery_terms=("simplicial cone", "cone apex"),
    examples=(
        OperationExample(
            name="cone_over_an_edge",
            description="Cone over a single edge: one triangular facet with a fresh apex.",
            input={
                "complex": {
                    "vertices": ["a", "b"],
                    "facets": [["a", "b"]],
                },
                "apex": "cone_apex",
            },
        ),
    ),
)


def compute_boundary_entry(request: BoundaryRequest) -> BoundaryResult:
    return compute_boundary(request)


_boundary_tool = MathTool(
    operation_id="topology.simplicial.boundary.compute",
    title="Compute the boundary of a pseudomanifold with boundary",
    description=(
        "Check one bounded finite simplicial complex is a pure pseudomanifold "
        "with boundary (every codimension-one face in one or two top facets, "
        "at least one ridge of incidence one) and return the exact boundary "
        "subcomplex generated by the incidence-one ridges with its component "
        "count. Closed pseudomanifolds and non-pseudomanifolds are domain "
        "rejects, not empty boundaries."
    ),
    request_type=BoundaryRequest,
    result_type=BoundaryResult,
    run=compute_boundary_entry,
    tags=("topology", "simplicial", "boundary", "exact"),
    discovery_terms=("simplicial boundary", "pseudomanifold boundary"),
    examples=(
        OperationExample(
            name="boundary_of_an_interval",
            description="Boundary of a two-edge path: its two endpoint vertices.",
            input={
                "complex": {
                    "vertices": ["a", "b", "c"],
                    "facets": [["a", "b"], ["b", "c"]],
                },
            },
        ),
    ),
)

_minimal_nonfaces_tool = MathTool(
    operation_id="topology.simplicial_complex.minimal_nonfaces.compute",
    title="Enumerate minimal nonfaces of a finite simplicial complex",
    description=(
        "Return every inclusion-minimal vertex subset absent from a canonical "
        "finite simplicial complex, as a source-bound antichain in source "
        "vertex order. The exact powerset, immediate-subface work, source "
        "closure, and output are preflight-bounded; the current carrier admits "
        "at most 14 vertices for this enumeration."
    ),
    request_type=MinimalNonfacesRequest,
    result_type=MinimalNonfacesResult,
    run=compute_minimal_nonfaces,
    tags=("topology", "simplicial", "minimal-nonfaces", "antichain", "exact"),
    discovery_terms=(
        "minimal nonfaces of a simplicial complex",
        "forbidden faces",
        "minimal generators of a Stanley-Reisner ideal",
    ),
    examples=(
        OperationExample(
            name="boundary_triangle_minimal_nonface",
            description=(
                "The three edges form the boundary of a triangle; the only "
                "minimal nonface is the full three-vertex set."
            ),
            input={"complex": _CANONICAL_CIRCLE},
        ),
    ),
)

_stanley_reisner_ideal_tool = MathTool(
    operation_id="topology.simplicial_complex.stanley_reisner_ideal.compute",
    title="Construct the Stanley-Reisner ideal of a finite simplicial complex",
    description=(
        "Return the exact squarefree monomial ideal generated by all minimal "
        "nonfaces, with an explicit ordered binding from source vertex IDs to "
        "collision-free polynomial variables. The current polynomial carrier "
        "admits at most eight vertices and 64 generators; a simplex returns "
        "the canonical zero ideal. Enumeration and result size are bounded."
    ),
    request_type=StanleyReisnerIdealRequest,
    result_type=StanleyReisnerIdealResult,
    run=compute_stanley_reisner_ideal,
    tags=("topology", "simplicial", "Stanley-Reisner", "monomial-ideal", "exact"),
    discovery_terms=(
        "Stanley-Reisner ideal of a simplicial complex",
        "minimal nonfaces squarefree monomial generators",
    ),
    examples=(
        OperationExample(
            name="triangle_boundary_ideal",
            description=(
                "The boundary of a triangle has one minimal nonface, "
                "so its ideal is generated by v0*v1*v2."
            ),
            input={"complex": _CANONICAL_CIRCLE},
        ),
    ),
)

_order_complex_tool = MathTool(
    operation_id="topology.poset.order_complex.compute",
    title="Compute the order complex of a finite poset",
    description=(
        "Return the canonical simplicial complex whose vertices retain the "
        "poset element labels and whose faces are all nonempty chains. The "
        "result includes the source poset and each maximal chain as a facet. "
        "Exact chain, dimension, facet, work, and output bounds are checked "
        "before chain enumeration."
    ),
    request_type=OrderComplexRequest,
    result_type=OrderComplexResult,
    run=compute_order_complex,
    tags=("topology", "poset", "order-complex", "simplicial", "exact"),
    discovery_terms=(
        "finite poset order complex",
        "simplicial complex of chains",
        "barycentric subdivision of a face poset",
    ),
    examples=(
        OperationExample(
            name="two_element_chain_order_complex",
            description="A two-element chain gives one edge and its two vertices.",
            input={
                "poset": {
                    "elements": ["a", "b"],
                    "strict_order_pairs": [{"lower": "a", "upper": "b"}],
                    "cover_relations": [{"lower": "a", "upper": "b"}],
                    "incomparable_pairs": [],
                    "minimal_elements": ["a"],
                    "maximal_elements": ["b"],
                    "graded": True,
                    "ranks": [
                        {"element": "a", "rank": 0},
                        {"element": "b", "rank": 1},
                    ],
                    "poset_digest": "sha256:501fec56e614e0c8e24b67bf7ad4f4872124cc902e4279671a0dbcc6a0999872",
                }
            },
        ),
    ),
)

TOOLS: MathTools = (
    *TOPOLOGY_OPERATIONS,
    *RELEASE_TOOLS,
    _cone_tool,
    _boundary_tool,
    _minimal_nonfaces_tool,
    _stanley_reisner_ideal_tool,
    _f_vector_tool,
    _link_tool,
    _star_tool,
    _vertex_deletion_tool,
    _induced_subcomplex_tool,
    _skeleton_tool,
    _join_tool,
    _barycentric_subdivision_tool,
    _pseudomanifold_tool,
    _shelling_check_tool,
    _elementary_collapse_tool,
    _order_complex_tool,
)
