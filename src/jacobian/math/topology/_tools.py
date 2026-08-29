"""Finite simplicial topology domain."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
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
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
    FVectorRequest,
    FVectorResult,
    JoinRequest,
    JoinResult,
    LinkRequest,
    LinkResult,
    SkeletonRequest,
    SkeletonResult,
    StarRequest,
    StarResult,
    VertexDeletionRequest,
    VertexDeletionResult,
    compute_elementary_collapse,
    compute_f_vector,
    compute_join,
    compute_link,
    compute_skeleton,
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
            example(
                "triangle_boundary",
                "Canonicalize the three-edge simplicial model of a circle.",
                {
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
            example(
                "circle_integer_chain_complex",
                "Construct the oriented integer boundary matrices of a circle.",
                {
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
                        "complex_digest": "sha256:0cfbfd8d7c8d23a25d567cd58726d913d44d1e2c7302f86dbe78a6e9e46f1647",
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
            example(
                "circle_homology_mod_two",
                "Compute H_0 and H_1 over F_2 for a triangle boundary.",
                {
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
                        "complex_digest": "sha256:0cfbfd8d7c8d23a25d567cd58726d913d44d1e2c7302f86dbe78a6e9e46f1647",
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
            "Compute the free rank, torsion invariant factors, and simplex-basis "
            "cycle generators of every integral homology group, with explicit "
            "Smith transformations and bounding chains. Each chain group is "
            "bounded by the certified Smith-certificate dimension."
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
            example(
                "integral_circle_homology",
                "Compute H_0 and H_1 over the integers for a triangle boundary.",
                {
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
                        "complex_digest": "sha256:0cfbfd8d7c8d23a25d567cd58726d913d44d1e2c7302f86dbe78a6e9e46f1647",
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
    "complex_digest": (
        "sha256:0cfbfd8d7c8d23a25d567cd58726d913d44d1e2c7302f86dbe78a6e9e46f1647"
    ),
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
        example(
            "triangle_f_vector",
            "Compute f-vector of a triangle (3 vertices, 3 edges, 1 face); "
            "facets must be a list of simplices.",
            {
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
        example(
            "link_of_vertex_in_triangle",
            "Compute the link of one vertex in a triangle.",
            {
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
        example(
            "star_of_vertex_in_triangle",
            "Compute the star of one vertex in a triangle.",
            {
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
        example(
            "delete_vertex_from_triangle",
            "Delete one vertex from a triangle, leaving the opposite edge; "
            "the deletion must leave at least one simplex.",
            {
                "complex": {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1", "v2"]],
                },
                "vertices_to_delete": ["v2"],
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
        example(
            "one_skeleton_of_triangle",
            "Compute the 1-skeleton of a triangle (all edges).",
            {
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
        example(
            "join_of_two_points",
            "Join two single-vertex complexes (a point join a point is an edge).",
            {
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
        "are strict face chains."
    ),
    request_type=BarycentricSubdivisionRequest,
    result_type=BarycentricSubdivisionResult,
    run=compute_barycentric_subdivision,
    tags=("topology", "simplicial", "exact"),
    examples=(
        example(
            "barycentric_subdivision_of_edge",
            "Subdivide an edge into two edges.",
            {
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
        example(
            "circle_is_pseudomanifold",
            "Check that a triangle boundary is a closed pseudomanifold.",
            {"complex": _CIRCLE},
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
        example(
            "valid_shelling_of_edge",
            "Check a valid shelling of a single edge.",
            {
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
        example(
            "collapse_edge_endpoint",
            "Collapse a free vertex from an edge (the vertex is free, the edge is the coface).",
            {
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

TOOLS: MathTools = (
    *TOPOLOGY_OPERATIONS,
    _f_vector_tool,
    _link_tool,
    _star_tool,
    _vertex_deletion_tool,
    _skeleton_tool,
    _join_tool,
    _barycentric_subdivision_tool,
    _pseudomanifold_tool,
    _shelling_check_tool,
    _elementary_collapse_tool,
)
