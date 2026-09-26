from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.edge_paths._models import (
    FundamentalGroupBasepointChangeRequest,
    FundamentalGroupMapRequest,
    FundamentalGroupMapResult,
    PresentationMapCompositionRequest,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    DirectRelatorMatchRequest,
    DirectRelatorMatchResult,
    change_fundamental_group_basepoint,
    compose_fundamental_group_maps,
    direct_relator_match,
    induced_fundamental_group_map,
)


def _run_basepoint_change(request: FundamentalGroupBasepointChangeRequest) -> Any:
    return change_fundamental_group_basepoint(request)


def _run(r: Any) -> Any:
    return direct_relator_match(r.source, r.target, r.generator_images)


def _run_induced(r: Any) -> Any:
    return induced_fundamental_group_map(r)


def _run_compose(r: Any) -> Any:
    return compose_fundamental_group_maps(r)


_P = {"generators": ["x"], "relators": []}
_CIRCLE = canonical_complex(
    ("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))
).model_dump(mode="json")
_POINT = {
    "vertices": ["a"],
    "maximal_simplices": [["a"]],
    "faces_by_dimension": [{"dimension": 0, "faces": [["a"]]}],
    "dimension": 0,
    "f_vector": [1],
    "closure_size": 1,
    "orientation_convention": "LEXICOGRAPHIC_VERTEX_ORDER",
    "empty_simplex_stored": False,
}
_TRIVIAL_GROUP_PRESENTATION = {
    "generators": [],
    "relators": [],
}
_TRIVIAL_PRESENTATION_RESULT = {
    "complex": _POINT,
    "base_vertex": "a",
    "component_vertices": ["a"],
    "spanning_tree_edges": [],
    "non_tree_edges": [],
    "edge_words": [],
    "triangle_relators": [],
    "presentation": _TRIVIAL_GROUP_PRESENTATION,
    "abelianization": {
        "relation_matrix": {
            "domain": "ZZ",
            "row_count": 0,
            "column_count": 0,
            "entries": [],
        },
        "rank": 0,
        "free_rank": 0,
        "torsion_invariant_factors": [],
    },
}
_TRIVIAL_BASEPOINT_MAP = {
    "map": {
        "complex": _POINT,
        "source_base_vertex": "a",
        "target_base_vertex": "a",
        "path_vertices": ["a"],
    },
    "source_presentation": _TRIVIAL_PRESENTATION_RESULT,
    "target_presentation": _TRIVIAL_PRESENTATION_RESULT,
    "generator_images": [],
    "abelianization_map": {
        "domain": "ZZ",
        "row_count": 0,
        "column_count": 0,
        "entries": [],
    },
    "relator_images": [],
}
TOOLS = (
    MathTool(
        operation_id="topology.group_presentation.direct_relator_match.compute",
        title="Check direct relator-match compatibility",
        description="Return a complete finite witness for the sufficient condition that every source relator maps literally to a target relator or its inverse. Absence of this witness does not decide normal-closure membership or deny a presentation homomorphism.",
        request_type=DirectRelatorMatchRequest,
        result_type=DirectRelatorMatchResult,
        run=_run,
        tags=("topology", "presentation", "relator-match", "exact"),
        examples=(
            OperationExample(
                name="free_generator_identity",
                description="Map the one-generator free presentation to itself; the presentation has no relators to preserve.",
                input={
                    "source": _P,
                    "target": _P,
                    "generator_images": [
                        {"letters": [{"generator": 0, "exponent": 1}]}
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.fundamental_group.map.compose.compute",
        title="Compose based fundamental-group presentation maps",
        description=(
            "Compose two exact induced presentation maps when the complete middle "
            "presentation carriers agree. Return reduced outer-axis generator words "
            "and the corresponding integer matrix on abelianizations."
        ),
        request_type=PresentationMapCompositionRequest,
        result_type=FundamentalGroupMapResult,
        run=_run_compose,
        tags=("topology", "fundamental-group", "composition", "exact"),
        discovery_terms=(
            "compose induced fundamental group maps",
            "compose presentation homomorphisms",
            "fundamental group functoriality",
        ),
        examples=(
            OperationExample(
                name="compose_identity_basepoint_paths",
                description=(
                    "Compose two identity paths at the sole vertex of a point; "
                    "the composite is the same based path map."
                ),
                input={
                    "first": _TRIVIAL_BASEPOINT_MAP,
                    "second": _TRIVIAL_BASEPOINT_MAP,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.fundamental_group.induced_map.compute",
        title="Construct the induced fundamental-group presentation map",
        description=(
            "Given an exact simplicial vertex map and compatible source/target "
            "basepoints, derive each source generator image using the canonical "
            "spanning trees, together with its exact generator-axis matrix on "
            "abelianizations. Return a target-relator conjugacy witness for every "
            "source triangle relation; degenerate triangle images freely reduce "
            "to the identity."
        ),
        request_type=FundamentalGroupMapRequest,
        result_type=FundamentalGroupMapResult,
        run=_run_induced,
        tags=("topology", "fundamental-group", "simplicial-map", "exact"),
        discovery_terms=(
            "induced fundamental group map",
            "simplicial map on pi1 presentations",
            "generator word images",
        ),
        examples=(
            OperationExample(
                name="circle_identity_fundamental_group_map",
                description=(
                    "The identity simplicial map of a 3-edge circle sends its "
                    "canonical fundamental-group generator to itself."
                ),
                input={
                    "map": {
                        "source": _CIRCLE,
                        "target": _CIRCLE,
                        "vertex_map": ["a", "b", "c"],
                    },
                    "source_base_vertex": "a",
                    "target_base_vertex": "a",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.fundamental_group.basepoint_change.compute",
        title="Transport a fundamental-group presentation along an edge path",
        description=(
            "Given one finite simplicial complex and an explicit edge path p "
            "from basepoint b0 to b1, return the based isomorphism sending each "
            "loop a at b0 to p^-1 a p at b1. The result retains the path, exact "
            "source and target presentations, generator words, target-relator "
            "conjugacy witnesses, and the induced integer abelianization map. "
            "The typed morphism composes with other based presentation maps."
        ),
        request_type=FundamentalGroupBasepointChangeRequest,
        result_type=FundamentalGroupMapResult,
        run=_run_basepoint_change,
        tags=("topology", "fundamental-group", "basepoint-change", "exact"),
        discovery_terms=(
            "fundamental group change of basepoint path",
            "basepoint transport isomorphism on pi1",
            "conjugate loop under a basepoint path",
        ),
        examples=(
            OperationExample(
                name="circle_basepoint_change",
                description=(
                    "Transport the canonical fundamental-group generator of a "
                    "3-edge circle along one edge."
                ),
                input={
                    "path": {
                        "complex": _CIRCLE,
                        "source_base_vertex": "a",
                        "target_base_vertex": "b",
                        "path_vertices": ["a", "b"],
                    }
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
