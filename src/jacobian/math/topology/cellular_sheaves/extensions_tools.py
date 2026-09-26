# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cellular_sheaves._models import (
    SheafSectionRestriction,
    SheafSectionSpace,
)
from jacobian.math.topology.cellular_sheaves.direct_sum import (
    SheafDirectSumRequest,
    SheafDirectSumResult,
    direct_sum,
)
from jacobian.math.topology.cellular_sheaves.extensions import *
from jacobian.math.topology.cellular_sheaves.hodge import (
    SheafHodgeRequest,
    SheafHodgeResult,
    compute_hodge,
)


def _sections(r: Any) -> Any:
    return sections(r.sheaf)


def _restrict_sections(r: Any) -> Any:
    return restrict_sections(r.sheaf, r.subcomplex)


def _restriction(r: Any) -> Any:
    return restriction(r.sheaf, r.source, r.target)


def _morphism(r: Any) -> Any:
    return morphism(r.source, r.target, r.components)


def _compose_morphisms(r: Any) -> Any:
    return compose_morphisms(r.first, r.second)


def _hodge(r: SheafHodgeRequest) -> SheafHodgeResult:
    return compute_hodge(r)


def _direct_sum(r: SheafDirectSumRequest) -> SheafDirectSumResult:
    return direct_sum(r.left, r.right)


_S = {
    "complex": {
        "vertices": ["a", "b"],
        "maximal_simplices": [["a", "b"]],
        "faces_by_dimension": [
            {"dimension": 0, "faces": [["a"], ["b"]]},
            {"dimension": 1, "faces": [["a", "b"]]},
        ],
        "dimension": 1,
        "f_vector": [2, 1],
        "closure_size": 3,
    },
    "coefficient_field": "QQ",
    "prime": None,
    "stalks": [
        {"simplex": ["a"], "basis": ["x"]},
        {"simplex": ["b"], "basis": ["x"]},
        {"simplex": ["a", "b"], "basis": ["x"]},
    ],
    "cover_restrictions": [
        {
            "source": ["a"],
            "target": ["a", "b"],
            "row_basis": ["x"],
            "column_basis": ["x"],
            "entries": [[{"num": "1", "den": "1"}]],
            "cover_path": [["a"], ["a", "b"]],
        },
        {
            "source": ["b"],
            "target": ["a", "b"],
            "row_basis": ["x"],
            "column_basis": ["x"],
            "entries": [[{"num": "1", "den": "1"}]],
            "cover_path": [["b"], ["a", "b"]],
        },
    ],
    "derived_restrictions": [],
    "diamonds": 0,
    "comparable_pairs": 2,
}
TOOLS = (
    MathTool(
        operation_id="cellular_sheaf.direct_sum.compute",
        title="Form the pointwise direct sum of two cellular sheaves",
        description=(
            "Form the exact block direct sum of two sheaves on the same finite "
            "simplicial complex and coefficient field. Return the sum sheaf "
            "and canonical stalkwise injections from both source sheaves."
        ),
        request_type=SheafDirectSumRequest,
        result_type=SheafDirectSumResult,
        run=_direct_sum,
        tags=("topology", "cellular-sheaf", "direct-sum", "exact"),
        examples=(
            OperationExample(
                name="interval_rank_one_direct_sum",
                description="Sum two constant rank-one sheaves on an interval, retaining both stalk injections.",
                input={"left": _S, "right": _S},
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.sections.compute",
        title="Compute global sections of a cellular sheaf",
        description=(
            "Return the exact kernel of the full stalk-compatibility matrix, "
            "with source-bound ambient and equation axes, a basis of compatible "
            "stalk assignments, and evaluation maps into every stalk."
        ),
        request_type=SheafSectionsRequest,
        result_type=SheafSectionSpace,
        run=_sections,
        tags=("topology", "cellular-sheaf", "sections", "exact"),
        examples=(
            OperationExample(
                name="interval_sections",
                description="Compute sections of the constant rank-one sheaf on an interval; compatible vertex values must agree in the edge stalk.",
                input={"sheaf": _S},
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.sections.restrict",
        title="Restrict global sections to an included subcomplex",
        description=(
            "Return the exact linear map on section spaces induced by inclusion "
            "of a finite simplicial subcomplex, with both parent sheaves and "
            "their canonical section bases retained."
        ),
        request_type=SheafSectionRestrictionRequest,
        result_type=SheafSectionRestriction,
        run=_restrict_sections,
        tags=("topology", "cellular-sheaf", "sections", "restriction", "exact"),
        examples=(
            OperationExample(
                name="interval_sections_to_vertices",
                description=(
                    "Restrict the one-dimensional constant section space of an "
                    "interval to its two isolated vertices."
                ),
                input={
                    "sheaf": _S,
                    "subcomplex": {
                        "vertices": ["a", "b"],
                        "maximal_simplices": [["a"], ["b"]],
                        "faces_by_dimension": [
                            {"dimension": 0, "faces": [["a"], ["b"]]}
                        ],
                        "dimension": 0,
                        "f_vector": [2],
                        "closure_size": 2,
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.restriction.compute",
        title="Read a sheaf restriction map",
        description="Return the canonical exact restriction matrix for one comparable face pair from a checked cellular sheaf.",
        request_type=SheafRestrictionRequest,
        result_type=SheafRestrictionResult,
        run=_restriction,
        tags=("topology", "cellular-sheaf", "restriction", "exact"),
        examples=(
            OperationExample(
                name="vertex_to_edge",
                description="Read the restriction from vertex a to edge ab; both faces must be in the sheaf diagram.",
                input={"sheaf": _S, "source": ["a"], "target": ["a", "b"]},
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.morphism.compute",
        title="Check naturality of a cellular-sheaf morphism",
        description="Check component maps against every cover restriction square and return a source-bound naturality result.",
        request_type=SheafMorphismRequest,
        result_type=SheafMorphismResult,
        run=_morphism,
        tags=("topology", "cellular-sheaf", "morphism", "naturality", "exact"),
        examples=(
            OperationExample(
                name="identity_morphism",
                description="Check the identity morphism of the interval rank-one sheaf; one component is required for each stalk in canonical order.",
                input={
                    "source": _S,
                    "target": _S,
                    "components": [
                        ["a", [[{"num": "1", "den": "1"}]]],
                        ["b", [[{"num": "1", "den": "1"}]]],
                        ["a.b", [[{"num": "1", "den": "1"}]]],
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.morphism.compose",
        title="Compose cellular-sheaf morphisms",
        description=(
            "Compose two serialized source-bound natural morphisms over the "
            "same exact field and cell complex, returning the pointwise exact "
            "component maps with the original and final parent sheaves."
        ),
        request_type=SheafMorphismComposeRequest,
        result_type=SheafMorphismResult,
        run=_compose_morphisms,
        tags=("topology", "cellular-sheaf", "morphism", "composition", "exact"),
        examples=(
            OperationExample(
                name="compose_interval_identity_maps",
                description=(
                    "Compose two serialized identity maps of the checked "
                    "interval rank-one sheaf."
                ),
                input={
                    "first": {
                        "source": _S,
                        "target": _S,
                        "components": [
                            [["a"], [[{"num": "1", "den": "1"}]]],
                            [["b"], [[{"num": "1", "den": "1"}]]],
                            [["a", "b"], [[{"num": "1", "den": "1"}]]],
                        ],
                        "natural": True,
                        "obstruction": None,
                    },
                    "second": {
                        "source": _S,
                        "target": _S,
                        "components": [
                            [["a"], [[{"num": "1", "den": "1"}]]],
                            [["b"], [[{"num": "1", "den": "1"}]]],
                            [["a", "b"], [[{"num": "1", "den": "1"}]]],
                        ],
                        "natural": True,
                        "obstruction": None,
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cellular_sheaf.hodge_laplacians.compute",
        title="Compute exact cellular-sheaf Hodge Laplacians",
        description=(
            "For a rational cellular sheaf, use the standard positive-definite "
            "identity Gram matrix on each stalk and return the exact degreewise "
            "Hodge Laplacian and its harmonic basis in the source-bound cochain "
            "axes. Harmonic dimensions are checked against cellular sheaf "
            "cohomology."
        ),
        request_type=SheafHodgeRequest,
        result_type=SheafHodgeResult,
        run=_hodge,
        tags=("topology", "cellular-sheaf", "hodge-theory", "laplacian", "exact"),
        examples=(
            OperationExample(
                name="interval_constant_sheaf_hodge",
                description=(
                    "Compute the exact Hodge Laplacians of the constant rank-one "
                    "sheaf on an interval; the degree-zero harmonic space is "
                    "one-dimensional and the degree-one harmonic space is zero."
                ),
                input={"sheaf": _S},
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
