# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cellular_sheaves.extensions import *


def _sections(r: Any) -> Any:
    return sections(r.sheaf)


def _restriction(r: Any) -> Any:
    return restriction(r.sheaf, r.source, r.target)


def _morphism(r: Any) -> Any:
    return morphism(r.source, r.target, r.components)


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
            "entries": [["1"]],
            "cover_path": [["a"], ["a", "b"]],
        },
        {
            "source": ["b"],
            "target": ["a", "b"],
            "row_basis": ["x"],
            "column_basis": ["x"],
            "entries": [["1"]],
            "cover_path": [["b"], ["a", "b"]],
        },
    ],
    "derived_restrictions": [],
    "diamonds": 0,
    "comparable_pairs": 2,
}
TOOLS = (
    MathTool(
        operation_id="cellular_sheaf.sections.compute",
        title="Compute global sections of a cellular sheaf",
        description="Compute H^0 as the exact vector space of compatible stalk assignments, retaining its sheaf and cochain axes.",
        request_type=SheafSectionsRequest,
        result_type=SheafSectionsResult,
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
                    "components": [["a", [["1"]]], ["b", [["1"]]], ["a.b", [["1"]]]],
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
