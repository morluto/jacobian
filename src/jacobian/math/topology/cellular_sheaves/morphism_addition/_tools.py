"""Catalog declaration for exact cellular-sheaf morphism addition."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cellular_sheaves.extensions import SheafMorphismResult
from jacobian.math.topology.cellular_sheaves.morphism_add import (
    SheafMorphismAddRequest,
    add_morphisms,
)


def _run(request: SheafMorphismAddRequest) -> SheafMorphismResult:
    return add_morphisms(request.left, request.right)


_SHEAF = {
    "complex": {
        "vertices": ["v"],
        "maximal_simplices": [["v"]],
        "faces_by_dimension": [{"dimension": 0, "faces": [["v"]]}],
        "dimension": 0,
        "f_vector": [1],
        "closure_size": 1,
    },
    "coefficient_field": "QQ",
    "prime": None,
    "stalks": [{"simplex": ["v"], "basis": ["e"]}],
    "cover_restrictions": [],
    "derived_restrictions": [],
    "diamonds": 0,
    "comparable_pairs": 0,
}

_MORPHISM = {
    "source": _SHEAF,
    "target": _SHEAF,
    "components": [[["v"], [[{"num": "1", "den": "1"}]]]],
    "natural": True,
    "obstruction": None,
}

TOOLS = (
    MathTool(
        operation_id="cellular_sheaf.morphism.add.compute",
        title="Add natural cellular-sheaf morphisms",
        description=(
            "Add two natural cellular-sheaf morphisms pointwise. They must have "
            "exactly equal source and target sheaves over the same coefficient field."
        ),
        request_type=SheafMorphismAddRequest,
        result_type=SheafMorphismResult,
        run=_run,
        tags=("topology", "cellular-sheaf", "morphism", "addition", "exact"),
        discovery_terms=("sum of sheaf morphisms", "add cellular sheaf maps"),
        examples=(
            OperationExample(
                name="add_identity_endomorphisms",
                description=(
                    "Add two identity maps on the one-point rank-one sheaf; both "
                    "maps must have exactly the same source and target."
                ),
                input={"left": _MORPHISM, "right": _MORPHISM},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
