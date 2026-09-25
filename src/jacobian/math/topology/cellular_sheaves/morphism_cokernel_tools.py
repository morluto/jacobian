"""Catalog operation for pointwise cellular-sheaf morphism cokernels."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cellular_sheaves.morphism_cokernel import (
    SheafMorphismCokernelRequest,
    SheafMorphismCokernelResult,
    cokernel_of_morphism,
)


def _run(request: SheafMorphismCokernelRequest) -> SheafMorphismCokernelResult:
    return cokernel_of_morphism(request.morphism)


_ONE_POINT = {
    "complex": {
        "vertices": ["a"],
        "maximal_simplices": [["a"]],
        "faces_by_dimension": [{"dimension": 0, "faces": [["a"]]}],
        "dimension": 0,
        "f_vector": [1],
        "closure_size": 1,
    },
    "coefficient_field": "QQ",
    "prime": None,
    "stalks": [{"simplex": ["a"], "basis": ["x", "y"]}],
    "cover_restrictions": [],
    "derived_restrictions": [],
    "diamonds": 0,
    "comparable_pairs": 0,
}


TOOLS = (
    MathTool(
        operation_id="cellular_sheaf.morphism.cokernel.compute",
        title="Compute the cokernel sheaf of a natural morphism",
        description=(
            "Compute exact pointwise quotient stalks over QQ or GF(p), derive "
            "their restriction maps, and return the canonical projection from "
            "the morphism target."
        ),
        request_type=SheafMorphismCokernelRequest,
        result_type=SheafMorphismCokernelResult,
        run=_run,
        tags=("topology", "cellular-sheaf", "morphism", "cokernel", "exact"),
        discovery_terms=("cokernel sheaf", "cellular sheaf morphism cokernel"),
        examples=(
            OperationExample(
                name="projection_onto_one_point_quotient",
                description="Quotient a two-dimensional stalk by the span of its first basis vector.",
                input={
                    "morphism": {
                        "source": {
                            **_ONE_POINT,
                            "stalks": [{"simplex": ["a"], "basis": ["u"]}],
                        },
                        "target": _ONE_POINT,
                        "components": [[
                            ["a"],
                            [[{"num": "1", "den": "1"}], [{"num": "0", "den": "1"}]],
                        ]],
                        "natural": True,
                        "obstruction": None,
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
