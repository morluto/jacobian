"""Catalog operation for pointwise cellular-sheaf morphism images."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cellular_sheaves.morphism_image import (
    SheafMorphismImageRequest,
    SheafMorphismImageResult,
    image_of_morphism,
)


def _run(request: SheafMorphismImageRequest) -> SheafMorphismImageResult:
    return image_of_morphism(request.morphism)


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
    "stalks": [{"simplex": ["a"], "basis": ["x"]}],
    "cover_restrictions": [],
    "derived_restrictions": [],
    "diamonds": 0,
    "comparable_pairs": 0,
}

TOOLS = (
    MathTool(
        operation_id="cellular_sheaf.morphism.image.compute",
        title="Compute the image sheaf of a natural morphism",
        description=(
            "Compute exact pointwise image stalks and their induced restrictions "
            "over QQ or GF(p), returning the image inclusion and source factorization."
        ),
        request_type=SheafMorphismImageRequest,
        result_type=SheafMorphismImageResult,
        run=_run,
        tags=("topology", "cellular-sheaf", "morphism", "image", "exact"),
        discovery_terms=("image sheaf", "cellular sheaf morphism image"),
        examples=(
            OperationExample(
                name="identity_on_one_point",
                description="The identity endomorphism has the full stalk as its image.",
                input={
                    "morphism": {
                        "source": _ONE_POINT,
                        "target": _ONE_POINT,
                        "components": [[["a"], [[{"num": "1", "den": "1"}]]]],
                        "natural": True,
                        "obstruction": None,
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
