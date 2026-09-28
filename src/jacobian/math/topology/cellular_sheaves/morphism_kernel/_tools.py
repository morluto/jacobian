"""Catalog declaration for pointwise kernels of cellular sheaf morphisms."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cellular_sheaves.morphism_kernel import (
    SheafMorphismKernelRequest,
    SheafMorphismKernelResult,
    kernel_of_morphism,
)


def _run(request: SheafMorphismKernelRequest) -> SheafMorphismKernelResult:
    return kernel_of_morphism(request.morphism)


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
        operation_id="cellular_sheaf.morphism.kernel.compute",
        title="Compute the kernel sheaf of a natural morphism",
        description=(
            "Compute exact pointwise kernels over QQ or GF(p), derive every "
            "kernel restriction map, and return the kernel sheaf with its "
            "canonical inclusion into the source."
        ),
        request_type=SheafMorphismKernelRequest,
        result_type=SheafMorphismKernelResult,
        run=_run,
        tags=("topology", "cellular-sheaf", "morphism", "kernel", "exact"),
        discovery_terms=("kernel sheaf", "cellular sheaf morphism kernel"),
        examples=(
            OperationExample(
                name="zero_map_on_one_point",
                description="The zero endomorphism has the full one-dimensional stalk as its kernel.",
                input={
                    "morphism": {
                        "source": _ONE_POINT,
                        "target": _ONE_POINT,
                        "components": [[["a"], [[{"num": "0", "den": "1"}]]]],
                        "natural": True,
                        "obstruction": None,
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
