"""Public operation manifest for free-algebra homogeneous projections."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.free_algebras.homogeneous_component._models import (
    FreeAlgebraHomogeneousComponent,
    FreeAlgebraHomogeneousComponentRequest,
)
from jacobian.math.free_algebras.homogeneous_component.operations import (
    homogeneous_component,
)

TOOLS = (
    MathTool(
        operation_id="free_algebra.polynomial.homogeneous_component.compute",
        title="Project a free-algebra polynomial to one degree",
        description=(
            "Return the exact sum of terms of one requested word degree in a "
            "sparse QQ-linear free-algebra polynomial. The ordered generator "
            "alphabet is preserved, including for a zero component. Terms are "
            "already stored in degree-first order, so selection preserves the "
            "canonical polynomial representation without coefficient arithmetic. "
            "The result retains the selected degree as well as the polynomial. "
            "Admission bounds the scan and estimated serialized output before "
            "constructing the result."
        ),
        request_type=FreeAlgebraHomogeneousComponentRequest,
        result_type=FreeAlgebraHomogeneousComponent,
        run=homogeneous_component,
        tags=("free-algebra", "polynomial", "graded", "homogeneous-component", "exact"),
        discovery_terms=(
            "noncommutative polynomial homogeneous component",
            "free algebra degree projection",
            "graded component of a free polynomial",
        ),
        examples=(
            OperationExample(
                name="degree_one_part",
                description=(
                    "Project x + 2y + 3xy onto degree one, retaining x + 2y "
                    "over the same ordered alphabet."
                ),
                input={
                    "polynomial": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "3", "den": "1"},
                                "word": ["x", "y"],
                            },
                            {"coefficient": {"num": "2", "den": "1"}, "word": ["y"]},
                            {"coefficient": {"num": "1", "den": "1"}, "word": ["x"]},
                        ],
                    },
                    "degree": 1,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
