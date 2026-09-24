"""Public exact operations on integral quadratic forms."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.integral import operations as native
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticFormInclusion,
    IntegralQuadraticFormInclusionRequest,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quadratic_form.integral.rational_extension.compute",
        title="Extend an integral quadratic form to rational coefficients",
        description=(
            "Apply the explicit coefficient inclusion ZZ→QQ to an integral "
            "quadratic polynomial. The result retains both source and target "
            "forms, their shared ordered coordinate axis, and the typed map "
            "identity; it does not silently coerce a rational form back to ZZ."
        ),
        request_type=IntegralQuadraticFormInclusionRequest,
        result_type=IntegralQuadraticFormInclusion,
        run=native.integral_form_to_rational,
        tags=("quadratic-form", "integral", "coefficient-map", "exact"),
        discovery_terms=(
            "integral quadratic form over the integers",
            "include a ZZ quadratic form into QQ",
            "rational scalar extension of an integral quadratic form",
        ),
        examples=(
            OperationExample(
                name="odd_cross_coefficient_is_preserved",
                description=(
                    "The polynomial coefficient 3 stays 3 over QQ; the implicit "
                    "half-polar matrix convention must not halve the polynomial."
                ),
                input={
                    "form": {
                        "axis": ["u", "v"],
                        "diagonal_coefficients": ["2", "0"],
                        "cross_terms": [{"left": 0, "right": 1, "coefficient": "3"}],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
