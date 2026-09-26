"""Public declaration for integral unimodular coordinate change."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.integral.unimodular._models import (
    UnimodularChangeRequest,
    UnimodularChangeResult,
)
from jacobian.math.number_theory.quadratic_forms.integral.unimodular.operations import (
    unimodular_change,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quadratic_form.unimodular_change.compute",
        title="Change an integral quadratic form by a unimodular basis map",
        description=(
            "For an integral quadratic form Q and a supplied integer matrix M "
            "with determinant ±1, return Q(M y) on the requested target axis. "
            "The exact inverse map is included, so the result is an integral "
            "equivalent presentation with an explicit two-way coordinate map. "
            "Dimension is at most 32; coefficient, inverse growth, work, and "
            "aggregate exact-result digits are admitted before transformation."
        ),
        request_type=UnimodularChangeRequest,
        result_type=UnimodularChangeResult,
        run=unimodular_change,
        tags=("quadratic-form", "integral", "basis-change", "exact"),
        discovery_terms=(
            "unimodular equivalence transformation of an integral quadratic form",
            "change basis of an integer quadratic form",
            "integral congruence by a matrix in GL(n,Z)",
        ),
        examples=(
            OperationExample(
                name="shear_with_odd_cross_coefficient",
                description=(
                    "Apply an integral shear to x^2+3xy+2y^2, retaining the "
                    "polynomial cross-term convention."
                ),
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": ["1", "2"],
                        "cross_terms": [{"left": 0, "right": 1, "coefficient": "3"}],
                    },
                    "matrix": {
                        "row_count": 2,
                        "column_count": 2,
                        "entries": [["1", "1"], ["0", "1"]],
                    },
                    "target_axis": ["u", "v"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
