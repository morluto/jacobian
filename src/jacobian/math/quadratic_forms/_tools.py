"""Quadratic-form operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.quadratic_forms._models import EvaluationRequest, EvaluationResult
from jacobian.math.quadratic_forms._operations import evaluate_form

TOOLS = (
    MathTool(
        operation_id="quadratic_form.evaluate.compute",
        version="2",
        title="Evaluate an exact rational quadratic form",
        description=(
            "Evaluate Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j exactly over QQ. "
            "Axis labels are unique; diagonal_coefficients carries exactly "
            "one coefficient per axis label and every cross-term index lies "
            "within that axis. The ordered vector axis must equal the form "
            "axis. The form stores polynomial coefficients; its polar matrix "
            "has diagonal 2*a_i and off-diagonal c_ij."
        ),
        request_type=EvaluationRequest,
        result_type=EvaluationResult,
        run=evaluate_form,
        tags=("algebra", "quadratic-form", "exact-rational"),
        examples=(
            example(
                "binary_cross_term",
                "Evaluate 2*x^2+3*x*y+5*y^2 at (1/2, 2); axis labels are "
                "unique, diagonal_coefficients and coordinates hold exactly "
                "one entry per label in axis order, every cross-term index "
                "lies inside the axis, and the vector axis equals the form "
                "axis.",
                {
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "2", "den": "1"},
                            {"num": "5", "den": "1"},
                        ],
                        "cross_terms": [
                            {
                                "left": 0,
                                "right": 1,
                                "coefficient": {"num": "3", "den": "1"},
                            }
                        ],
                    },
                    "vector": {
                        "axis": ["x", "y"],
                        "coordinates": [
                            {"num": "1", "den": "2"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
