"""Public bounded operation for quadratic-form determinant conventions."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.general.determinant_discriminant._models import (
    MAX_POLAR_DETERMINANT_AXIS,
    MAX_POLAR_DETERMINANT_INTERMEDIATE_DIGITS,
    MAX_POLAR_DETERMINANT_MATRIX_ENTRIES,
    MAX_POLAR_DETERMINANT_OUTPUT_DIGITS,
    MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS,
    MAX_POLAR_DETERMINANT_SUPPORT_TERMS,
    MAX_POLAR_DETERMINANT_WORK,
    DeterminantDiscriminantRequest,
    DeterminantDiscriminantResult,
)
from jacobian.math.number_theory.quadratic_forms.general.determinant_discriminant.operations import (
    polar_gram_determinant_discriminant,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quadratic_form.determinant_discriminant.compute",
        title="Compute a rational quadratic form determinant and signed discriminant",
        description=(
            "Return det(G) and (-1)^(n(n-1)/2) det(G), where G is the full "
            "polar Gram matrix of Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j: "
            "G_ii=2*a_i and G_ij=c_ij. This differs from the coefficient "
            "matrix A in Q=x^T A x, whose off-diagonal entries are c_ij/2. "
            "The signed value is a rational representative; only its square "
            "class is basis invariant, and that class requires nondegeneracy. "
            f"Dimension is at most {MAX_POLAR_DETERMINANT_AXIS}; exact Bareiss "
            f"matrix entries and stored form support are each at most "
            f"{MAX_POLAR_DETERMINANT_MATRIX_ENTRIES}/{MAX_POLAR_DETERMINANT_SUPPORT_TERMS}; "
            f"work is at most {MAX_POLAR_DETERMINANT_WORK}; intermediate and "
            f"output heights are bounded by {MAX_POLAR_DETERMINANT_INTERMEDIATE_DIGITS} "
            f"and {MAX_POLAR_DETERMINANT_OUTPUT_DIGITS} decimal digits, with "
            f"{MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS} retained source digits."
        ),
        request_type=DeterminantDiscriminantRequest,
        result_type=DeterminantDiscriminantResult,
        run=polar_gram_determinant_discriminant,
        tags=("quadratic-form", "determinant", "discriminant", "exact"),
        examples=(
            OperationExample(
                name="binary-sign-convention",
                description=(
                    "For Q=x^2+xy+y^2, det of the full polar Gram matrix is 3 "
                    "and the signed discriminant is -3, matching b^2-4ac. "
                    "Precondition: exact rational coefficients in the stated "
                    "Q convention and distinct in-range cross-term indices."
                ),
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "cross_terms": [
                            {
                                "left": 0,
                                "right": 1,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
