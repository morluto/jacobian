"""Elementary symmetric family declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._elementary_symmetric import (
    ElementarySymmetricFamilyRequest,
    ElementarySymmetricFamilyResult,
    elementary_symmetric_family,
)

ELEMENTARY_SYMMETRIC_FAMILY_OPERATION = MathTool(
    operation_id="polynomial.elementary_symmetric.family.compute",
    title="Compute an elementary symmetric polynomial family",
    description=(
        "Return the canonical sparse rational polynomials e_0 through e_k on "
        "an ordered variable axis, including e_0 = 1 for the empty axis."
    ),
    request_type=ElementarySymmetricFamilyRequest,
    result_type=ElementarySymmetricFamilyResult,
    run=elementary_symmetric_family,
    tags=("polynomial", "symmetric", "elementary", "family", "exact"),
    examples=(
        OperationExample(
            name="three_variables",
            description="Compute e_0, e_1, and e_2 in x, y, z.",
            input={"variables": ["x", "y", "z"], "maximum_degree": 2},
        ),
    ),
)
