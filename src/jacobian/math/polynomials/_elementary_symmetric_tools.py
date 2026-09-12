"""Elementary symmetric family declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._elementary_symmetric import (
    ElementarySymmetricFamilyRequest,
    ElementarySymmetricFamilyResult,
    _elementary_symmetric_family_from_request,
)


def _run_elementary_symmetric_family(
    request: ElementarySymmetricFamilyRequest,
) -> ElementarySymmetricFamilyResult:
    return _elementary_symmetric_family_from_request(request)


ELEMENTARY_SYMMETRIC_FAMILY_OPERATION = MathTool(
    operation_id="polynomial.symmetric.elementary_family.compute",
    title="Compute an elementary symmetric polynomial family",
    description=(
        "Return the complete canonical sparse rational family e_0 through e_k "
        "on an ordered variable axis, including e_0 = 1; k must not exceed "
        "the number of distinct variables."
    ),
    request_type=ElementarySymmetricFamilyRequest,
    result_type=ElementarySymmetricFamilyResult,
    run=_run_elementary_symmetric_family,
    tags=("polynomial", "symmetric", "elementary", "family", "exact"),
    examples=(
        OperationExample(
            name="three_variables",
            description=(
                "Compute e_0, e_1, and e_2 in x, y, z; maximum_degree must not "
                "exceed the variable count."
            ),
            input={"variables": ["x", "y", "z"], "maximum_degree": 2},
        ),
    ),
)
