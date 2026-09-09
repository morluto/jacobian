"""Linear-code canonicalization declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.codes.linear._canonicalization import (
    LinearCodeCanonicalizationRequest,
    LinearCodeCanonicalizationResult,
    canonicalize_linear_code,
)

LINEAR_CODE_CANONICALIZATION_OPERATION = MathTool(
    operation_id="code.linear.prime_field.coordinate_canonicalize.compute",
    title="Canonicalize a prime-field linear code under coordinate permutations",
    description="Traverse a bounded full symmetric or supplied permutation-group action, reduce each transported encoder to RREF, and return the least encoder, transporter, orbit size, stabilizer size, and transported axis.",
    request_type=LinearCodeCanonicalizationRequest,
    result_type=LinearCodeCanonicalizationResult,
    run=canonicalize_linear_code,
    tags=("code", "linear-code", "canonicalization", "permutation-group", "exact"),
    examples=(
        OperationExample(
            name="binary_length_three",
            description="Canonicalize a binary one-dimensional length-three code under all coordinates.",
            input={
                "encoder": {
                    "field_order": 2,
                    "message_axis": ["m"],
                    "coordinate_axis": ["x", "y", "z"],
                    "generator_matrix": [[1, 0, 1]],
                },
                "action": "FULL_SYMMETRIC",
            },
        ),
    ),
)

__all__ = ["LINEAR_CODE_CANONICALIZATION_OPERATION"]
