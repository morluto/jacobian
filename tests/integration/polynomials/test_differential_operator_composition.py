"""Cross-owner composition for exact differential-operator output."""

from jacobian.math.polynomial_vector_calc._models import ScalarFieldRequest
from jacobian.math.polynomials.differential_operators._operations import (
    compute_differential_operator_application,
)
from jacobian.math.polynomials.differential_operators._tools import TOOLS


def test_output_serializes_directly_into_an_existing_polynomial_consumer() -> None:
    request = TOOLS[0].request_type.model_validate(TOOLS[0].examples[0].input)
    output = compute_differential_operator_application(request).output

    consumer = ScalarFieldRequest.model_validate(
        {"polynomial": output.model_dump(mode="json")}
    )

    assert consumer.polynomial == output
