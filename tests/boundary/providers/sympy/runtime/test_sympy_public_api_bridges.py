from fractions import Fraction

import sympy

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.math import arithmetic, matrices


def _assert_computed_result(result) -> None:
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.artifact_uris == ()
    assert result.relationships == ()


def test_native_and_capability_arithmetic_agree(authorized_complete_runtime) -> None:
    runtime = authorized_complete_runtime
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="rational.compute.sum",
            input={
                "left": {"num": "1", "den": "3"},
                "right": {"num": "1", "den": "6"},
            },
        )
    )

    native = arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6))
    assert result.output["result"]["value"] == {
        "num": str(native.numerator),
        "den": str(native.denominator),
    }
    _assert_computed_result(result)


def test_native_and_capability_matrix_inverse_agree(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.inverse.compute",
            input={
                "matrix": {
                    "domain": "ZZ",
                    "entries": [["1", "2"], ["3", "4"]],
                }
            },
        )
    )

    native = matrices.inverse(sympy.Matrix([[1, 2], [3, 4]]))
    wire = result.output["result"]["inverse"]["entries"]
    bridged = sympy.Matrix(
        [
            [sympy.Rational(int(value["num"]), int(value["den"])) for value in row]
            for row in wire
        ]
    )
    assert bridged == native
    _assert_computed_result(result)


def test_capability_provider_provenance_is_unchanged(
    authorized_complete_runtime,
) -> None:
    providers = {
        descriptor.capability_id: descriptor.provider
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id
        in {"rational.compute.sum", "matrix.inverse.compute"}
    }
    assert providers == {
        "rational.compute.sum": "jacobian.sympy",
        "matrix.inverse.compute": "jacobian.sympy",
    }
