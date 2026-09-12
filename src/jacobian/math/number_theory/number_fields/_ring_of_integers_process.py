"""Process adapter for exact ring-of-integers basis construction."""

from __future__ import annotations

import time

from jacobian._execution import (
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._integral_basis import (
    require_factorizable_discriminant,
)
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    run_integral_basis_worker,
)
from jacobian.math.number_theory.number_fields._models import (
    MAX_INTEGRAL_BASIS_DEGREE,
    NumberFieldRequest,
    NumberFieldRingOfIntegersRequest,
)
from jacobian.math.number_theory.number_fields._ring_of_integers import (
    NumberFieldRingOfIntegersResult,
)
from jacobian.math.number_theory.number_fields.values import SimpleNumberFieldElement


def compute_nf_ring_of_integers(
    request: NumberFieldRingOfIntegersRequest,
) -> NumberFieldRingOfIntegersResult:
    """Construct the public result from a bounded worker projection."""

    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return compute_nf_ring_of_integers(request)
    if request.field.degree > MAX_INTEGRAL_BASIS_DEGREE:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.ring_of_integers_degree_bound",
            message=(
                "the ring-of-integers operation is limited to degree "
                f"{MAX_INTEGRAL_BASIS_DEGREE}"
            ),
        )
    require_factorizable_discriminant(request.field)
    worker_result = run_integral_basis_worker(
        NumberFieldRequest(field=request.field),
        include_basis=True,
    )
    if worker_result is None:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.defining_polynomial_must_be_irreducible",
            message="a number field requires an irreducible defining polynomial",
        )
    if worker_result.basis is None:
        raise RuntimeError("number-field worker returned no integral basis")
    result = NumberFieldRingOfIntegersResult(
        field=request.field,
        basis=tuple(
            SimpleNumberFieldElement(
                presentation=request.field,
                coefficients_ascending=vector,
            )
            for vector in worker_result.basis
        ),
        field_discriminant=worker_result.field_discriminant,
    )
    result.require_canonical_basis()
    request_checkpoint("after number-field ring-of-integers result construction")
    return result


__all__ = ["compute_nf_ring_of_integers"]
