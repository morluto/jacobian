"""Killable process boundary for exact number-field discriminants."""

from __future__ import annotations

import time

from jacobian._execution import (
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    _WORKER as _WORKER,
)
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    _WORKER_ADDRESS_SPACE_BYTES as _WORKER_ADDRESS_SPACE_BYTES,
)
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    _WORKER_FILE_SIZE_BYTES as _WORKER_FILE_SIZE_BYTES,
)
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    _WORKER_STDERR_BYTES as _WORKER_STDERR_BYTES,
)
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    _WORKER_TIMEOUT_SECONDS as _WORKER_TIMEOUT_SECONDS,
)
from jacobian.math.number_theory.number_fields._integral_basis_process import (
    IntegralBasisWorkerResult,
    run_integral_basis_worker,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)

# Keep the worker envelope visible to existing process tests and other
# number-field owners while discriminant and basis consumers share one
# process implementation.


def compute_nf_discriminant(
    request: NumberFieldRequest,
) -> NumberFieldDiscriminantResult:
    """Return a field discriminant established by the isolated basis worker."""

    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return compute_nf_discriminant(request)
    worker_result = run_integral_basis_worker(request, include_basis=False)
    if worker_result is None:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.not_irreducible",
            message="number-field polynomial must be irreducible over QQ",
        )
    result = _discriminant_result(request, worker_result)
    request_checkpoint("after number-field discriminant result construction")
    return result


def _discriminant_result(
    request: NumberFieldRequest,
    worker_result: IntegralBasisWorkerResult,
) -> NumberFieldDiscriminantResult:
    return NumberFieldDiscriminantResult(
        field=request.field,
        discriminant=worker_result.field_discriminant,
    )


__all__ = ["compute_nf_discriminant"]
