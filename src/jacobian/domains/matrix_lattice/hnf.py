"""In-process Python-FLINT row-HNF producer owned by the matrix-lattice domain."""

from __future__ import annotations

from typing import Any, Never

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.matrices import IntegerMatrix
from jacobian.contracts.matrix_lattice import (
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
)
from jacobian.contracts.operations import (
    OperationDiagnostic,
    ProviderAvailability,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains._examples import example
from jacobian.operation_declarations import InlineOperation
from jacobian.operations import (
    OperationAbortError,
)
from jacobian.providers.flint_runtime import python_flint_hnf_provider_runtime

HNF_RUNTIME = python_flint_hnf_provider_runtime()


def _failure(status: ExecutionStatus, code: str, message: str) -> Never:
    raise OperationAbortError(
        status,
        OperationDiagnostic(
            code=code,
            stage="matrix_hnf_provider",
            message=message,
            hint="Install the pinned Python-FLINT HNF provider and retry.",
        ),
    )


def _matrix(value: Any) -> IntegerMatrix:
    return IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(int(value[row, column]))
                for column in range(value.ncols())
            )
            for row in range(value.nrows())
        )
    )


def compute_hermite_normal_form(
    request: HermiteNormalFormRequest,
) -> HermiteNormalFormResult:
    runtime = python_flint_hnf_provider_runtime(refresh=True)
    if (
        HNF_RUNTIME.availability is not ProviderAvailability.AVAILABLE
        or runtime != HNF_RUNTIME
    ):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_PROVIDER_UNAVAILABLE",
            "The pinned Python-FLINT HNF provider is unavailable.",
        )
    integer_entries = [
        [parse_canonical_integer(value) for value in row]
        for row in request.matrix.entries
    ]
    try:
        import flint

        normal_form, transformation = flint.fmpz_mat(integer_entries).hnf(True)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_EXECUTION_FAILED",
            "The pinned Python-FLINT HNF computation did not complete successfully.",
        )
    if python_flint_hnf_provider_runtime(refresh=True) != HNF_RUNTIME:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_RUNTIME_CHANGED",
            "The Python-FLINT HNF runtime changed during the bounded computation.",
        )
    return HermiteNormalFormResult(
        normal_form=_matrix(normal_form),
        transformation=_matrix(transformation),
    )


HERMITE_NORMAL_FORM_OPERATION: InlineOperation[
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
] = InlineOperation(
    operation_id="matrix.normal_form.hermite.compute",
    version="1",
    title="Compute an exact row Hermite normal form",
    description=(
        "Use pinned Python-FLINT to retain H and U for one bounded integer matrix, "
        "with the proposed relation H = U A."
    ),
    request_type=HermiteNormalFormRequest,
    result_type=HermiteNormalFormResult,
    run=compute_hermite_normal_form,
    tags=(
        "matrix",
        "integer",
        "hermite-normal-form",
        "certificate",
        "python-flint",
    ),
    examples=(
        example(
            "unit_matrix",
            "Compute the row HNF of the one-by-one unit matrix.",
            {"matrix": {"entries": [["1"]]}},
        ),
    ),
)

__all__ = ["HERMITE_NORMAL_FORM_OPERATION", "compute_hermite_normal_form"]
