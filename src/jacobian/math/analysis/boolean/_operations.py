"""Domain-owned Boolean truth-table operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.boolean import walsh_hadamard_transform
from jacobian.math.analysis.boolean._models import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)


def compute_walsh_hadamard_transform(
    request: BooleanTruthTableRequest,
) -> BooleanWalshTransformResult:
    size = len(request.truth_table)
    if size & (size - 1):
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean.fourier.walsh_transform.power_of_two",
            message="truth table length must be a power of two",
        )
    spectrum = walsh_hadamard_transform(list(request.truth_table))
    variable_count = size.bit_length() - 1
    return BooleanWalshTransformResult(
        spectrum=tuple(format_canonical_integer(value) for value in spectrum),
        variable_count=variable_count,
    )
