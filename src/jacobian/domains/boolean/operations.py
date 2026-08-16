"""Domain adapter for Boolean truth-table operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.boolean import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.math.boolean import walsh_hadamard_transform


def compute_walsh_hadamard_transform(
    request: BooleanTruthTableRequest,
) -> BooleanWalshTransformResult:
    spectrum = walsh_hadamard_transform(list(request.truth_table))
    n = len(request.truth_table)
    variable_count = n.bit_length() - 1
    return BooleanWalshTransformResult(
        spectrum=tuple(format_canonical_integer(value) for value in spectrum),
        variable_count=variable_count,
    )
