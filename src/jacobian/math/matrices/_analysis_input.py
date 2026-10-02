"""Request-only rational presentations for full-carrier matrix analysis.

Unlike bounded linear algebra's 256-digit inputs, these consumers admit source
heights using their own work and output budgets. Preserve the canonical carrier's
raw component ceiling before reduction, without widening any execution envelope.
"""

from typing import Annotated

from pydantic import ConfigDict, Field

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.matrices._rational_input import (
    RationalInputEncoding,
    RationalValueInputEncoding,
)
from jacobian.math.matrices.values import MAX_RATIONAL_MATRIX_AXIS, RationalMatrix

AnalysisRationalInput = Annotated[
    CanonicalRational,
    RationalInputEncoding(max_digits=MAX_CANONICAL_RATIONAL_DIGITS),
]


class _AnalysisRationalMatrixInput(RationalMatrix):
    model_config = ConfigDict(title="AnalysisRationalMatrixInput")

    entries: tuple[tuple[AnalysisRationalInput, ...], ...] = Field(
        default=(),
        max_length=MAX_RATIONAL_MATRIX_AXIS,
        description=(
            "Exact rational entries. JSON requests accept unreduced ratios and "
            "either denominator sign. Each canonical decimal string component "
            "has at most 32768 digits before reduction; denominators are nonzero. "
            "Dimensions and entry order are preserved. Operation-specific work, "
            "growth and output admission still applies to the normalized matrix."
        ),
    )


AnalysisRationalMatrixInput = Annotated[
    RationalMatrix,
    RationalValueInputEncoding(RationalMatrix, _AnalysisRationalMatrixInput),
]
