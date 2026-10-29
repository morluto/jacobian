"""Request-only rational presentations for full-carrier matrix analysis.

Unlike bounded linear algebra's 256-digit inputs, these consumers admit source
heights using their own work and output budgets. Preserve the canonical carrier's
raw component ceiling before reduction, without widening any execution envelope.
"""

from typing import Annotated

from pydantic import ConfigDict, Field

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math.matrices._input_diagnostics import _matrix_storage_branch
from jacobian.math.matrices._rational_input import (
    RationalInputEncoding,
    RationalValueInputEncoding,
)
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_AXIS,
    EmbeddedRealSimpleNumberFieldMatrix,
    RationalMatrix,
)


def _exact_real_matrix_branch(value: object) -> str | None:
    """Identify explicit scalar-domain evidence only after union parsing fails.

    This never selects the successful decoder. Ambiguous or unusual native
    storage retains both diagnostics, with representation names in messages.
    """

    if type(value) is RationalMatrix:
        return "rational"
    if type(value) is EmbeddedRealSimpleNumberFieldMatrix:
        return "algebraic"
    if (
        type(value) is not dict
        or len(value) > 5
        or any(type(key) is not str for key in value)
    ):
        return None
    domain = value.get("domain")
    if type(domain) is str:
        if domain == "QQ":
            return "rational"
        if domain == "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD":
            return "algebraic"
        return None
    if domain is not None:
        return None
    if "embedding" in value:
        return "algebraic"
    if _matrix_storage_branch(value) != "dense":
        return None
    if all(
        type(scalar) is CanonicalRational
        or (type(scalar) is dict and scalar and scalar.keys() <= {"num", "den"})
        for row in value["entries"]
        for scalar in row
    ):
        return "rational"
    return None


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
