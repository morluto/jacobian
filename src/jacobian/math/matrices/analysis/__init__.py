"""Native exact matrix-analysis operations and canonical values."""

from jacobian.math.matrices.analysis._models import (
    InertiaResult,
    RationalSpectrumMultiplicityClaim,
)
from jacobian.math.matrices.analysis.operations import (
    check_farkas_certificate,
    check_rational_spectrum_claim,
    compute_inertia,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    ExactRealMatrix,
    RationalMatrix,
)

__all__ = [
    "EmbeddedRealSimpleNumberFieldMatrix",
    "ExactRealMatrix",
    "InertiaResult",
    "RationalMatrix",
    "RationalSpectrumMultiplicityClaim",
    "check_farkas_certificate",
    "check_rational_spectrum_claim",
    "compute_inertia",
]
