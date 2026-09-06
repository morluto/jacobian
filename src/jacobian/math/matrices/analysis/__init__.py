"""Native exact matrix-analysis operations and canonical values."""

from jacobian.math.matrices.analysis._models import (
    InertiaResult,
    RationalSpectrumMultiplicityClaim,
)
from jacobian.math.matrices.analysis.operations import (
    check_farkas_certificate,
    check_rational_spectrum_claim,
    compute_inertia,
    verify_inertia,
)

__all__ = [
    "InertiaResult",
    "RationalSpectrumMultiplicityClaim",
    "check_farkas_certificate",
    "check_rational_spectrum_claim",
    "compute_inertia",
    "verify_inertia",
]
