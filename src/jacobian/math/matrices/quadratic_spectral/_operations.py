"""Wire adapters for exact real-quadratic matrix spectra."""

from jacobian.math.matrices.quadratic_spectral._models import (
    RealQuadraticInertiaRequest,
    RealQuadraticSingularSpectrumRequest,
    RealQuadraticSymmetricSpectrumRequest,
)
from jacobian.math.matrices.quadratic_spectral.operations import (
    inertia,
    singular_spectrum,
    symmetric_spectrum,
)
from jacobian.math.matrices.quadratic_spectral.values import (
    RealQuadraticInertia,
    RealQuadraticSpectrum,
)


def compute_symmetric_spectrum(
    request: RealQuadraticSymmetricSpectrumRequest,
) -> RealQuadraticSpectrum:
    return symmetric_spectrum(request.matrix)


def compute_singular_spectrum(
    request: RealQuadraticSingularSpectrumRequest,
) -> RealQuadraticSpectrum:
    return singular_spectrum(request.matrix)


def compute_inertia(request: RealQuadraticInertiaRequest) -> RealQuadraticInertia:
    return inertia(request.matrix)


__all__ = [
    "compute_inertia",
    "compute_singular_spectrum",
    "compute_symmetric_spectrum",
]
