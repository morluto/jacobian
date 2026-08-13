"""Provider-owned runtime declaration for typed SymPy normalization."""

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.contracts.polynomial_expressions import (
    SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION,
)
from jacobian.provider_runtime import (
    SYMPY_VERSION,
    _unavailable_runtime,
    python_distribution_provider_runtime,
)


def sympy_polynomial_normalization_provider_runtime(
    *,
    refresh: bool = False,
) -> ProviderObservation:
    """Identify the pinned typed polynomial-normalization profile."""

    runtime = python_distribution_provider_runtime(
        "jacobian.sympy",
        distribution_name="sympy",
        import_name="sympy",
        required_attributes=("Add", "Mul", "Poly", "Pow", "QQ", "Rational", "Symbol"),
        install_tier=ProviderInstallTier.T0,
        license_id="BSD-3-Clause",
        features=(
            "typed-polynomial-expression",
            "exact-rational",
            "canonical-sparse-normalization",
        ),
        configuration=SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION,
        refresh=refresh,
    )
    if (
        runtime.availability is ProviderAvailability.AVAILABLE
        and runtime.version != SYMPY_VERSION
    ):
        return _unavailable_runtime(
            provider="jacobian.sympy",
            install_tier=ProviderInstallTier.T0,
            license_id="BSD-3-Clause",
            diagnostic=(
                "SymPy is installed but does not match the pinned "
                f"{SYMPY_VERSION} polynomial-normalization profile."
            ),
        )
    return runtime
