"""Provider-owned runtime declarations for direct FLINT-backed mathematics."""

from __future__ import annotations

import importlib

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_runtime import (
    PYTHON_FLINT_HNF_FLINT_VERSION,
    PYTHON_FLINT_LLL_CONFIGURATION,
    PYTHON_FLINT_VERSION,
    _unavailable_runtime,
    python_distribution_provider_runtime,
)


def _python_flint_runtime(
    *,
    required_attributes: tuple[str, ...],
    features: tuple[str, ...],
    configuration: dict[str, object],
    profile: str,
    refresh: bool,
) -> ProviderObservation:
    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=required_attributes,
        install_tier=ProviderInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=features,
        configuration=configuration,
        refresh=refresh,
    )
    if (
        runtime.availability is ProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=ProviderInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} {profile} profile."
            ),
        )
    return runtime


def python_flint_provider_runtime(*, refresh: bool = False) -> ProviderObservation:
    """Identify the exact packaged Python-FLINT compatibility profile."""

    return _python_flint_runtime(
        required_attributes=("fmpq", "fmpq_mat"),
        features=("exact-rational", "dense-matrix", "reduced-row-echelon-form"),
        configuration={
            "domain": "QQ",
            "operation": "fmpq_mat.rref",
            "maximum_rows": 32,
            "maximum_columns": 32,
            "free_variable_policy": "ZERO",
        },
        profile="compatibility",
        refresh=refresh,
    )


def _checked_flint_runtime(
    *,
    required_attributes: tuple[str, ...],
    features: tuple[str, ...],
    configuration: dict[str, object],
    profile: str,
    refresh: bool,
) -> ProviderObservation:
    runtime = _python_flint_runtime(
        required_attributes=required_attributes,
        features=features,
        configuration=configuration,
        profile=profile,
        refresh=refresh,
    )
    if not refresh or runtime.availability is not ProviderAvailability.AVAILABLE:
        return runtime
    try:
        flint = importlib.import_module("flint")
    except (ImportError, OSError):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=ProviderInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=f"The pinned Python-FLINT {profile} runtime cannot be imported.",
        )
    if getattr(flint, "__FLINT_VERSION__", None) != PYTHON_FLINT_HNF_FLINT_VERSION:
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=ProviderInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but its linked FLINT library does "
                f"not match the pinned {profile} profile."
            ),
        )
    return runtime


def python_flint_hnf_provider_runtime(*, refresh: bool = False) -> ProviderObservation:
    """Identify the pinned Python-FLINT integer row-HNF profile."""

    return _checked_flint_runtime(
        required_attributes=("fmpz", "fmpz_mat"),
        features=(
            "exact-integer",
            "dense-matrix",
            "row-hermite-normal-form",
            "left-transformation",
        ),
        configuration={
            "domain": "ZZ",
            "operation": "fmpz_mat.hnf(transform=True)",
            "flint_library_version": PYTHON_FLINT_HNF_FLINT_VERSION,
            "maximum_rows": 32,
            "maximum_columns": 32,
            "normal_form_convention": "FLINT_ROW_HNF",
            "relation": "H=U*A",
        },
        profile="HNF",
        refresh=refresh,
    )


def python_flint_lll_provider_runtime(*, refresh: bool = False) -> ProviderObservation:
    """Identify the pinned Python-FLINT exact-gram LLL profile."""

    return _checked_flint_runtime(
        required_attributes=("fmpz", "fmpz_mat"),
        features=(
            "exact-integer",
            "dense-matrix",
            "lll-reduction",
            "left-transformation",
        ),
        configuration=PYTHON_FLINT_LLL_CONFIGURATION,
        profile="LLL",
        refresh=refresh,
    )
