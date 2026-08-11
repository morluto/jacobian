"""Focused production graph for SymPy polynomial normalization tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.polynomial_expression_capabilities import (
    install_polynomial_expression_checker,
)
from jacobian.providers.sympy_runtime import (
    sympy_polynomial_normalization_provider_runtime,
)
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.sympy_polynomial_normalization import (
    install_sympy_polynomial_normalization_capability,
)


@dataclass(frozen=True, slots=True)
class PolynomialNormalizationTestServices(DomainTestServices):
    """Services and measured provider owned by the normalization boundary."""

    provider_runtime: CapabilityProviderRuntime


@contextmanager
def open_polynomial_normalization_services(
    root: Path,
    *,
    with_checker: bool = False,
) -> Iterator[PolynomialNormalizationTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if with_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        runtime = sympy_polynomial_normalization_provider_runtime()
        with atomic_installation(services.core):
            producer = install_sympy_polynomial_normalization_capability(
                services.core.polynomial_expressions,
                runtime,
            )
            services.installation.register_capability(producer)
            if with_checker:
                checker, _installation = install_polynomial_expression_checker(
                    services.core.store,
                    services.core.schemas,
                    services.core.artifacts,
                    services.core.polynomial_expressions,
                    services.application.verification,
                    services.core.checkers,
                    authorize_checker=True,
                )
                assert checker is not None
                services.installation.register_capability(checker)
        yield PolynomialNormalizationTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            provider_runtime=runtime,
        )
