"""Focused production graph for SymPy polynomial normalization tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.operations import ProviderObservation
from jacobian.polynomial_expression_operations import (
    install_polynomial_expression_checker,
)
from jacobian.providers.sympy_runtime import (
    sympy_polynomial_normalization_provider_runtime,
)
from jacobian.sympy_polynomial_normalization import (
    bind_sympy_polynomial_normalization,
)


@dataclass(frozen=True, slots=True)
class PolynomialNormalizationTestServices(DomainTestServices):
    """Services and producer identity owned by the normalization boundary."""

    producer: ProviderObservation


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
            producer = bind_sympy_polynomial_normalization(
                services.core.polynomial_expressions,
                runtime,
            )
            services.installation.register_operation(producer)
            if with_checker:
                checker, _installation = install_polynomial_expression_checker(
                    services.core.store,
                    services.core.schemas,
                    services.core.artifacts,
                    services.core.polynomial_expressions,
                    services.verification,
                    services.core.checkers,
                    authorize_checker=True,
                )
                assert checker is not None
                services.installation.register_operation(checker)
        yield PolynomialNormalizationTestServices(
            core=services.core,
            verification=services.verification,
            polytope=services.polytope,
            installation=services.installation,
            producer=runtime,
        )
