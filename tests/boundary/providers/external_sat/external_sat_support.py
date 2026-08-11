from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.providers.external_solver_runtime import (
    cadical_provider_runtime,
    drat_trim_provider_runtime,
)
from jacobian.runtime import CheckerAuthorityMode
from jacobian.sat_smt.cadical import install_cadical_capabilities
from jacobian.sat_smt.sat_capabilities import (
    install_sat_assignment_checker,
    install_sat_unsat_proof_checker,
)


@contextmanager
def open_cadical_services(root: Path) -> Iterator[DomainTestServices]:
    """Open the production CaDiCaL producer graph without unrelated bundles."""

    with _open_external_sat_services(root) as services:
        yield services


@contextmanager
def open_verified_external_sat_services(
    root: Path,
) -> Iterator[DomainTestServices]:
    """Open CaDiCaL producers with both production SAT evidence checkers."""

    with _open_external_sat_services(
        root,
        install_assignment_checker=True,
        install_proof_checker=True,
    ) as services:
        yield services


@contextmanager
def open_verified_unsat_services(root: Path) -> Iterator[DomainTestServices]:
    """Open the production CaDiCaL-to-DRAT verification graph."""

    with _open_external_sat_services(
        root,
        install_proof_checker=True,
    ) as services:
        yield services


@contextmanager
def _open_external_sat_services(
    root: Path,
    *,
    install_assignment_checker: bool = False,
    install_proof_checker: bool = False,
) -> Iterator[DomainTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if install_assignment_checker or install_proof_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            cadical = cadical_provider_runtime()
            for adapter in install_cadical_capabilities(services.core.sat, cadical):
                services.installation.register_capability(adapter)

            if install_assignment_checker:
                assignment, _installation = install_sat_assignment_checker(
                    services.core.store,
                    services.core.schemas,
                    services.core.artifacts,
                    services.core.sat,
                    services.application.verification,
                    services.core.checkers,
                    authorize_checker=services.installation.authorizes_bundled_checkers,
                )
                if assignment is None:
                    raise RuntimeError("the SAT assignment checker was not installed")
                services.installation.register_capability(assignment)

            if install_proof_checker:
                proof, _installation = install_sat_unsat_proof_checker(
                    services.core.store,
                    services.core.schemas,
                    services.core.artifacts,
                    services.core.sat,
                    services.application.verification,
                    services.core.checkers,
                    drat_trim_provider_runtime(),
                    authorize_checker=services.installation.authorizes_bundled_checkers,
                )
                if proof is None:
                    raise RuntimeError("the SAT proof checker was not installed")
                services.installation.register_capability(proof)
        yield services
