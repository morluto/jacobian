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

from jacobian.atomic_capabilities import install_atomic_capabilities
from jacobian.graphs.installation import GraphInstallation, install_graph_capabilities
from jacobian.runtime import CheckerAuthorityMode


@dataclass(frozen=True, slots=True)
class GraphTestServices(DomainTestServices):
    graph: GraphInstallation


@contextmanager
def open_graph_services(
    root: Path,
    *,
    authorize_checker: bool = False,
) -> Iterator[GraphTestServices]:
    """Install the production core graph graph with optional checker authority."""

    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            for adapter in install_atomic_capabilities(
                services.installation, services.application
            ):
                services.installation.register_capability(adapter)
            adapters, graph = install_graph_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            for adapter in adapters:
                services.installation.register_capability(adapter)
        yield GraphTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            graph=graph,
        )
