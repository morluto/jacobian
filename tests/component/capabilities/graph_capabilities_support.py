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

from jacobian.graphs.installation import GraphInstallation, install_graph_capabilities
from jacobian.runtime.config import CheckerAuthorityMode


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
            adapters, graph = install_graph_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
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
