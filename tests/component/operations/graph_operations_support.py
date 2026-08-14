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

from jacobian.graphs.operation_resources import (
    GraphOperationResources,
    build_graph_operations,
)


@dataclass(frozen=True, slots=True)
class GraphTestServices(DomainTestServices):
    graph: GraphOperationResources


@contextmanager
def open_graph_services(
    root: Path,
    *,
    authorize_checker: bool = False,
) -> Iterator[GraphTestServices]:
    """Build the core graph operations with optional checker authority."""

    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            adapters, graph_resources = build_graph_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorize_bundled_checkers,
            )
            for adapter in adapters:
                services.installation.register_operation(adapter)
        yield GraphTestServices(
            core=services.core,
            verification=services.verification,
            polytope=services.polytope,
            installation=services.installation,
            graph=graph_resources,
        )
