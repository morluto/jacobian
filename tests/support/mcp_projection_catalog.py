"""Typed catalog fixture for MCP projection tests without a complete runtime."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from jacobian.domains.combinatorics import combinatorics_operations
from jacobian.domains.geometry import geometry_operations
from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.domains.number_theory import number_theory_operations
from jacobian.domains.polynomial import polynomial_operations
from jacobian.domains.topology import topology_operations
from tests.support.services import DomainTestServices, open_domain_services


@contextmanager
def open_mcp_projection_catalog(
    root: str | Path,
) -> Iterator[DomainTestServices]:
    """Open a multi-bundle catalog for MCP projection/compaction tests.

    Projection helpers need ``catalog()`` / ``discover()`` over enough
    descriptors to exercise byte-budget compaction, not a complete portfolio or
    MCP transport.
    """

    with open_domain_services(
        root,
        polynomial_operations(),
        matrix_operations(),
        topology_operations(),
        number_theory_operations(),
        combinatorics_operations(),
        geometry_operations(),
    ) as services:
        yield services
