"""Typed catalog fixture for MCP projection tests without a complete runtime."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.geometry import build_geometry_bundle
from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.polynomial import build_polynomial_bundle
from jacobian.domains.topology import build_topology_bundle
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
        build_polynomial_bundle(),
        build_matrix_bundle(),
        build_topology_bundle(),
        build_number_theory_bundle(),
        build_combinatorics_bundle(),
        build_geometry_bundle(),
    ) as services:
        yield services
