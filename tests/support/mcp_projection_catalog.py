"""Typed catalog fixture for MCP projection tests without a complete runtime."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from jacobian.domains.polynomial import build_polynomial_bundle
from tests.support.services import DomainTestServices, open_domain_services


@contextmanager
def open_mcp_projection_catalog(
    root: str | Path,
) -> Iterator[DomainTestServices]:
    """Open a domain service graph whose catalog can drive MCP projections.

    Projection/compaction tests need ``catalog()`` / ``discover()``, not a
    complete portfolio or MCP transport.
    """

    with open_domain_services(root, build_polynomial_bundle()) as services:
        yield services
