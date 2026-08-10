"""Shared harness for installing exact-domain checkers in domain tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from jacobian.operations import DomainBundle
from jacobian.runtime.config import CheckerAuthorityMode
from tests.support.services import DomainTestServices, open_domain_services


@contextmanager
def open_exact_domain_services(
    root: str | Path,
    *bundles: DomainBundle,
) -> Iterator[DomainTestServices]:
    """Open domain services with bundled exact-domain checker authority."""

    with open_domain_services(
        root,
        *bundles,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        yield services
