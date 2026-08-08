"""Exact matrix and lattice operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operations import DomainBundle

__all__ = ["build_lattice_bundle", "build_matrix_bundle"]


def build_matrix_bundle() -> DomainBundle:
    """Build the matrix bundle without loading it during native API imports."""

    from jacobian.domains.matrix_lattice.bundle import build_matrix_bundle as build

    return build()


def build_lattice_bundle() -> DomainBundle:
    """Build the lattice bundle without loading it during native API imports."""

    from jacobian.domains.matrix_lattice.lattice_bundle import (
        build_lattice_bundle as build,
    )

    return build()
