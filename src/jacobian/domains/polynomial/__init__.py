"""Exact rational polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operations import DomainBundle

__all__ = ["build_polynomial_bundle"]


def build_polynomial_bundle() -> DomainBundle:
    """Construct the optional capability bundle without polluting native imports."""

    from jacobian.domains.polynomial.bundle import build_polynomial_bundle as _build

    return _build()
