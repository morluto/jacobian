"""Domain-owned exact rational-linear capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operations import DomainBundle


def build_rational_linear_bundle() -> DomainBundle:
    from jacobian.domains.rational_linear.bundle import (
        build_rational_linear_bundle as build,
    )

    return build()


__all__ = ["build_rational_linear_bundle"]
