"""Atomic elementary collapses of finite simplicial complexes."""

from jacobian.math.topology.simplicial_collapse._models import (
    ElementaryCollapsePair,
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
)
from jacobian.math.topology.simplicial_collapse.operations import elementary_collapse

__all__ = [
    "ElementaryCollapsePair",
    "ElementaryCollapseRequest",
    "ElementaryCollapseResult",
    "elementary_collapse",
]
