"""Typed requests for algebraic-torus operations."""

from jacobian._models import StrictModel
from jacobian.math.algebraic_tori.values import HomogeneousMonomialSystem


class HomogeneousMonomialSolutionRequest(StrictModel):
    system: HomogeneousMonomialSystem


__all__ = ["HomogeneousMonomialSolutionRequest"]
