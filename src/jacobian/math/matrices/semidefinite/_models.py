"""Requests for a single rational exposed-face reduction."""

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.semidefinite.values import RationalSemidefiniteSystem


class SemidefiniteFaceReductionRequest(StrictModel):
    system: RationalSemidefiniteSystem
    multipliers: tuple[CanonicalRational, ...] = Field(
        max_length=8192,
        description="Supplied y with nonzero PSD sum y_i A_i and sum y_i b_i = 0.",
    )
