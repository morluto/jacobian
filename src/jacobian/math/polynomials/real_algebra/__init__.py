"""Exact real algebra operations and canonical profiles."""

from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    CommonInterlacingDoesNotExist,
    CommonInterlacingExists,
    CommonInterlacingGap,
    CommonInterlacingObstruction,
    CommonInterlacingOutcome,
    CommonInterlacingProfile,
    EmptyGapObstruction,
    LabelledRationalPolynomial,
    NonRealRootObstruction,
    PolynomialRealRoot,
    PolynomialRootReference,
    SourceRootProfile,
)
from jacobian.math.polynomials.real_algebra.operations import (
    common_interlacing_profile,
    root_count,
    sturm_chain,
)

__all__ = [
    "CommonInterlacingDoesNotExist",
    "CommonInterlacingExists",
    "CommonInterlacingGap",
    "CommonInterlacingObstruction",
    "CommonInterlacingOutcome",
    "CommonInterlacingProfile",
    "EmptyGapObstruction",
    "LabelledRationalPolynomial",
    "NonRealRootObstruction",
    "PolynomialRealRoot",
    "PolynomialRootReference",
    "SourceRootProfile",
    "common_interlacing_profile",
    "root_count",
    "sturm_chain",
]
