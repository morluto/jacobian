"""Typed wire contracts for plane algebraic curve operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel

MAX_VARS = 3
MAX_COEFF = 4096


class AffineCurveRequest(StrictModel):
    """An affine plane curve f(x, y) = 0."""

    variables: tuple[str, ...] = Field(min_length=2, max_length=2)
    polynomial: str = Field(min_length=1, max_length=MAX_COEFF)


class ProjectiveClosureRequest(StrictModel):
    """Compute the projective closure of an affine curve."""

    variables: tuple[str, ...] = Field(min_length=2, max_length=2)
    polynomial: str = Field(min_length=1, max_length=MAX_COEFF)


class AffineChartRequest(StrictModel):
    """Extract an affine chart from a projective curve."""

    variables: tuple[str, ...] = Field(min_length=3, max_length=3)
    polynomial: str = Field(min_length=1, max_length=MAX_COEFF)
    chart_variable: str = Field(min_length=1, max_length=64)


# Results


class AffineCurveResult(StrictModel):
    is_valid: bool
    degree: int = Field(ge=0)
    method: str = "SYmpy_CURVE_CHECK"


class ProjectiveClosureResult(StrictModel):
    polynomial: str
    variables: tuple[str, ...]
    method: str = "HOMOGENIZATION"


class AffineChartResult(StrictModel):
    polynomial: str
    variables: tuple[str, ...]
    method: str = "DEhomOGENIZATION"
