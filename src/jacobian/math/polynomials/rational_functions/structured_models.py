"""Structured exact differential and residue values over QQ(x)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalFunction, RationalPolynomial


class RationalFunctionRequest(StrictModel):
    function: RationalFunction


class LogarithmicDifferentialTerm(StrictModel):
    """A rationally represented dlog row; no analytic log branch is chosen."""

    factor: RationalPolynomial
    numerator: RationalPolynomial
    exponent: Literal[1] = 1


class LogarithmicDifferentialResult(StrictModel):
    source: RationalFunction
    terms: tuple[LogarithmicDifferentialTerm, ...] = Field(default=())
    reconstructed: RationalFunction


class ResidueRow(StrictModel):
    factor: RationalPolynomial
    root_index: int = Field(ge=0)
    pole_order: int = Field(ge=1)
    numerator_at_root: RationalPolynomial
    derivative_factor: RationalPolynomial


class GlobalResidueResult(StrictModel):
    source: RationalFunction
    finite_poles: tuple[ResidueRow, ...]
    residue_at_infinity: CanonicalRational
    finite_residue_sum: CanonicalRational
    total_residue: CanonicalRational


class FormalAntiderivativeResult(StrictModel):
    source: RationalFunction
    rational_part: RationalFunction
    logarithmic_part: LogarithmicDifferentialResult


def _rational_primitive_schema(schema: dict) -> None:
    """Publish status branches, including the zero/nonzero remainder shape."""

    common = {
        "source": {"type": "object"},
        "rational_part": {"type": "object"},
    }
    schema["oneOf"] = [
        {
            "title": "RationalPrimitive",
            "properties": {
                **common,
                "status": {"const": "RATIONAL_PRIMITIVE"},
                "remainder": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "numerator": {
                                    "type": "object",
                                    "properties": {"terms": {"maxItems": 0}},
                                }
                            },
                        },
                    ]
                },
            },
            "required": ["source", "status", "rational_part", "remainder"],
        },
        {
            "title": "NoRationalPrimitive",
            "properties": {
                **common,
                "status": {"const": "NO_RATIONAL_PRIMITIVE"},
                "remainder": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "numerator": {
                                    "type": "object",
                                    "properties": {"terms": {"minItems": 1}},
                                }
                            },
                        },
                    ]
                },
            },
            "required": ["source", "status", "rational_part", "remainder"],
        },
    ]
    schema["discriminator"] = {
        "propertyName": "status",
        "mapping": {
            "RATIONAL_PRIMITIVE": "#/oneOf/0",
            "NO_RATIONAL_PRIMITIVE": "#/oneOf/1",
        },
    }


class RationalPrimitiveResult(StrictModel):
    model_config = ConfigDict(json_schema_extra=_rational_primitive_schema)

    source: RationalFunction
    status: Literal["RATIONAL_PRIMITIVE", "NO_RATIONAL_PRIMITIVE"]
    rational_part: RationalFunction
    remainder: RationalFunction

    @model_validator(mode="after")
    def require_status_payload(self) -> Self:
        if (
            self.rational_part.variables != self.source.variables
            or self.remainder.variables != self.source.variables
        ):
            raise ValueError("primitive result values must retain the source variables")
        has_remainder = bool(self.remainder.numerator.terms)
        if self.status == "RATIONAL_PRIMITIVE" and has_remainder:
            raise ValueError("a rational primitive must have zero remainder")
        if self.status == "NO_RATIONAL_PRIMITIVE" and not has_remainder:
            raise ValueError("a non-primitive result must retain its remainder")
        return self


__all__ = [
    "FormalAntiderivativeResult",
    "GlobalResidueResult",
    "LogarithmicDifferentialResult",
    "LogarithmicDifferentialTerm",
    "RationalFunctionRequest",
    "RationalPrimitiveResult",
    "ResidueRow",
]
