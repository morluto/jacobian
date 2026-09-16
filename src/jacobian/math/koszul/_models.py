"""Typed wire contracts for the Koszul complex construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.koszul.values import (
    MAX_KOSZUL_SEQUENCE_LENGTH,
    MAX_KOSZUL_VARIABLES,
)
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"koszul.{reason}", message)


class KoszulComplexRequest(StrictModel):
    """Construct the exact Koszul complex ``K(f)`` of one ordered sequence.

    The ambient ring is ``R = QQ[x_1, ..., x_m]`` with bounded ``m`` and the
    module is the ring itself; based-module variants are deferred. Numeric
    envelope admission (term/degree/coefficient growth, wedge basis and
    differential entry counts, and the exact ``d^2 = 0`` replay work) runs in
    the shared kernel admission, so the native and catalog paths charge the
    same request identically.
    """

    variables: tuple[PolynomialVariable, ...] = Field(
        default=(),
        max_length=MAX_KOSZUL_VARIABLES,
        description=(
            "The ordered variable axis of the ambient ring QQ[x_1, ..., x_m]; "
            f"at most {MAX_KOSZUL_VARIABLES} variables. The empty axis is the "
            "coefficient field QQ itself."
        ),
    )
    sequence: tuple[RationalPolynomial, ...] = Field(
        default=(),
        max_length=MAX_KOSZUL_SEQUENCE_LENGTH,
        description=(
            "The ordered sequence (f_1, ..., f_c) of ring elements in the "
            f"canonical sparse QQ wire format; at most "
            f"{MAX_KOSZUL_SEQUENCE_LENGTH} elements. Repeated and zero "
            "elements are allowed and the empty sequence gives the identity "
            "complex R in degree 0. Order is presentation data affecting "
            "wedge basis signs."
        ),
    )

    @model_validator(mode="after")
    def require_one_ordered_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "request_variables", "ambient ring variables must be unique"
            )
        for element in self.sequence:
            if element.variables != self.variables:
                raise _validation_error(
                    "ring_mismatch",
                    "every sequence element must belong to the single "
                    "declared ordered QQ polynomial ring",
                )
        return self


__all__ = ["KoszulComplexRequest"]
