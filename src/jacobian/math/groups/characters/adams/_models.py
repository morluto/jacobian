"""Wire contract for exact Adams operations."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.groups.characters._models import CharacterRingElement


class AdamsOperationRequest(StrictModel):
    """Apply a positive integral Adams operation to a table-bound virtual character."""

    character: CharacterRingElement
    exponent: ExactInteger = Field(
        description=(
            "Positive integer k in psi^k(chi)(g) = chi(g^k); powering work "
            "is admitted from its bit length and the retained group representation."
        ),
    )

    @model_validator(mode="after")
    def require_positive_exponent(self) -> Self:
        if self.exponent < 1:
            raise PydanticCustomError(
                "groups.characters.adams_exponent",
                "Adams-operation exponent must be a positive integer",
            )
        return self


__all__ = ["AdamsOperationRequest"]
