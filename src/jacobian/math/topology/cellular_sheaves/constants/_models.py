"""Request contract for constructing constant cellular sheaves."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_PRIME,
    MAX_SHEAF_STALK_RANK,
    BasisLabel,
    SheafField,
)


class ConstantSheafRequest(StrictModel):
    """The finite complex and one based exact vector space to copy to every cell."""

    complex: FiniteSimplicialComplex
    coefficient_field: SheafField = SheafField.RATIONAL
    prime: StrictInt | None = Field(default=None, ge=2, le=MAX_SHEAF_PRIME)
    basis: tuple[BasisLabel, ...] = Field(
        default=("e0",),
        max_length=MAX_SHEAF_STALK_RANK,
        description=(
            "Ordered basis identifiers of the common stalk vector space. The "
            "empty tuple requests the zero vector space."
        ),
    )

    @model_validator(mode="after")
    def require_unique_basis(self) -> ConstantSheafRequest:
        if len(set(self.basis)) != len(self.basis):
            raise ValueError("basis identifiers must be unique and ordered")
        return self


__all__ = ["ConstantSheafRequest"]
