"""Typed wire contracts for finitely generated abelian group operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_ORDERS = 32
MAX_GROUP_ORDER = 4_096


class AbelianPresentation(StrictModel):
    """An invariant-factor decomposition of a finitely generated abelian group."""

    invariant_factors: tuple[int, ...] = Field(min_length=0, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(f < 2 for f in self.invariant_factors):
            raise ValueError(
                "invariant factors must be integers >= 2; "
                "trivial factors of 1 must be omitted and zero (free) "
                "summands are not admitted by the finite-group contract"
            )
        if any(
            self.invariant_factors[i + 1] % self.invariant_factors[i] != 0
            for i in range(len(self.invariant_factors) - 1)
        ):
            raise ValueError(
                "invariant factors must satisfy d_i | d_{i+1} "
                "(each factor divides the next)"
            )
        return self


class ElementReduceRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates) != len(self.invariant_factors):
            raise ValueError("coordinates length must match invariant_factors length")
        if any(d < 2 for d in self.invariant_factors):
            raise ValueError("invariant factors must be integers >= 2")
        return self


class ElementEqualRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates_a: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates_b: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates_a) != len(self.invariant_factors):
            raise ValueError("coordinates_a length must match invariant_factors")
        if len(self.coordinates_b) != len(self.invariant_factors):
            raise ValueError("coordinates_b length must match invariant_factors")
        if any(d < 2 for d in self.invariant_factors):
            raise ValueError("invariant factors must be integers >= 2")
        return self


class ElementOrderRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates) != len(self.invariant_factors):
            raise ValueError("coordinates length must match invariant_factors")
        if any(d < 2 for d in self.invariant_factors):
            raise ValueError("invariant factors must be integers >= 2")
        return self


class SubgroupGeneratedRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    generators: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(len(g) != len(self.invariant_factors) for g in self.generators):
            raise ValueError("each generator must match invariant_factors length")
        if any(d < 2 for d in self.invariant_factors):
            raise ValueError("invariant factors must be integers >= 2")
        order = 1
        for d in self.invariant_factors:
            order *= d
        if order > MAX_GROUP_ORDER:
            raise ValueError(f"group order exceeds the {MAX_GROUP_ORDER}-element bound")
        return self


class QuotientRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    subgroup_generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_ORDERS
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(len(g) != len(self.invariant_factors) for g in self.subgroup_generators):
            raise ValueError("each generator must match invariant_factors length")
        if any(d < 2 for d in self.invariant_factors):
            raise ValueError("invariant factors must be integers >= 2")
        order = 1
        for d in self.invariant_factors:
            order *= d
        if order > MAX_GROUP_ORDER:
            raise ValueError(f"group order exceeds the {MAX_GROUP_ORDER}-element bound")
        return self


# Results


class PresentationNormalizeResult(StrictModel):
    invariant_factors: tuple[int, ...]
    order: int = Field(ge=1)
    rank: int = Field(ge=0)
    method: str = "SmithNormalForm"


class ElementReduceResult(StrictModel):
    reduced: tuple[int, ...]
    method: str = "MODULAR_REDUCTION"


class ElementEqualResult(StrictModel):
    equal: bool
    method: str = "MODULAR_COMPARISON"


class ElementOrderResult(StrictModel):
    order: int = Field(ge=1)
    method: str = "LCM"


class SubgroupGeneratedResult(StrictModel):
    index: int = Field(ge=1)
    coset_representatives: tuple[tuple[int, ...], ...] = ()
    method: str = "COSET_ENUMERATION"


class QuotientResult(StrictModel):
    quotient_invariant_factors: tuple[int, ...] = ()
    quotient_order: int = Field(ge=1)
    method: str = "SMITH_NORMAL_FORM"
