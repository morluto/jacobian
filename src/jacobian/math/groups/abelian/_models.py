"""Typed wire contracts for finitely generated abelian group operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_ORDERS = 32
MAX_GROUP_ORDER = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"abelian_presentation.{reason}", message)


class AbelianPresentation(StrictModel):
    """An invariant-factor decomposition of a finitely generated abelian group."""

    invariant_factors: tuple[int, ...] = Field(min_length=0, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(f < 2 for f in self.invariant_factors):
            raise _validation_error(
                "factor_not_finite",
                "invariant factors must be integers >= 2; "
                "trivial factors of 1 must be omitted and zero (free) "
                "summands are not admitted by the finite-group contract",
            )
        if any(
            self.invariant_factors[i + 1] % self.invariant_factors[i] != 0
            for i in range(len(self.invariant_factors) - 1)
        ):
            raise _validation_error(
                "factor_divisibility",
                "invariant factors must satisfy d_i | d_{i+1} "
                "(each factor divides the next)",
            )
        return self


class PresentationNormalizeRequest(StrictModel):
    """A bounded finite cyclic-factor presentation to normalize exactly.

    The factors describe a direct product of finite cyclic groups and need not
    already satisfy the invariant-factor divisibility convention.  The
    operation returns that canonical decomposition.
    """

    invariant_factors: tuple[int, ...] = Field(
        min_length=0,
        max_length=MAX_ORDERS,
        description=(
            "Finite cyclic-factor orders to normalize; input order and "
            "divisibility need not be canonical."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_finite_presentation(self) -> Self:
        if any(factor < 2 for factor in self.invariant_factors):
            raise _validation_error(
                "input_factor_not_finite",
                "presentation factors must be integers >= 2; trivial factors "
                "of 1 must be omitted and free summands are not admitted",
            )
        order = 1
        for factor in self.invariant_factors:
            order *= factor
            if order > MAX_GROUP_ORDER:
                raise _validation_error(
                    "input_order_bound",
                    f"presentation order exceeds the {MAX_GROUP_ORDER}-element bound",
                )
        return self


class ElementReduceRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates) != len(self.invariant_factors):
            raise _validation_error(
                "coordinate_length",
                "coordinates length must match invariant_factors length",
            )
        if any(d < 2 for d in self.invariant_factors):
            raise _validation_error(
                "factor_not_finite", "invariant factors must be integers >= 2"
            )
        return self


class ElementEqualRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates_a: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates_b: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates_a) != len(self.invariant_factors):
            raise _validation_error(
                "coordinate_a_length",
                "coordinates_a length must match invariant_factors",
            )
        if len(self.coordinates_b) != len(self.invariant_factors):
            raise _validation_error(
                "coordinate_b_length",
                "coordinates_b length must match invariant_factors",
            )
        if any(d < 2 for d in self.invariant_factors):
            raise _validation_error(
                "factor_not_finite", "invariant factors must be integers >= 2"
            )
        return self


class ElementOrderRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates) != len(self.invariant_factors):
            raise _validation_error(
                "coordinate_length", "coordinates length must match invariant_factors"
            )
        if any(d < 2 for d in self.invariant_factors):
            raise _validation_error(
                "factor_not_finite", "invariant factors must be integers >= 2"
            )
        return self


class SubgroupGeneratedRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    generators: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(len(g) != len(self.invariant_factors) for g in self.generators):
            raise _validation_error(
                "generator_length", "each generator must match invariant_factors length"
            )
        if any(d < 2 for d in self.invariant_factors):
            raise _validation_error(
                "factor_not_finite", "invariant factors must be integers >= 2"
            )
        order = 1
        for d in self.invariant_factors:
            order *= d
        if order > MAX_GROUP_ORDER:
            raise _validation_error(
                "group_order_bound",
                f"group order exceeds the {MAX_GROUP_ORDER}-element bound",
            )
        return self


class QuotientRequest(StrictModel):
    invariant_factors: tuple[int, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    subgroup_generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_ORDERS
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(len(g) != len(self.invariant_factors) for g in self.subgroup_generators):
            raise _validation_error(
                "generator_length", "each generator must match invariant_factors length"
            )
        if any(d < 2 for d in self.invariant_factors):
            raise _validation_error(
                "factor_not_finite", "invariant factors must be integers >= 2"
            )
        order = 1
        for d in self.invariant_factors:
            order *= d
        if order > MAX_GROUP_ORDER:
            raise _validation_error(
                "group_order_bound",
                f"group order exceeds the {MAX_GROUP_ORDER}-element bound",
            )
        return self


# Results


class PresentationNormalizeResult(StrictModel):
    invariant_factors: tuple[int, ...]
    order: int = Field(ge=1)
    rank: int = Field(ge=0)


class ElementReduceResult(StrictModel):
    reduced: tuple[int, ...]


class ElementEqualResult(StrictModel):
    equal: bool


class ElementOrderResult(StrictModel):
    order: int = Field(ge=1)


class SubgroupGeneratedResult(StrictModel):
    index: int = Field(ge=1)
    coset_representatives: tuple[tuple[int, ...], ...] = ()


class QuotientResult(StrictModel):
    quotient_invariant_factors: tuple[int, ...] = ()
    quotient_order: int = Field(ge=1)
