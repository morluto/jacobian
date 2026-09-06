"""Typed wire contracts for finitely generated abelian group operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import NativeInteger
from jacobian._models import StrictModel

MAX_ORDERS = 32
MAX_GROUP_ORDER = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"abelian_presentation.{reason}", message)


class AbelianPresentation(StrictModel):
    """An invariant-factor decomposition of a finitely generated abelian group."""

    invariant_factors: tuple[NativeInteger, ...] = Field(min_length=0, max_length=MAX_ORDERS)

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
        order = 1
        for factor in self.invariant_factors:
            order *= factor
            if order > MAX_GROUP_ORDER:
                raise _validation_error(
                    "group_order_bound",
                    f"group order exceeds the {MAX_GROUP_ORDER}-element bound",
                )
        return self


class PresentationNormalizeRequest(StrictModel):
    """A bounded finite cyclic-factor presentation to normalize exactly.

    The factors describe a direct product of finite cyclic groups and need not
    already satisfy the invariant-factor divisibility convention.  The
    operation returns that canonical decomposition.
    """

    invariant_factors: tuple[NativeInteger, ...] = Field(
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
    group: AbelianPresentation
    coordinates: tuple[NativeInteger, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates) != len(self.group.invariant_factors):
            raise _validation_error(
                "coordinate_length",
                "coordinates length must match invariant_factors length",
            )
        return self


class ElementEqualRequest(StrictModel):
    group: AbelianPresentation
    coordinates_a: tuple[NativeInteger, ...] = Field(min_length=1, max_length=MAX_ORDERS)
    coordinates_b: tuple[NativeInteger, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates_a) != len(self.group.invariant_factors):
            raise _validation_error(
                "coordinate_a_length",
                "coordinates_a length must match invariant_factors",
            )
        if len(self.coordinates_b) != len(self.group.invariant_factors):
            raise _validation_error(
                "coordinate_b_length",
                "coordinates_b length must match invariant_factors",
            )
        return self


class ElementOrderRequest(StrictModel):
    group: AbelianPresentation
    coordinates: tuple[NativeInteger, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.coordinates) != len(self.group.invariant_factors):
            raise _validation_error(
                "coordinate_length", "coordinates length must match invariant_factors"
            )
        return self


class SubgroupGeneratedRequest(StrictModel):
    group: AbelianPresentation
    generators: tuple[tuple[NativeInteger, ...], ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(len(g) != len(self.group.invariant_factors) for g in self.generators):
            raise _validation_error(
                "generator_length", "each generator must match invariant_factors length"
            )
        order = 1
        for d in self.group.invariant_factors:
            order *= d
        if order > MAX_GROUP_ORDER:
            raise _validation_error(
                "group_order_bound",
                f"group order exceeds the {MAX_GROUP_ORDER}-element bound",
            )
        return self


class QuotientRequest(StrictModel):
    group: AbelianPresentation
    subgroup_generators: tuple[tuple[NativeInteger, ...], ...] = Field(
        min_length=1, max_length=MAX_ORDERS
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(len(g) != len(self.group.invariant_factors) for g in self.subgroup_generators):
            raise _validation_error(
                "generator_length", "each generator must match invariant_factors length"
            )
        order = 1
        for d in self.group.invariant_factors:
            order *= d
        if order > MAX_GROUP_ORDER:
            raise _validation_error(
                "group_order_bound",
                f"group order exceeds the {MAX_GROUP_ORDER}-element bound",
            )
        return self


# Results


class AbelianElement(StrictModel):
    """A canonical coordinate value in one finite abelian group."""

    group: AbelianPresentation
    coordinates: tuple[NativeInteger, ...] = Field(
        min_length=0, max_length=MAX_ORDERS
    )

    @model_validator(mode="after")
    def require_canonical_coordinates(self) -> Self:
        if len(self.coordinates) != len(self.group.invariant_factors):
            raise _validation_error(
                "element_coordinate_length",
                "element coordinates must match the group invariant-factor axis",
            )
        if any(
            not 0 <= coordinate < factor
            for coordinate, factor in zip(
                self.coordinates, self.group.invariant_factors, strict=True
            )
        ):
            raise _validation_error(
                "element_not_reduced",
                "canonical element coordinates must be reduced modulo the group factors",
            )
        return self


class AbelianSubgroup(StrictModel):
    """A subgroup presentation with every generator bound to its parent."""

    group: AbelianPresentation
    generators: tuple[AbelianElement, ...] = Field(min_length=1, max_length=MAX_ORDERS)

    @model_validator(mode="after")
    def require_parent_bound_generators(self) -> Self:
        if any(generator.group != self.group for generator in self.generators):
            raise _validation_error(
                "subgroup_parent_mismatch", "subgroup generators must share the parent group"
            )
        return self


class AbelianQuotient(StrictModel):
    """A quotient presentation retaining its parent group and subgroup map."""

    group: AbelianPresentation
    subgroup: AbelianSubgroup
    invariant_factors: tuple[NativeInteger, ...] = ()

    @model_validator(mode="after")
    def require_parent_bound_subgroup(self) -> Self:
        if self.subgroup.group != self.group:
            raise _validation_error(
                "quotient_parent_mismatch", "quotient subgroup must use the parent group"
            )
        return self


class PresentationNormalizeResult(StrictModel):
    """The canonical finite abelian-group presentation."""

    source: PresentationNormalizeRequest
    presentation: AbelianPresentation

    @property
    def invariant_factors(self) -> tuple[NativeInteger, ...]:
        """Project the canonical invariant factors."""

        return self.presentation.invariant_factors

    @property
    def order(self) -> int:
        """Return the order of the presented finite group."""

        order = 1
        for factor in self.presentation.invariant_factors:
            order *= factor
        return order

    @property
    def rank(self) -> int:
        """The finite-only contract admits no free summands."""

        return 0


class ElementReduceResult(StrictModel):
    group: AbelianPresentation
    coordinates: tuple[NativeInteger, ...]
    reduced: AbelianElement


class ElementEqualResult(StrictModel):
    """A source-bound equality claim for two finite abelian group elements."""

    group: AbelianPresentation
    elements_a: AbelianElement
    elements_b: AbelianElement
    equal: bool


class ElementOrderResult(StrictModel):
    """A source-bound order claim for a finite abelian group element."""

    group: AbelianPresentation
    element: AbelianElement
    order: NativeInteger = Field(ge=1)


class SubgroupGeneratedResult(StrictModel):
    subgroup: AbelianSubgroup
    index: NativeInteger = Field(ge=1)
    coset_representatives: tuple[AbelianElement, ...] = ()


class QuotientResult(StrictModel):
    quotient: AbelianQuotient
    quotient_order: NativeInteger = Field(ge=1)
