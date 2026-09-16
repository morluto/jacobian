"""Typed wire contracts for finite class-function operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.groups.characters._cyclotomic import euler_phi

MAX_CLASS_COUNT = 128
MAX_GROUP_ORDER = 1_000_000
MAX_CYCLOTOMIC_ORDER = 60
MAX_VALUE_COEFFICIENT_DIGITS = 512
MAX_INNER_PRODUCT_WORK = 250_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"groups.characters.{reason}", message)


class CyclotomicValue(StrictModel):
    """One exact element of Q(zeta_order) in the distinguished power basis.

    ``coefficients`` are ascending powers of a primitive ``order``-th root of
    unity, reduced modulo the order's cyclotomic polynomial; equality is
    coefficient-wise.  Order one is the rational subfield with the single
    coefficient equal to the rational value.
    """

    order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER)
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_CYCLOTOMIC_ORDER,
        description="Ascending-power coefficients of length phi(order).",
    )

    @model_validator(mode="after")
    def require_cyclotomic_degree(self) -> Self:
        if len(self.coefficients) != euler_phi(self.order):
            raise _validation_error(
                "cyclotomic_degree",
                "value coefficients must have length phi(order)",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, order: int, coefficients: tuple[CanonicalRational, ...]
    ) -> CyclotomicValue:
        return cls.model_construct(order=order, coefficients=coefficients)


class ClassAxis(StrictModel):
    """The shared conjugacy-class axis of one finite group.

    ``class_sizes`` are the exact conjugacy-class cardinalities in the
    declared order and ``group_order`` is their sum.  ``cyclotomic_order`` is
    the common root-of-unity order declared for values on this axis.
    """

    class_sizes: tuple[int, ...] = Field(min_length=1, max_length=MAX_CLASS_COUNT)
    group_order: int = Field(ge=1, le=MAX_GROUP_ORDER)
    cyclotomic_order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER)

    @model_validator(mode="after")
    def require_complete_partition(self) -> Self:
        if any(size < 1 or size > self.group_order for size in self.class_sizes):
            raise _validation_error(
                "class_size",
                "every class size must be a positive part of the group order",
            )
        if sum(self.class_sizes) != self.group_order:
            raise _validation_error(
                "group_order",
                "group_order must equal the sum of the class sizes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, class_sizes: tuple[int, ...], cyclotomic_order: int
    ) -> ClassAxis:
        return cls.model_construct(
            class_sizes=class_sizes,
            group_order=sum(class_sizes),
            cyclotomic_order=cyclotomic_order,
        )


class FiniteClassFunction(StrictModel):
    """One complete exact class function on a single conjugacy-class axis.

    ``values`` contains exactly one exact cyclotomic value per class in
    ``axis.class_sizes`` order, all sharing the axis's cyclotomic order.  A
    same-length vector for another axis is structurally invalid.
    """

    axis: ClassAxis
    values: tuple[CyclotomicValue, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )

    @model_validator(mode="after")
    def require_axis_bound_values(self) -> Self:
        if len(self.values) != len(self.axis.class_sizes):
            raise _validation_error(
                "value_count",
                "a class function needs exactly one value per class",
            )
        if any(value.order != self.axis.cyclotomic_order for value in self.values):
            raise _validation_error(
                "value_cyclotomic_order",
                "every value must share the class axis cyclotomic order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, axis: ClassAxis, values: tuple[CyclotomicValue, ...]
    ) -> FiniteClassFunction:
        return cls.model_construct(axis=axis, values=values)


class ClassFunctionInnerProductRequest(StrictModel):
    """Two class functions on one shared conjugacy-class axis."""

    phi: FiniteClassFunction = Field(
        description="Left class function on the shared class axis."
    )
    psi: FiniteClassFunction = Field(
        description=(
            "Right class function on the same class axis; it must carry the "
            "identical class_sizes, group_order, and cyclotomic_order."
        )
    )


class ClassContribution(StrictModel):
    """One exact per-class term of the Hermitian inner product."""

    class_index: int = Field(ge=0)
    class_size: int = Field(ge=1)
    phi_value: CyclotomicValue
    psi_value: CyclotomicValue
    conjugate_psi_value: CyclotomicValue
    weighted_product: CyclotomicValue


class ClassFunctionInnerProductResult(StrictModel):
    """The exact Hermitian inner product with its complete contribution table."""

    axis: ClassAxis
    phi: FiniteClassFunction
    psi: FiniteClassFunction
    contributions: tuple[ClassContribution, ...] = Field(min_length=1)
    inner_product: CyclotomicValue

    @model_validator(mode="after")
    def require_bound_contributions(self) -> Self:
        if self.phi.axis != self.axis or self.psi.axis != self.axis:
            raise _validation_error(
                "result_axis",
                "result inputs must carry the declared shared class axis",
            )
        if len(self.contributions) != len(self.axis.class_sizes):
            raise _validation_error(
                "contribution_count",
                "the contribution table must cover every class",
            )
        if tuple(row.class_index for row in self.contributions) != tuple(
            range(len(self.axis.class_sizes))
        ):
            raise _validation_error(
                "contribution_order",
                "contributions must cover class indices in order",
            )
        if self.inner_product.order != self.axis.cyclotomic_order:
            raise _validation_error(
                "inner_product_order",
                "the inner product must share the axis cyclotomic order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        axis: ClassAxis,
        phi: FiniteClassFunction,
        psi: FiniteClassFunction,
        contributions: tuple[ClassContribution, ...],
        inner_product: CyclotomicValue,
    ) -> ClassFunctionInnerProductResult:
        return cls.model_construct(
            axis=axis,
            phi=phi,
            psi=psi,
            contributions=contributions,
            inner_product=inner_product,
        )


__all__ = [
    "MAX_CLASS_COUNT",
    "MAX_CYCLOTOMIC_ORDER",
    "MAX_GROUP_ORDER",
    "MAX_INNER_PRODUCT_WORK",
    "MAX_VALUE_COEFFICIENT_DIGITS",
    "ClassAxis",
    "ClassContribution",
    "ClassFunctionInnerProductRequest",
    "ClassFunctionInnerProductResult",
    "CyclotomicValue",
    "FiniteClassFunction",
]
