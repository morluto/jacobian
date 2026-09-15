"""Typed wire contracts for number field operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import BeforeValidator, Field, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldDiscriminantInteger,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
)

MAX_INTEGRAL_BASIS_DEGREE = 31
MAX_CLASS_GROUP_DEGREE = 16


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.{reason}", message)


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    field: SimpleNumberFieldPresentation


def _require_integral_basis_degree(value: Any) -> Any:
    """Apply the ring-of-integers degree envelope before shared parsing."""

    coefficients = (
        value.coefficients_descending
        if isinstance(value, SimpleNumberFieldPresentation)
        else value.get("coefficients_descending")
        if isinstance(value, Mapping)
        else None
    )
    if isinstance(coefficients, (list, tuple)) and len(coefficients) > (
        MAX_INTEGRAL_BASIS_DEGREE + 1
    ):
        raise _validation_error(
            "ring_of_integers_degree_bound",
            "ring-of-integers computation admits degree at most "
            f"{MAX_INTEGRAL_BASIS_DEGREE}",
        )
    if isinstance(value, Mapping):
        return canonicalize_json_containers(value)
    return value


def _integral_basis_field_schema() -> dict[str, Any]:
    schema = SimpleNumberFieldPresentation.model_json_schema(mode="validation")
    schema["properties"]["coefficients_descending"]["maxItems"] = (
        MAX_INTEGRAL_BASIS_DEGREE + 1
    )
    schema["description"] = (
        "A canonical simple number-field presentation of degree at most "
        f"{MAX_INTEGRAL_BASIS_DEGREE} for ring-of-integers computation."
    )
    return schema


_IntegralBasisField = Annotated[
    SimpleNumberFieldPresentation,
    BeforeValidator(_require_integral_basis_degree),
    WithJsonSchema(_integral_basis_field_schema()),
]


class NumberFieldRingOfIntegersRequest(StrictModel):
    """A simple number field within the integral-basis worker envelope."""

    field: _IntegralBasisField


class NumberFieldDiscriminantResult(StrictModel):
    """A field-discriminant claim retaining its exact field presentation."""

    field: SimpleNumberFieldPresentation
    discriminant: NumberFieldDiscriminantInteger


class NumberFieldEmbeddingsRequest(StrictModel):
    """Request every Archimedean embedding of one bounded presented field."""

    field: SimpleNumberFieldPresentation


class NumberFieldRealEmbeddingOrderRequest(StrictModel):
    """Compare two field elements at one selected real embedding record."""

    left: SimpleNumberFieldRealEmbeddingBinding
    right: SimpleNumberFieldRealEmbeddingBinding


class _BnfWorkerRequest(StrictModel):
    """Shared PARI worker request: one field within the class-group envelope."""

    field: _IntegralBasisField

    @model_validator(mode="after")
    def require_bnf_degree(self) -> _BnfWorkerRequest:
        if self.field.degree > MAX_CLASS_GROUP_DEGREE:
            raise _validation_error(
                "bnf_degree_bound",
                "class and unit group computation admits degree at most "
                f"{MAX_CLASS_GROUP_DEGREE}",
            )
        return self


class NumberFieldClassGroupRequest(StrictModel):
    """A simple number field within the PARI class-group envelope."""

    field: _IntegralBasisField

    @model_validator(mode="after")
    def require_class_group_degree(self) -> NumberFieldClassGroupRequest:
        if self.field.degree > MAX_CLASS_GROUP_DEGREE:
            raise _validation_error(
                "class_group_degree_bound",
                "class-group computation admits degree at most "
                f"{MAX_CLASS_GROUP_DEGREE}",
            )
        return self


class NumberFieldClassGroupResult(StrictModel):
    """The class group of one presented field, retaining its ideal representatives."""

    field: SimpleNumberFieldPresentation
    class_number: NumberFieldDiscriminantInteger
    abelian_invariants: tuple[int, ...] = Field(
        max_length=MAX_CLASS_GROUP_DEGREE,
        description=(
            "Elementary divisors d_1 | d_2 | ... of the class group; empty for "
            "the trivial group."
        ),
    )
    field_discriminant: NumberFieldDiscriminantInteger
    real_embedding_count: int = Field(ge=0, le=MAX_CLASS_GROUP_DEGREE)
    complex_embedding_pair_count: int = Field(ge=0, le=MAX_CLASS_GROUP_DEGREE)
    ideal_representatives: tuple[IntegerMatrix, ...] = Field(
        max_length=MAX_CLASS_GROUP_DEGREE,
        description=(
            "Hermite-normal-form generator matrices over the field's power "
            "basis, one per nontrivial invariant when available."
        ),
    )

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class NumberFieldUnitGroupRequest(StrictModel):
    """A simple number field within the PARI unit-group envelope."""

    field: _IntegralBasisField

    @model_validator(mode="after")
    def require_class_group_degree(self) -> NumberFieldUnitGroupRequest:
        if self.field.degree > MAX_CLASS_GROUP_DEGREE:
            raise _validation_error(
                "unit_group_degree_bound",
                "unit-group computation admits degree at most "
                f"{MAX_CLASS_GROUP_DEGREE}",
            )
        return self


class NumberFieldUnitGroupResult(StrictModel):
    """The unit group of one presented field with its exact fundamental units."""

    field: SimpleNumberFieldPresentation
    rank: int = Field(ge=0, le=MAX_CLASS_GROUP_DEGREE)
    torsion_order: int = Field(ge=1, le=2 * MAX_CLASS_GROUP_DEGREE)
    torsion_generator: SimpleNumberFieldElement
    fundamental_units: tuple[SimpleNumberFieldElement, ...] = Field(
        max_length=MAX_CLASS_GROUP_DEGREE
    )
    field_discriminant: NumberFieldDiscriminantInteger
    real_embedding_count: int = Field(ge=0, le=MAX_CLASS_GROUP_DEGREE)
    complex_embedding_pair_count: int = Field(ge=0, le=MAX_CLASS_GROUP_DEGREE)

    @model_validator(mode="after")
    def require_unit_decomposition(self) -> Self:
        if self.rank != len(self.fundamental_units):
            raise _validation_error(
                "unit_group_rank_shape",
                "rank must equal the number of fundamental units",
            )
        expected_rank = (
            self.real_embedding_count + self.complex_embedding_pair_count - 1
        )
        if self.rank != max(0, expected_rank):
            raise _validation_error(
                "unit_group_rank_signature",
                "rank must equal r1 + r2 - 1 for the field signature",
            )
        for element in (self.torsion_generator, *self.fundamental_units):
            if element.presentation != self.field:
                raise _validation_error(
                    "unit_group_element_parent",
                    "unit elements must use the field presentation",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_CLASS_GROUP_DEGREE",
    "MAX_INTEGRAL_BASIS_DEGREE",
    "NumberFieldClassGroupRequest",
    "NumberFieldClassGroupResult",
    "NumberFieldDiscriminantResult",
    "NumberFieldEmbeddingsRequest",
    "NumberFieldRealEmbeddingOrderRequest",
    "NumberFieldRequest",
    "NumberFieldRingOfIntegersRequest",
    "NumberFieldUnitGroupRequest",
    "NumberFieldUnitGroupResult",
]
