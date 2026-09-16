"""Typed wire contracts for finite function-field operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_CHARACTERISTIC = 257
MAX_EXTENSION_DEGREE = 6
MAX_POLYNOMIAL_X_DEGREE = 12
MAX_POLYNOMIAL_COEFFICIENTS = MAX_POLYNOMIAL_X_DEGREE + 1
MAX_LEDGER_ROWS = 64
MAX_FIELD_ADMISSION_WORK = 4_000_000
MAX_MULTIPLICATION_WORK = 2_000_000

FieldVariable = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,15}$", strict=True)
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"function_field.{reason}", message)


class PrimeFieldPolynomial(StrictModel):
    """A canonical univariate polynomial over GF(characteristic).

    Coefficients are ascending and reduced to ``0..p-1``; the zero polynomial
    has the unique spelling ``[0]`` and every nonzero polynomial has a nonzero
    leading coefficient.
    """

    characteristic: int = Field(ge=2, le=MAX_CHARACTERISTIC)
    coefficients: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_COEFFICIENTS
    )

    @model_validator(mode="after")
    def require_canonical_coefficients(self) -> Self:
        if any(
            type(value) is not int or value < 0 or value >= self.characteristic
            for value in self.coefficients
        ):
            raise _validation_error(
                "polynomial_coefficients",
                "coefficients must be canonical residues of the characteristic",
            )
        if len(self.coefficients) > 1 and self.coefficients[-1] == 0:
            raise _validation_error(
                "polynomial_trailing_zero",
                "a nonzero polynomial cannot have a trailing zero coefficient",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def is_zero(self) -> bool:
        return self.coefficients == (0,)

    def is_one(self) -> bool:
        return self.coefficients == (1,)


class PrimeFieldRationalFunction(StrictModel):
    """A presentation ``numerator/denominator`` over GF(characteristic).

    Deserialization checks only the bounded shape and a nonzero denominator.
    Canonical reduced form (coprime numerator/denominator and monic
    denominator) is established by the consuming function-field admission.
    """

    numerator: PrimeFieldPolynomial
    denominator: PrimeFieldPolynomial

    @model_validator(mode="after")
    def require_common_nonzero_fraction(self) -> Self:
        if self.numerator.characteristic != self.denominator.characteristic:
            raise _validation_error(
                "rational_function_characteristic",
                "numerator and denominator must share one characteristic",
            )
        if self.denominator.is_zero():
            raise _validation_error(
                "rational_function_zero_denominator",
                "a rational function requires a nonzero denominator",
            )
        return self

    @property
    def characteristic(self) -> int:
        return self.numerator.characteristic


class FiniteFunctionField(StrictModel):
    """One monic separable extension ``GF(p)(x)[y]/(f)``."""

    characteristic: int = Field(ge=2, le=MAX_CHARACTERISTIC)
    variable: FieldVariable = Field(
        default="x", description="Name of the rational-function variable."
    )
    generator: FieldVariable = Field(
        default="y", description="Name of the extension generator y."
    )
    defining_polynomial: tuple[PrimeFieldRationalFunction, ...] = Field(
        min_length=2, max_length=MAX_EXTENSION_DEGREE + 1
    )

    @model_validator(mode="after")
    def require_monic_extension(self) -> Self:
        if self.variable == self.generator:
            raise _validation_error(
                "extension_variable_collision",
                "the rational variable and extension generator must be distinct",
            )
        if any(
            coefficient.characteristic != self.characteristic
            for coefficient in self.defining_polynomial
        ):
            raise _validation_error(
                "extension_characteristic",
                "every defining-polynomial coefficient must share the characteristic",
            )
        if not self.defining_polynomial[-1].numerator.is_one() or not (
            self.defining_polynomial[-1].denominator.is_one()
        ):
            raise _validation_error(
                "extension_monic",
                "the defining polynomial must be monic in the generator",
            )
        if len(self.defining_polynomial) - 1 > MAX_EXTENSION_DEGREE:
            raise _validation_error(
                "extension_degree_bound",
                "extension degree exceeds the admitted bound",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.defining_polynomial) - 1


class FiniteFunctionFieldElement(StrictModel):
    """Reduced coordinates in the generator-power basis of one function field."""

    field: FiniteFunctionField
    coordinates: tuple[PrimeFieldRationalFunction, ...] = Field(
        min_length=1, max_length=MAX_EXTENSION_DEGREE
    )

    @model_validator(mode="after")
    def require_reduced_coordinates(self) -> Self:
        if len(self.coordinates) != self.field.degree:
            raise _validation_error(
                "element_coordinate_count",
                "coordinates must have exactly one entry per power-basis element",
            )
        if any(
            coordinate.characteristic != self.field.characteristic
            for coordinate in self.coordinates
        ):
            raise _validation_error(
                "element_characteristic",
                "every coordinate must share the field characteristic",
            )
        return self


class FunctionFieldElementMultiplyRequest(StrictModel):
    """Two reduced elements of one declared finite function field."""

    left: FiniteFunctionFieldElement
    right: FiniteFunctionFieldElement


class FunctionFieldProductTerm(StrictModel):
    """One nonzero generator-power term of an exact product."""

    y_power: int = Field(ge=0, le=2 * MAX_EXTENSION_DEGREE)
    coefficient: PrimeFieldRationalFunction


class FunctionFieldReductionStep(StrictModel):
    """One exact reduction of a high generator power modulo the defining polynomial."""

    y_power: int = Field(ge=1, le=2 * MAX_EXTENSION_DEGREE)
    coefficient: PrimeFieldRationalFunction
    replacement_terms: tuple[FunctionFieldProductTerm, ...] = Field(
        max_length=MAX_LEDGER_ROWS
    )


class FunctionFieldElementMultiplyResult(StrictModel):
    """The canonical reduced product with its complete reduction ledger."""

    field: FiniteFunctionField
    left: FiniteFunctionFieldElement
    right: FiniteFunctionFieldElement
    raw_product_terms: tuple[FunctionFieldProductTerm, ...] = Field(
        max_length=MAX_LEDGER_ROWS
    )
    reduction_steps: tuple[FunctionFieldReductionStep, ...] = Field(
        max_length=MAX_LEDGER_ROWS
    )
    product: FiniteFunctionFieldElement

    @model_validator(mode="after")
    def require_bound_result(self) -> Self:
        if (
            self.left.field != self.field
            or self.right.field != self.field
            or self.product.field != self.field
        ):
            raise _validation_error(
                "result_field",
                "inputs and product must share the declared function field",
            )
        if tuple(step.y_power for step in self.reduction_steps) != tuple(
            sorted(step.y_power for step in self.reduction_steps)
        ):
            raise _validation_error(
                "reduction_order",
                "reduction steps must be ordered by generator power",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        field: FiniteFunctionField,
        left: FiniteFunctionFieldElement,
        right: FiniteFunctionFieldElement,
        raw_product_terms: tuple[FunctionFieldProductTerm, ...],
        reduction_steps: tuple[FunctionFieldReductionStep, ...],
        product: FiniteFunctionFieldElement,
    ) -> FunctionFieldElementMultiplyResult:
        return cls.model_construct(
            field=field,
            left=left,
            right=right,
            raw_product_terms=raw_product_terms,
            reduction_steps=reduction_steps,
            product=product,
        )


__all__ = [
    "MAX_CHARACTERISTIC",
    "MAX_EXTENSION_DEGREE",
    "MAX_FIELD_ADMISSION_WORK",
    "MAX_LEDGER_ROWS",
    "MAX_MULTIPLICATION_WORK",
    "MAX_POLYNOMIAL_COEFFICIENTS",
    "MAX_POLYNOMIAL_X_DEGREE",
    "FiniteFunctionField",
    "FiniteFunctionFieldElement",
    "FunctionFieldElementMultiplyRequest",
    "FunctionFieldElementMultiplyResult",
    "FunctionFieldProductTerm",
    "FunctionFieldReductionStep",
    "PrimeFieldPolynomial",
    "PrimeFieldRationalFunction",
]
