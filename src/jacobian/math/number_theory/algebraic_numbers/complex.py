"""Canonical exact nonreal algebraic values and rational isolation evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from math import gcd
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, ValidateAs, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    MAX_REAL_ALGEBRAIC_DEGREE,
)

MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"complex_algebraic.{reason}", message)


def _integer_coefficients(
    coefficients: Sequence[CanonicalInteger],
) -> tuple[int, ...]:
    return tuple(parse_canonical_integer(coefficient) for coefficient in coefficients)


def algebraic_root_separation_denominator_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Return ``B`` such that distinct roots have distance greater than ``1/B``.

    For a square-free integer polynomial of degree ``n``, Mignotte's bound is

    ``sep(f) > sqrt(3) / (n^((n+2)/2) * ||f||_2^(n-1))``.

    With coefficient height ``H``, ``||f||_2 <= sqrt(n+1) H``.  Replacing both
    square roots by larger integer factors gives the deliberately conservative
    denominator below.  The public carriers independently require
    irreducibility, hence square-freeness in characteristic zero.
    """

    integers = _integer_coefficients(coefficients)
    degree = len(integers) - 1
    if degree <= 1:
        return 1
    height: int = max(abs(coefficient) for coefficient in integers)
    degree_factor: int = degree ** ((degree + 3) // 2)
    norm_factor: int = ((degree + 1) * height) ** (degree - 1)
    return int(degree_factor * norm_factor)


@lru_cache(maxsize=128)
def _real_part_elimination_polynomial(
    coefficients: tuple[int, ...],
) -> tuple[int, ...]:
    """Eliminate the imaginary coordinate from ``f(u + i*v) = 0`` exactly."""

    import sympy

    real_coordinate, imaginary_coordinate = sympy.symbols("u v", real=True)
    degree = len(coefficients) - 1
    argument = real_coordinate + sympy.I * imaginary_coordinate
    expression = sum(
        coefficient * argument ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )
    real_part, imaginary_part = map(sympy.expand, expression.as_real_imag())
    resultant = sympy.resultant(real_part, imaginary_part, imaginary_coordinate)
    polynomial = sympy.Poly(resultant, real_coordinate, domain=sympy.ZZ)
    _content, primitive = polynomial.primitive()
    square_free = primitive.sqf_part()
    if square_free.LC() < 0:
        square_free = -square_free
    return tuple(int(coefficient) for coefficient in square_free.all_coeffs())


def algebraic_real_part_separation_denominator_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Bound separation between distinct real coordinates of roots of ``f``."""

    integer_coefficients = _integer_coefficients(coefficients)
    elimination = _real_part_elimination_polynomial(integer_coefficients)
    degree = len(elimination) - 1
    if degree <= 1:
        return 1
    height: int = max(abs(coefficient) for coefficient in elimination)
    bound: int = degree ** ((degree + 3) // 2) * ((degree + 1) * height) ** (degree - 1)
    return int(bound)


def algebraic_root_magnitude_numerator_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Return an integer strictly above every root magnitude (Cauchy's bound)."""

    integers = _integer_coefficients(coefficients)
    leading = abs(integers[0])
    tail_height = max((abs(coefficient) for coefficient in integers[1:]), default=0)
    return 2 + tail_height // leading


def complex_isolator_component_digit_bound(
    coefficients: Sequence[CanonicalInteger],
) -> int:
    """Bound decimal component digits of the dyadic evidence constructed here."""

    separation_denominator = algebraic_root_separation_denominator_bound(coefficients)
    grid_bits = separation_denominator.bit_length() + 4
    magnitude_bits = algebraic_root_magnitude_numerator_bound(coefficients).bit_length()
    component_bits = grid_bits + magnitude_bits + 3
    # 0.30103 is a strict decimal upper bound for log10(2).
    return (component_bits * 30_103) // 100_000 + 2


class RationalComplexIsolatingRectangle(StrictModel):
    """A closed rational rectangle carrying one certified nonreal root.

    All four boundary segments are included.  Embedding records additionally
    prove that a rational error box for their selected exact root lies strictly
    inside this rectangle, so boundary-root ambiguities in backend root counts
    cannot affect the public identity.
    """

    real_lower: CanonicalRational
    real_upper: CanonicalRational
    imaginary_lower: CanonicalRational
    imaginary_upper: CanonicalRational
    boundary: Literal["CLOSED"] = "CLOSED"

    @model_validator(mode="before")
    @classmethod
    def require_raw_component_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        for name in (
            "real_lower",
            "real_upper",
            "imaginary_lower",
            "imaginary_upper",
        ):
            component = data.get(name)
            if not isinstance(component, Mapping):
                continue
            for part in ("num", "den"):
                raw = component.get(part)
                if isinstance(raw, str) and len(raw.lstrip("-")) > (
                    MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS
                ):
                    raise _validation_error(
                        "isolator_component_bound",
                        "complex isolator components exceed the "
                        f"{MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS:,}-digit bound",
                    )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_nonempty_rectangle(self) -> Self:
        if any(
            len(component.num.lstrip("-")) > MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS
            or len(component.den) > MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS
            for component in (
                self.real_lower,
                self.real_upper,
                self.imaginary_lower,
                self.imaginary_upper,
            )
        ):
            raise _validation_error(
                "isolator_component_bound",
                "complex isolator components exceed the "
                f"{MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS:,}-digit bound",
            )
        if self.real_lower.as_fraction() >= self.real_upper.as_fraction():
            raise _validation_error(
                "real_axis_order",
                "complex isolator real bounds must be strictly increasing",
            )
        if self.imaginary_lower.as_fraction() >= self.imaginary_upper.as_fraction():
            raise _validation_error(
                "imaginary_axis_order",
                "complex isolator imaginary bounds must be strictly increasing",
            )
        return self

    def conjugate(self) -> RationalComplexIsolatingRectangle:
        """Reflect this rectangle exactly across the real axis."""

        return RationalComplexIsolatingRectangle(
            real_lower=self.real_lower,
            real_upper=self.real_upper,
            imaginary_lower=CanonicalRational.from_fraction(
                -self.imaginary_upper.as_fraction()
            ),
            imaginary_upper=CanonicalRational.from_fraction(
                -self.imaginary_lower.as_fraction()
            ),
        )


class _ComplexAlgebraicValueShape(StrictModel):
    """Canonical structural representation of an indexed nonreal root.

    ``polynomial`` is primitive irreducible in ``ZZ[x]`` with positive leading
    coefficient.  ``root_index`` uses the mathematical global order: all real
    roots increasingly first; then conjugate pairs ordered lexicographically
    by their positive-imaginary representative ``(Re(z), Im(z))``; and within
    each pair the negative root precedes the positive root.  Thus the
    polynomial and index, rather than any one of infinitely many valid
    isolating rectangles, are the value's identity.

    This structural view checks the bounded canonical representation. A public
    value constructor or mathematical consumer must additionally recognize
    irreducibility and that the index selects a nonreal root.
    """

    polynomial: tuple[CanonicalInteger, ...] = Field(
        min_length=2,
        max_length=MAX_REAL_ALGEBRAIC_DEGREE + 1,
        description=(
            "Primitive irreducible ZZ[x] coefficients in descending degree, "
            "with positive leading coefficient."
        ),
    )
    root_index: StrictInt = Field(
        ge=0,
        le=MAX_REAL_ALGEBRAIC_DEGREE - 1,
        description=(
            "Index in the real-first, positive-representative conjugate-pair "
            "order declared by ComplexAlgebraicValue."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_polynomial_bound(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        polynomial = data.get("polynomial")
        if isinstance(polynomial, (list, tuple)):
            if len(polynomial) > MAX_REAL_ALGEBRAIC_DEGREE + 1:
                raise _validation_error(
                    "degree_bound",
                    "complex algebraic degree exceeds the bounded root envelope",
                )
            if any(
                isinstance(value, str)
                and len(value.lstrip("-")) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
                for value in polynomial
            ):
                raise _validation_error(
                    "coefficient_bound",
                    "complex algebraic polynomial coefficients exceed the "
                    f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit bound",
                )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_indexed_root_shape(self) -> Self:
        coefficients = _integer_coefficients(self.polynomial)
        if coefficients[0] <= 0:
            raise _validation_error(
                "leading_sign",
                "complex algebraic minimal polynomial must have positive leading coefficient",
            )
        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise _validation_error(
                "not_primitive",
                "complex algebraic minimal polynomial must be primitive over ZZ",
            )

        if self.root_index >= len(self.polynomial) - 1:
            raise _validation_error(
                "root_index",
                "root_index must be smaller than the polynomial degree",
            )
        return self


class ComplexAlgebraicValue(_ComplexAlgebraicValueShape):
    """Structural indexed-root value established by an admitted result owner.

    Parsing checks only its bounded canonical shape. Mathematical consumers
    must recognize irreducibility and the selected nonreal root within their
    own admitted execution path; model validation never replays that work.
    """

    @classmethod
    def _from_admitted_polynomial(
        cls,
        *,
        polynomial: tuple[CanonicalInteger, ...],
        root_index: int,
    ) -> ComplexAlgebraicValue:
        """Construct after an owner has admitted the canonical polynomial/root."""

        return cls.model_construct(polynomial=polynomial, root_index=root_index)


def _unrecognized_complex_value_from_shape(
    shape: _ComplexAlgebraicValueShape,
) -> ComplexAlgebraicValue:
    if isinstance(shape, ComplexAlgebraicValue):
        return shape
    return ComplexAlgebraicValue.model_construct(
        polynomial=shape.polynomial,
        root_index=shape.root_index,
    )


_UnrecognizedComplexAlgebraicValue = Annotated[
    ComplexAlgebraicValue,
    ValidateAs(_ComplexAlgebraicValueShape, _unrecognized_complex_value_from_shape),
    WithJsonSchema(ComplexAlgebraicValue.model_json_schema()),
]


__all__ = [
    "MAX_COMPLEX_ISOLATOR_COMPONENT_DIGITS",
    "ComplexAlgebraicValue",
    "RationalComplexIsolatingRectangle",
    "_UnrecognizedComplexAlgebraicValue",
    "algebraic_real_part_separation_denominator_bound",
    "algebraic_root_magnitude_numerator_bound",
    "algebraic_root_separation_denominator_bound",
    "complex_isolator_component_digit_bound",
]
