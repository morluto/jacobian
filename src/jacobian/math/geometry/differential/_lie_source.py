"""Bounded structural source checks for cancellation-aware Lie admission."""

from __future__ import annotations

import re

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_EXPONENT,
    MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
    MAX_RATIONAL_TENSOR_RANK,
    RationalCoordinateTensor,
)
from jacobian.math.polynomials.rational_functions._bounds import BoundsLedger
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_COMPONENT_LIMIT = 10**MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS


def _invalid() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=(),
        code="differential_geometry.lie_derivative.source_not_canonical",
        message="Lie sources must retain canonical tensor, scalar and locus structure",
    )


def _polynomial_work(
    value: object, dimension: int, exponent_limit: int, ledger: BoundsLedger
) -> int:
    if (
        type(value) is not SparseRationalPolynomial
        or type(getattr(value, "terms", None)) is not tuple
        or len(value.terms) > MAX_RATIONAL_FUNCTION_TERMS
    ):
        raise _invalid()
    ledger.charge("source_conversion", 1 + len(value.terms) * (16 + 3 * dimension))
    previous: tuple[int, ...] | None = None
    rebuild_work = 1
    for term in value.terms:
        if (
            type(term) is not RationalPolynomialTerm
            or type(getattr(term, "exponents", None)) is not tuple
            or len(term.exponents) != dimension
            or any(
                type(power) is not int or not 0 <= power <= exponent_limit
                for power in term.exponents
            )
            or (previous is not None and term.exponents >= previous)
            or type(getattr(term, "coefficient", None)) is not CanonicalRational
        ):
            raise _invalid()
        previous = term.exponents
        coefficient = term.coefficient
        if (
            type(getattr(coefficient, "num", None)) is not int
            or type(getattr(coefficient, "den", None)) is not int
            or coefficient.num == 0
            or coefficient.den <= 0
            or max(coefficient.num.bit_length(), coefficient.den.bit_length())
            > 4 * MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
            or abs(coefficient.num) >= _COMPONENT_LIMIT
            or coefficient.den >= _COMPONENT_LIMIT
        ):
            raise _invalid()
        bits = max(coefficient.num.bit_length(), coefficient.den.bit_length())
        rebuild_work += (16 + 3 * dimension) * max(1, (bits + 31) // 32) ** 2
    return rebuild_work


def _source_work(source: RationalCoordinateTensor, ledger: BoundsLedger) -> int:
    if (
        type(source) is not RationalCoordinateTensor
        or type(getattr(source, "coordinate_axis", None)) is not tuple
        or not 1 <= len(source.coordinate_axis) <= MAX_POLYNOMIAL_VARIABLES
        or any(
            type(name) is not str
            or len(name) > 32
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name) is None
            for name in source.coordinate_axis
        )
        or len(set(source.coordinate_axis)) != len(source.coordinate_axis)
        or type(getattr(source, "variance", None)) is not tuple
        or len(source.variance) > MAX_RATIONAL_TENSOR_RANK
        or any(value not in ("CONTRAVARIANT", "COVARIANT") for value in source.variance)
        or type(getattr(source, "components", None)) is not tuple
        or not 1 <= len(source.components) <= MAX_RATIONAL_TENSOR_COMPONENTS
        or len(source.components) != len(source.coordinate_axis) ** len(source.variance)
        or type(getattr(source, "retained_nonzero_denominators", None)) is not tuple
        or len(source.retained_nonzero_denominators) > MAX_RATIONAL_TENSOR_LOCUS_GUARDS
    ):
        raise _invalid()
    dimension = len(source.coordinate_axis)
    ledger.charge(
        "source_conversion",
        1
        + dimension
        + len(source.components)
        + len(source.retained_nonzero_denominators),
    )
    work = 1
    for component in source.components:
        if (
            type(component) is not RationalFunction
            or getattr(component, "variables", None) != source.coordinate_axis
        ):
            raise _invalid()
        work += _polynomial_work(
            getattr(component, "numerator", None),
            dimension,
            MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
            ledger,
        )
        work += _polynomial_work(
            getattr(component, "denominator", None),
            dimension,
            MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
            ledger,
        )
        if not component.denominator.terms or component.denominator.terms[
            0
        ].coefficient.as_integer_ratio() != (1, 1):
            raise _invalid()
    for guard in source.retained_nonzero_denominators:
        work += _polynomial_work(guard, dimension, MAX_RATIONAL_TENSOR_EXPONENT, ledger)
        if not guard.terms:
            raise _invalid()
    return work


class LieSourceAdmission:
    """Preflight once, then revalidate structural invariants once if refining."""

    def __init__(
        self,
        vector: RationalCoordinateTensor,
        tensor: RationalCoordinateTensor,
        ledger: BoundsLedger,
    ) -> None:
        self.sources = (vector, tensor)
        self.ledger = ledger
        self.rebuild_work = sum(_source_work(source, ledger) for source in self.sources)
        self.validated = False

    def require_canonical(self) -> None:
        if self.validated:
            return
        self.ledger.charge("source_conversion", self.rebuild_work)
        try:
            for source in self.sources:
                # Every container and scalar was bounded before this copy.
                # This is structural decoding, not the coprimality replay.
                RationalCoordinateTensor.model_validate(
                    source.model_dump(mode="python")
                )
        except ValidationError as error:
            raise _invalid() from error
        self.validated = True
