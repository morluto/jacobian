"""Bounded signed presolve used only to refine refused Lie-result envelopes.

The original formula remains the executor's input. These checks neither replace
SymPy nor establish canonical source recognition or remove inherited loci.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundsLedger,
    FractionBound,
    _add_fractions,
    _check_raw_polynomial,
    _one_polynomial,
    _polynomial_admission_work_units,
    _polynomial_bound,
    _remove_guaranteed_common_monomial,
    _zero_fraction,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

if TYPE_CHECKING:
    from jacobian.math.geometry.differential._bounds import (
        FactorReference,
        LieProductTerm,
    )


type _TermKey = tuple[tuple[int, ...], int, int]


@dataclass(frozen=True)
class _FactorIdentity:
    variables: tuple[str, ...]
    numerator: tuple[_TermKey, ...]
    denominator: tuple[_TermKey, ...]
    derivative_axis: int | None


class _Presolve:
    def __init__(
        self,
        vector: RationalCoordinateTensor,
        tensor: RationalCoordinateTensor,
        ledger: BoundsLedger,
    ) -> None:
        self.vector = vector
        self.tensor = tensor
        self.ledger = ledger
        self.dimension = len(tensor.coordinate_axis)
        self.polynomials: dict[
            FactorReference, tuple[tuple[tuple[int, ...], Fraction], ...]
        ] = {}
        self.identities: dict[FactorReference, tuple[_FactorIdentity, Fraction]] = {}

    def value(self, reference: FactorReference) -> RationalFunction:
        owner = self.vector if reference.owner == "VECTOR" else self.tensor
        return owner.components[reference.component]

    def charge_scalar(
        self,
        bits: int,
        category: Literal["source_conversion", "multiplication", "addition"],
    ) -> None:
        # 2^(3*d) < 10^d: this conservative bit cap bounds every temporary
        # scalar before Fraction multiplication, division, addition or gcd.
        if bits > 3 * self.ledger.limits.raw_digits:
            self.ledger.limits.reject(
                "presolve_height", "Lie signed presolve exceeds its scalar envelope"
            )
        self.ledger.charge(category, max(1, (bits + 31) // 32) ** 2)

    def scalar(self, value: CanonicalRational) -> Fraction:
        self.charge_scalar(
            max(value.num.bit_length(), value.den.bit_length()), "source_conversion"
        )
        return Fraction(value.num, value.den)

    def multiply(
        self, left: Fraction, right: Fraction, *, divide: bool = False
    ) -> Fraction:
        numerator = right.denominator if divide else right.numerator
        denominator = right.numerator if divide else right.denominator
        bits = max(
            left.numerator.bit_length() + numerator.bit_length(),
            left.denominator.bit_length() + denominator.bit_length(),
        )
        self.charge_scalar(bits, "multiplication")
        return left / right if divide else left * right

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        bits = max(
            left.numerator.bit_length() + right.denominator.bit_length() + 1,
            right.numerator.bit_length() + left.denominator.bit_length() + 1,
            left.denominator.bit_length() + right.denominator.bit_length(),
        )
        self.charge_scalar(bits, "addition")
        return left + right

    def polynomial_terms(
        self, reference: FactorReference
    ) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
        if reference in self.polynomials:
            return self.polynomials[reference]
        source = self.value(reference)
        self.ledger.charge(
            "source_conversion", len(source.numerator.terms) * (self.dimension + 1)
        )
        rows: list[tuple[tuple[int, ...], Fraction]] = []
        for term in source.numerator.terms:
            exponent = term.exponents
            axis = reference.derivative_axis
            if axis is not None and exponent[axis] == 0:
                continue
            coefficient = self.scalar(term.coefficient)
            if axis is not None:
                coefficient = self.multiply(coefficient, Fraction(exponent[axis]))
                exponent = tuple(
                    power - int(index == axis) for index, power in enumerate(exponent)
                )
            rows.append((exponent, coefficient))
        result = tuple(rows)
        self.polynomials[reference] = result
        return result

    def identity(self, reference: FactorReference) -> tuple[_FactorIdentity, Fraction]:
        if reference in self.identities:
            return self.identities[reference]
        source = self.value(reference)
        self.ledger.charge(
            "source_conversion",
            (len(source.numerator.terms) + len(source.denominator.terms))
            * (self.dimension + 3),
        )
        scale = (
            self.scalar(source.numerator.terms[0].coefficient)
            if source.numerator.terms
            else Fraction(0)
        )
        numerator: list[_TermKey] = []
        for term in source.numerator.terms:
            coefficient = self.multiply(
                self.scalar(term.coefficient), scale, divide=True
            )
            numerator.append(
                (term.exponents, coefficient.numerator, coefficient.denominator)
            )
        denominator = tuple(
            (term.exponents, term.coefficient.num, term.coefficient.den)
            for term in source.denominator.terms
        )
        result = (
            _FactorIdentity(
                source.variables,
                tuple(numerator),
                denominator,
                reference.derivative_axis,
            ),
            scale,
        )
        self.identities[reference] = result
        return result


def _unit_denominator(value: RationalFunction) -> bool:
    terms = value.denominator.terms
    return (
        len(terms) == 1
        and not any(terms[0].exponents)
        and terms[0].coefficient.as_integer_ratio() == (1, 1)
    )


def _polynomial_refinement(
    terms: tuple[LieProductTerm, ...], presolve: _Presolve
) -> FractionBound:
    coefficients: dict[tuple[int, ...], Fraction] = {}
    for term in terms:
        left = presolve.polynomial_terms(term.left)
        right = presolve.polynomial_terms(term.right)
        presolve.ledger.charge(
            "multiplication", len(left) * len(right) * (presolve.dimension + 1)
        )
        for left_exponent, left_coefficient in left:
            for right_exponent, right_coefficient in right:
                exponent = tuple(
                    a + b for a, b in zip(left_exponent, right_exponent, strict=True)
                )
                if (
                    exponent not in coefficients
                    and len(coefficients) >= presolve.ledger.limits.raw_terms
                ):
                    presolve.ledger.limits.reject(
                        "presolve_support",
                        "Lie signed presolve exceeds its intermediate support envelope",
                    )
                coefficient = presolve.multiply(left_coefficient, right_coefficient)
                updated = presolve.add(
                    coefficients.get(exponent, Fraction(0)),
                    coefficient if term.sign > 0 else -coefficient,
                )
                if updated:
                    coefficients[exponent] = updated
                else:
                    coefficients.pop(exponent, None)
    if not coefficients:
        return _zero_fraction(presolve.dimension)
    # Bound denominator clearing before the existing content/height helper.
    denominator_bits = sum(
        denominator.bit_length()
        for denominator in {
            value.denominator
            for value in coefficients.values()
            if value.denominator != 1
        }
    )
    clear_bits = (
        max(value.numerator.bit_length() for value in coefficients.values())
        + denominator_bits
    )
    presolve.charge_scalar(clear_bits, "source_conversion")
    presolve.ledger.charge(
        "source_conversion",
        len(coefficients)
        * (6 + 3 * presolve.dimension)
        * max(1, (clear_bits + 31) // 32) ** 2,
    )
    polynomial = SparseRationalPolynomial.model_construct(
        terms=tuple(
            RationalPolynomialTerm.model_construct(
                exponents=exponent,
                coefficient=CanonicalRational.model_construct(
                    num=value.numerator, den=value.denominator
                ),
            )
            for exponent, value in sorted(coefficients.items(), reverse=True)
        )
    )
    presolve.ledger.charge(
        "source_conversion",
        _polynomial_admission_work_units(polynomial, presolve.dimension),
    )
    bound = _polynomial_bound(polynomial)
    _check_raw_polynomial(bound, presolve.ledger.limits)
    return FractionBound(bound, _one_polynomial(presolve.dimension))


def refine_signed_result(
    terms: tuple[LieProductTerm, ...],
    product_bounds: tuple[FractionBound, ...],
    vector: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
    ledger: BoundsLedger,
) -> FractionBound:
    """Refine only final support/height; original execution work stays charged."""
    presolve = _Presolve(vector, tensor, ledger)
    references = {reference for term in terms for reference in (term.left, term.right)}
    if all(_unit_denominator(presolve.value(reference)) for reference in references):
        return _polynomial_refinement(terms, presolve)
    groups: dict[frozenset[_FactorIdentity], Fraction] = {}
    keys: list[frozenset[_FactorIdentity]] = []
    for term in terms:
        left, left_scale = presolve.identity(term.left)
        right, right_scale = presolve.identity(term.right)
        # Each product has exactly two factors: a singleton key uniquely
        # denotes a square, so this unordered key preserves multiplicities.
        key = frozenset((left, right))
        keys.append(key)
        scale = presolve.multiply(left_scale, right_scale)
        groups[key] = presolve.add(
            groups.get(key, Fraction(0)), scale if term.sign > 0 else -scale
        )
    bound = _zero_fraction(presolve.dimension)
    for key, product_bound in zip(keys, product_bounds, strict=True):
        if groups[key]:
            bound = _add_fractions(bound, product_bound, ledger)
    return _remove_guaranteed_common_monomial(bound)
