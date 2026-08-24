"""Typed wire contracts for polynomial map operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.maps.values import (
    MAX_MAP_INPUTS,
    MAX_MAP_OUTPUTS,
    RationalPolynomialMap,
    require_map_polynomial,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    PolynomialVariable,
    RationalFunction,
    RationalPolynomial,
)

_MAX_COMPOSITION_DEGREE = 128

MAX_GENERIC_DEGREE_SOURCE_VARIABLES = 3
MAX_GENERIC_DEGREE_TARGET_VARIABLES = 3
MAX_GENERIC_DEGREE_COMPONENT_TERMS = 48
MAX_GENERIC_DEGREE_AGGREGATE_TERMS = 96
MAX_GENERIC_DEGREE_TOTAL_DEGREE = 8
MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS = 64
MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES = 64 * 1024
MAX_GENERIC_DEGREE_BEZOUT_BOUND = 512
MAX_GENERIC_FIBER_BASIS_POLYNOMIALS = 32
MAX_GENERIC_FIBER_POLYNOMIAL_TERMS = 4_096
MAX_GENERIC_FIBER_CERTIFICATE_TERMS = 4_096
MAX_GENERIC_FIBER_COEFFICIENT_TERMS = 16_384
MAX_GENERIC_FIBER_STANDARD_MONOMIALS = 512
MAX_GENERIC_FIBER_CERTIFICATE_SOURCE_EXPONENT = MAX_POLYNOMIAL_EXPONENT
MAX_GENERIC_FIBER_STANDARD_MONOMIAL_EXPONENT = (
    MAX_GENERIC_FIBER_STANDARD_MONOMIALS - 1
)
MAX_GENERIC_FIBER_STANDARD_MONOMIAL_CANDIDATES = (
    1
    + MAX_GENERIC_DEGREE_SOURCE_VARIABLES
    * MAX_GENERIC_FIBER_STANDARD_MONOMIALS
)
MAX_GENERIC_FIBER_REPLAY_PRODUCTS = 262_144
MAX_GENERIC_FIBER_REPLAY_SOURCE_PRODUCTS = 262_144
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_OPERATIONS = 1_048_576
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_PRODUCTS = 1_048_576
MAX_GENERIC_FIBER_REPLAY_REDUCTION_STEPS = 65_536
MAX_GENERIC_FIBER_REPLAY_SOURCE_TERMS = 8_192
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_TERMS = 4_096
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_BITS = 16_384
MAX_GENERIC_FIBER_REPLAY_PARAMETER_EXPONENT = MAX_POLYNOMIAL_EXPONENT
MAX_GENERIC_FIBER_REPLAY_SOURCE_EXPONENT = 2 * MAX_POLYNOMIAL_EXPONENT


class VariablePoint(StrictModel):
    """One rational point on an explicitly ordered polynomial axis."""

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_MAP_INPUTS
    )
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_MAP_INPUTS
    )

    @model_validator(mode="after")
    def require_matching_axis(self) -> Self:
        if len(self.variables) != len(self.values):
            raise ValueError("point variables and values must have the same length")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("point variables must be unique")
        return self


class EvalRequest(StrictModel):
    """Evaluate one canonical rational polynomial at a complete rational point."""

    polynomial: RationalPolynomial
    point: VariablePoint

    @model_validator(mode="after")
    def require_complete_ordered_point(self) -> Self:
        require_map_polynomial(self.polynomial, label="evaluation polynomial")
        if self.point.variables != self.polynomial.variables:
            raise ValueError(
                "evaluation point must use the polynomial's complete ordered axis"
            )
        value = Fraction(0)
        coordinates = tuple(item.as_fraction() for item in self.point.values)
        for term in self.polynomial.polynomial.terms:
            monomial = term.coefficient.as_fraction()
            for coordinate, exponent in zip(coordinates, term.exponents, strict=True):
                monomial *= coordinate**exponent
            value += monomial
        CanonicalRational.from_fraction(value)
        return self


class EvalResult(StrictModel):
    """The exact rational value at the requested point."""

    value: CanonicalRational


JacobianRequest = RationalPolynomialMap


class JacobianResult(StrictModel):
    """The row-major Jacobian matrix over the source polynomial ring."""

    n_inputs: int = Field(ge=1, le=MAX_MAP_INPUTS)
    n_outputs: int = Field(ge=1, le=MAX_MAP_OUTPUTS)
    entries: tuple[RationalPolynomial, ...] = Field(max_length=160)

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if len(self.entries) != self.n_inputs * self.n_outputs:
            raise ValueError("Jacobian entry count must match its matrix dimensions")
        if self.entries:
            variables = self.entries[0].variables
            if any(entry.variables != variables for entry in self.entries):
                raise ValueError("Jacobian entries must use one ordered ring")
        return self


class CompositionRequest(StrictModel):
    """Compose two bounded univariate rational polynomials."""

    outer: RationalPolynomial
    inner: RationalPolynomial
    inner_variable: PolynomialVariable
    outer_variable: PolynomialVariable

    @model_validator(mode="after")
    def require_univariate_bounded_composition(self) -> Self:
        require_map_polynomial(self.outer, label="outer polynomial")
        require_map_polynomial(self.inner, label="inner polynomial")
        if self.outer.variables != (self.outer_variable,):
            raise ValueError("outer polynomial must use exactly outer_variable")
        if self.inner.variables != (self.inner_variable,):
            raise ValueError("inner polynomial must use exactly inner_variable")
        outer_degree = max(
            (term.exponents[0] for term in self.outer.polynomial.terms), default=0
        )
        inner_degree = max(
            (term.exponents[0] for term in self.inner.polynomial.terms), default=0
        )
        if outer_degree * inner_degree > _MAX_COMPOSITION_DEGREE:
            raise ValueError(f"composition exceeds degree {_MAX_COMPOSITION_DEGREE}")
        return self


class CompositionResult(StrictModel):
    """The canonical polynomial obtained by substitution."""

    polynomial: RationalPolynomial


class GenericDegreeComputationBudget(StrictModel):
    """Caller-selected wall-time envelope for one generic-fiber computation."""

    wall_seconds: StrictInt = Field(default=20, ge=1, le=60)


def _total_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )


class GenericDegreeRequest(StrictModel):
    """Compute the generic-fiber degree of one bounded polynomial map over QQ.

    The source is materialized sparse data. The operation envelope has
    at most three source variables, three ordered target components, 96 input
    terms, component total degree at most 8, 64-digit coefficient components,
    64 KiB encoded map, and finite-fiber Bezout bound at most 512. The backend
    runs once under the declared wall/CPU, 1 GiB address-space, 512 KiB
    protocol, and fixed exact certificate limits.
    """

    polynomial_map: RationalPolynomialMap = Field(
        description=(
            "A materialized sparse QQ polynomial map with at most 3 source "
            "variables, 3 ordered components, 48 terms per component, 96 "
            "aggregate terms, total degree at most 8, 64-digit coefficient "
            "components, and 65536 encoded bytes. Maps with at least as many "
            "target as source coordinates also require a finite-fiber Bezout "
            "bound at most 512."
        )
    )
    resource_budget: GenericDegreeComputationBudget = Field(
        default_factory=GenericDegreeComputationBudget
    )

    @model_validator(mode="after")
    def require_generic_fiber_envelope(self) -> Self:
        _require_generic_degree_map_budget(self.polynomial_map)
        return self


def _require_generic_degree_map_budget(polynomial_map: RationalPolynomialMap) -> None:
    source_count = len(polynomial_map.input_variables)
    target_count = len(polynomial_map.output_polynomials)
    if source_count > MAX_GENERIC_DEGREE_SOURCE_VARIABLES:
        raise ValueError(
            "generic-degree source exceeds the 3-variable operation budget"
        )
    if target_count > MAX_GENERIC_DEGREE_TARGET_VARIABLES:
        raise ValueError(
            "generic-degree target exceeds the 3-component operation budget"
        )
    aggregate_terms = sum(
        len(polynomial.polynomial.terms)
        for polynomial in polynomial_map.output_polynomials
    )
    if aggregate_terms > MAX_GENERIC_DEGREE_AGGREGATE_TERMS:
        raise ValueError(
            "generic-degree map exceeds the 96-term aggregate input budget"
        )
    if (
        len(polynomial_map.model_dump_json().encode("utf-8"))
        > MAX_GENERIC_DEGREE_ENCODED_MAP_BYTES
    ):
        raise ValueError("generic-degree map exceeds the 65536-byte input budget")
    degrees: list[int] = []
    for polynomial in polynomial_map.output_polynomials:
        if len(polynomial.polynomial.terms) > MAX_GENERIC_DEGREE_COMPONENT_TERMS:
            raise ValueError(
                "generic-degree component exceeds the 48-term input budget"
            )
        degree = _total_degree(polynomial)
        degrees.append(degree)
        if degree > MAX_GENERIC_DEGREE_TOTAL_DEGREE:
            raise ValueError("generic-degree component exceeds total degree 8")
        for term in polynomial.polynomial.terms:
            if (
                len(term.coefficient.num.lstrip("-"))
                > MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS
                or len(term.coefficient.den) > MAX_GENERIC_DEGREE_COEFFICIENT_DIGITS
            ):
                raise ValueError(
                    "generic-degree coefficient exceeds the 64-digit input budget"
                )
    if target_count >= source_count:
        bezout_bound = 1
        for degree in sorted(degrees)[:source_count]:
            bezout_bound *= max(1, degree)
        if bezout_bound > MAX_GENERIC_DEGREE_BEZOUT_BOUND:
            raise ValueError("generic-degree finite-fiber Bezout bound exceeds 512")


class GenericFiberTerm(StrictModel):
    """One source monomial with coefficient in the generic target field."""

    coefficient: RationalFunction
    source_exponents: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_GENERIC_DEGREE_SOURCE_VARIABLES,
    )

    @model_validator(mode="after")
    def require_nonzero_bounded_term(self) -> Self:
        if not self.coefficient.numerator.terms:
            raise ValueError("zero generic-fiber terms must be omitted")
        if any(
            exponent < 0
            or exponent > MAX_GENERIC_FIBER_CERTIFICATE_SOURCE_EXPONENT
            for exponent in self.source_exponents
        ):
            raise ValueError(
                "generic-fiber source exponent exceeds the certificate "
                "representation limit"
            )
        return self


class GenericFiberPolynomial(StrictModel):
    """A sparse polynomial over ``QQ(t_1,...,t_m)`` in the source variables."""

    terms: tuple[GenericFiberTerm, ...] = Field(
        default=(),
        max_length=MAX_GENERIC_FIBER_POLYNOMIAL_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_term_order(self) -> Self:
        exponents = tuple(term.source_exponents for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise ValueError("generic-fiber exponent tuples must be unique")
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise ValueError(
                "generic-fiber terms must use descending lexicographic order"
            )
        return self


class GenericFiberCertificate(StrictModel):
    """Exact Gröbner and standard-monomial evidence for the generic fiber.

    ``basis_from_source[i][j]`` is the coefficient multiplying source
    generator ``F_i-t_i`` in basis polynomial ``j``. Thus
    ``basis = source_generators * basis_from_source``.
    """

    target_parameters: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_GENERIC_DEGREE_TARGET_VARIABLES,
    )
    source_variable_order: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_GENERIC_DEGREE_SOURCE_VARIABLES,
    )
    monomial_order: Literal["LEX"] = "LEX"
    basis: tuple[GenericFiberPolynomial, ...] = Field(
        min_length=1,
        max_length=MAX_GENERIC_FIBER_BASIS_POLYNOMIALS,
    )
    basis_from_source: tuple[tuple[GenericFiberPolynomial, ...], ...] = Field(
        min_length=1,
        max_length=MAX_GENERIC_DEGREE_TARGET_VARIABLES,
    )
    standard_monomials: tuple[tuple[int, ...], ...] = Field(
        default=(),
        max_length=MAX_GENERIC_FIBER_STANDARD_MONOMIALS,
    )

    @model_validator(mode="before")
    @classmethod
    def require_serialized_certificate_bounds(cls, value: Any) -> Any:
        """Reject oversized nested payloads before constructing coefficient values."""

        if not isinstance(value, Mapping):
            return value
        basis = value.get("basis")
        transformation = value.get("basis_from_source")
        if not isinstance(basis, Sequence) or isinstance(basis, (str, bytes)):
            return value
        if not isinstance(transformation, Sequence) or isinstance(
            transformation, (str, bytes)
        ):
            return value
        if len(basis) > MAX_GENERIC_FIBER_BASIS_POLYNOMIALS:
            raise ValueError("generic-fiber basis exceeds the result bound")
        if len(transformation) > MAX_GENERIC_DEGREE_TARGET_VARIABLES:
            raise ValueError("generic-fiber transformation exceeds the result bound")

        polynomials: list[Any] = list(basis)
        for row in transformation:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                return value
            if len(row) > MAX_GENERIC_FIBER_BASIS_POLYNOMIALS:
                raise ValueError("generic-fiber transformation exceeds the result bound")
            polynomials.extend(row)

        term_groups: list[Sequence[Any]] = []
        certificate_terms = 0
        for polynomial in polynomials:
            terms = (
                polynomial.terms
                if isinstance(polynomial, GenericFiberPolynomial)
                else polynomial.get("terms")
                if isinstance(polynomial, Mapping)
                else None
            )
            if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
                return value
            certificate_terms += len(terms)
            if certificate_terms > MAX_GENERIC_FIBER_CERTIFICATE_TERMS:
                raise ValueError(
                    "generic-fiber certificate exceeds the 4096-term result bound"
                )
            term_groups.append(terms)

        coefficient_terms = 0
        for terms in term_groups:
            for term in terms:
                coefficient = (
                    term.coefficient
                    if isinstance(term, GenericFiberTerm)
                    else term.get("coefficient")
                    if isinstance(term, Mapping)
                    else None
                )
                if isinstance(coefficient, RationalFunction):
                    coefficient_terms += len(coefficient.numerator.terms)
                    coefficient_terms += len(coefficient.denominator.terms)
                elif isinstance(coefficient, Mapping):
                    numerator = coefficient.get("numerator")
                    denominator = coefficient.get("denominator")
                    if not isinstance(numerator, Mapping) or not isinstance(
                        denominator, Mapping
                    ):
                        return value
                    numerator_terms = numerator.get("terms")
                    denominator_terms = denominator.get("terms")
                    if not isinstance(numerator_terms, Sequence) or not isinstance(
                        denominator_terms, Sequence
                    ):
                        return value
                    coefficient_terms += len(numerator_terms) + len(denominator_terms)
                else:
                    return value
                if coefficient_terms > MAX_GENERIC_FIBER_COEFFICIENT_TERMS:
                    raise ValueError(
                        "generic-fiber coefficient support exceeds the result bound"
                    )
        return value

    @model_validator(mode="after")
    def require_bounded_certificate_shape(self) -> Self:
        if len(set(self.target_parameters)) != len(self.target_parameters):
            raise ValueError("generic target parameters must be unique")
        if len(set(self.source_variable_order)) != len(self.source_variable_order):
            raise ValueError("generic-fiber source variables must be unique")
        basis_count = len(self.basis)
        if any(len(row) != basis_count for row in self.basis_from_source):
            raise ValueError(
                "generic-fiber transformation rows must match the basis length"
            )
        polynomials = [
            *self.basis,
            *(polynomial for row in self.basis_from_source for polynomial in row),
        ]
        certificate_terms = sum(len(polynomial.terms) for polynomial in polynomials)
        if certificate_terms > MAX_GENERIC_FIBER_CERTIFICATE_TERMS:
            raise ValueError(
                "generic-fiber certificate exceeds the 4096-term result bound"
            )
        coefficient_terms = sum(
            len(term.coefficient.numerator.terms)
            + len(term.coefficient.denominator.terms)
            for polynomial in polynomials
            for term in polynomial.terms
        )
        if coefficient_terms > MAX_GENERIC_FIBER_COEFFICIENT_TERMS:
            raise ValueError(
                "generic-fiber coefficient support exceeds the result bound"
            )
        for polynomial in polynomials:
            for term in polynomial.terms:
                if len(term.source_exponents) != len(self.source_variable_order):
                    raise ValueError(
                        "generic-fiber monomials must match the source variable order"
                    )
                if term.coefficient.variables != self.target_parameters:
                    raise ValueError(
                        "generic-fiber coefficients must use the target parameter field"
                    )
        if len(set(self.standard_monomials)) != len(self.standard_monomials):
            raise ValueError("standard monomials must be unique")
        if self.standard_monomials != tuple(sorted(self.standard_monomials)):
            raise ValueError(
                "standard monomials must use ascending lexicographic order"
            )
        if any(
            len(exponents) != len(self.source_variable_order)
            or any(
                exponent < 0
                or exponent > MAX_GENERIC_FIBER_STANDARD_MONOMIAL_EXPONENT
                for exponent in exponents
            )
            for exponents in self.standard_monomials
        ):
            raise ValueError(
                "standard monomials must match the bounded source variable order"
            )
        return self


GenericDegreeOutcome = Literal[
    "GENERICALLY_FINITE",
    "NOT_DOMINANT",
    "DOMINANT_NOT_GENERICALLY_FINITE",
    "UNAVAILABLE",
    "TIMEOUT",
    "CANCELLED",
    "BOUND_EXCEEDED",
    "ERROR",
]


class GenericDegreeResult(StrictModel):
    """An exact source-bound generic-fiber conclusion or operational failure."""

    outcome: GenericDegreeOutcome
    source: RationalPolynomialMap
    degree: int | None = Field(default=None, ge=1, le=MAX_GENERIC_DEGREE_BEZOUT_BOUND)
    evidence: GenericFiberCertificate | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_source_bound_outcome(self) -> Self:
        _require_generic_degree_map_budget(self.source)
        mathematical = {
            "GENERICALLY_FINITE",
            "NOT_DOMINANT",
            "DOMINANT_NOT_GENERICALLY_FINITE",
        }
        if self.outcome not in mathematical:
            if (
                self.degree is not None
                or self.evidence is not None
                or not self.detail
            ):
                raise ValueError(
                    "operational generic-degree outcomes require only source and detail"
                )
            return self
        if self.evidence is None or self.detail:
            raise ValueError(
                "mathematical generic-degree outcomes require exact evidence"
            )
        from jacobian.math.polynomials.maps._generic_degree import (
            validate_generic_fiber_certificate,
        )

        expected_outcome, expected_degree = validate_generic_fiber_certificate(
            self.source,
            self.evidence,
        )
        if self.outcome != expected_outcome or self.degree != expected_degree:
            raise ValueError(
                "generic-degree outcome does not match the source-bound quotient evidence"
            )
        return self


__all__ = [
    "CompositionRequest",
    "CompositionResult",
    "EvalRequest",
    "EvalResult",
    "GenericDegreeComputationBudget",
    "GenericDegreeOutcome",
    "GenericDegreeRequest",
    "GenericDegreeResult",
    "GenericFiberCertificate",
    "GenericFiberPolynomial",
    "GenericFiberTerm",
    "JacobianRequest",
    "JacobianResult",
    "VariablePoint",
]
