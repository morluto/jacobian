"""Typed wire contracts for polynomial map operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.polynomials.maps.values import (
    MAX_MAP_INPUTS,
    MAX_MAP_OUTPUTS,
    RationalPolynomialMap,
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
MAX_GENERIC_FIBER_PARAMETER_TERMS = 256
MAX_GENERIC_FIBER_STANDARD_MONOMIALS = 512
MAX_GENERIC_FIBER_CERTIFICATE_SOURCE_EXPONENT = MAX_POLYNOMIAL_EXPONENT
MAX_GENERIC_FIBER_STANDARD_MONOMIAL_EXPONENT = MAX_GENERIC_FIBER_STANDARD_MONOMIALS - 1
MAX_GENERIC_FIBER_REPLAY_PRODUCTS = 262_144
MAX_GENERIC_FIBER_REPLAY_SOURCE_PRODUCTS = 262_144
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_OPERATIONS = 1_048_576
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_PRODUCTS = 1_048_576
MAX_GENERIC_FIBER_REPLAY_REDUCTION_STEPS = 65_536
MAX_GENERIC_FIBER_REPLAY_SOURCE_TERMS = 8_192
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_TERMS = 4_096
MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_BITS = 16_384
MAX_GENERIC_FIBER_REPLAY_SOURCE_EXPONENT = 2 * MAX_POLYNOMIAL_EXPONENT


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.map_invariant", message)


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
            raise _validation_error(
                "point variables and values must have the same length"
            )
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error("point variables must be unique")
        return self


class EvalRequest(StrictModel):
    """Evaluate one canonical rational polynomial at a complete rational point."""

    polynomial: RationalPolynomial
    point: VariablePoint


class EvalResult(StrictModel):
    """The exact rational value at the requested point."""

    value: CanonicalRational


class JacobianResult(StrictModel):
    """The row-major Jacobian matrix over the source polynomial ring."""

    n_inputs: int = Field(ge=1, le=MAX_MAP_INPUTS)
    n_outputs: int = Field(ge=1, le=MAX_MAP_OUTPUTS)
    entries: tuple[RationalPolynomial, ...] = Field(
        max_length=MAX_MAP_INPUTS * MAX_MAP_OUTPUTS
    )

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if len(self.entries) != self.n_inputs * self.n_outputs:
            raise _validation_error(
                "Jacobian entry count must match its matrix dimensions"
            )
        if self.entries:
            variables = self.entries[0].variables
            if any(entry.variables != variables for entry in self.entries):
                raise _validation_error("Jacobian entries must use one ordered ring")
        return self


class CompositionRequest(StrictModel):
    """Compose two bounded univariate rational polynomials."""

    outer: RationalPolynomial
    inner: RationalPolynomial
    inner_variable: PolynomialVariable
    outer_variable: PolynomialVariable


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
            raise _validation_error("zero generic-fiber terms must be omitted")
        if any(
            exponent < 0 or exponent > MAX_GENERIC_FIBER_CERTIFICATE_SOURCE_EXPONENT
            for exponent in self.source_exponents
        ):
            raise _validation_error(
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
            raise _validation_error("generic-fiber exponent tuples must be unique")
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise _validation_error(
                "generic-fiber terms must use descending lexicographic order"
            )
        return self


def _serialized_certificate_polynomials(value: Any) -> list[Any] | None:
    """Collect one serialized certificate's polynomial payloads."""

    if not isinstance(value, Mapping):
        return None
    basis = value.get("basis")
    transformation = value.get("basis_from_source")
    if not isinstance(basis, Sequence) or isinstance(basis, (str, bytes)):
        return None
    if not isinstance(transformation, Sequence) or isinstance(
        transformation,
        (str, bytes),
    ):
        return None
    if len(basis) > MAX_GENERIC_FIBER_BASIS_POLYNOMIALS:
        raise _validation_error("generic-fiber basis exceeds the result bound")
    if len(transformation) > MAX_GENERIC_DEGREE_TARGET_VARIABLES:
        raise _validation_error("generic-fiber transformation exceeds the result bound")
    polynomials: list[Any] = list(basis)
    for row in transformation:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return None
        if len(row) > MAX_GENERIC_FIBER_BASIS_POLYNOMIALS:
            raise _validation_error(
                "generic-fiber transformation exceeds the result bound"
            )
        polynomials.extend(row)
    return polynomials


def _serialized_polynomial_terms(polynomial: Any) -> Sequence[Any] | None:
    if isinstance(polynomial, GenericFiberPolynomial):
        return polynomial.terms
    if isinstance(polynomial, Mapping):
        terms = polynomial.get("terms")
        if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes)):
            return terms
    return None


def _serialized_term_support(term: Any) -> int | None:
    """Count one serialized term's coefficient support, or ``None`` if malformed."""

    if isinstance(term, GenericFiberTerm):
        return len(term.coefficient.numerator.terms) + len(
            term.coefficient.denominator.terms
        )
    if not isinstance(term, Mapping):
        return None
    coefficient = term.get("coefficient")
    if not isinstance(coefficient, Mapping):
        return None
    numerator = coefficient.get("numerator")
    denominator = coefficient.get("denominator")
    if not isinstance(numerator, Mapping) or not isinstance(denominator, Mapping):
        return None
    numerator_terms = numerator.get("terms")
    denominator_terms = denominator.get("terms")
    if (
        not isinstance(numerator_terms, Sequence)
        or isinstance(numerator_terms, (str, bytes))
        or not isinstance(denominator_terms, Sequence)
        or isinstance(denominator_terms, (str, bytes))
    ):
        return None
    return len(numerator_terms) + len(denominator_terms)


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

        value = canonicalize_json_containers(value)

        polynomials = _serialized_certificate_polynomials(value)
        if polynomials is None:
            return value

        term_groups: list[Sequence[Any]] = []
        certificate_terms = 0
        for polynomial in polynomials:
            terms = _serialized_polynomial_terms(polynomial)
            if terms is None:
                return value
            certificate_terms += len(terms)
            if certificate_terms > MAX_GENERIC_FIBER_CERTIFICATE_TERMS:
                raise _validation_error(
                    "generic-fiber certificate exceeds the 4096-term result bound"
                )
            term_groups.append(terms)

        coefficient_terms = 0
        for terms in term_groups:
            for term in terms:
                support = _serialized_term_support(term)
                if support is None:
                    return value
                coefficient_terms += support
                if coefficient_terms > MAX_GENERIC_FIBER_COEFFICIENT_TERMS:
                    raise _validation_error(
                        "generic-fiber coefficient support exceeds the result bound"
                    )
        return value

    @model_validator(mode="after")
    def require_bounded_certificate_shape(self) -> Self:
        if len(set(self.target_parameters)) != len(self.target_parameters):
            raise _validation_error("generic target parameters must be unique")
        if len(set(self.source_variable_order)) != len(self.source_variable_order):
            raise _validation_error("generic-fiber source variables must be unique")
        basis_count = len(self.basis)
        if any(len(row) != basis_count for row in self.basis_from_source):
            raise _validation_error(
                "generic-fiber transformation rows must match the basis length"
            )
        polynomials = [
            *self.basis,
            *(polynomial for row in self.basis_from_source for polynomial in row),
        ]
        certificate_terms = sum(len(polynomial.terms) for polynomial in polynomials)
        if certificate_terms > MAX_GENERIC_FIBER_CERTIFICATE_TERMS:
            raise _validation_error(
                "generic-fiber certificate exceeds the 4096-term result bound"
            )
        coefficient_terms = sum(
            len(term.coefficient.numerator.terms)
            + len(term.coefficient.denominator.terms)
            for polynomial in polynomials
            for term in polynomial.terms
        )
        if coefficient_terms > MAX_GENERIC_FIBER_COEFFICIENT_TERMS:
            raise _validation_error(
                "generic-fiber coefficient support exceeds the result bound"
            )
        for polynomial in polynomials:
            for term in polynomial.terms:
                if len(term.source_exponents) != len(self.source_variable_order):
                    raise _validation_error(
                        "generic-fiber monomials must match the source variable order"
                    )
                if term.coefficient.variables != self.target_parameters:
                    raise _validation_error(
                        "generic-fiber coefficients must use the target parameter field"
                    )
        if len(set(self.standard_monomials)) != len(self.standard_monomials):
            raise _validation_error("standard monomials must be unique")
        if self.standard_monomials != tuple(sorted(self.standard_monomials)):
            raise _validation_error(
                "standard monomials must use ascending lexicographic order"
            )
        if any(
            len(exponents) != len(self.source_variable_order)
            or any(
                exponent < 0 or exponent > MAX_GENERIC_FIBER_STANDARD_MONOMIAL_EXPONENT
                for exponent in exponents
            )
            for exponents in self.standard_monomials
        ):
            raise _validation_error(
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


def _is_unit_generic_fiber_basis(certificate: GenericFiberCertificate) -> bool:
    """Report whether the certificate basis presents the constant-one ideal."""

    if len(certificate.basis) != 1 or len(certificate.basis[0].terms) != 1:
        return False
    term = certificate.basis[0].terms[0]
    if term.source_exponents != (0,) * len(certificate.source_variable_order):
        return False
    constant = (0,) * len(certificate.target_parameters)
    numerator = term.coefficient.numerator.terms
    denominator = term.coefficient.denominator.terms
    return (
        len(numerator) == 1
        and len(denominator) == 1
        and numerator[0].exponents == constant
        and denominator[0].exponents == constant
        and numerator[0].coefficient.as_fraction() == 1
        and denominator[0].coefficient.as_fraction() == 1
    )


class GenericDegreeResult(StrictModel):
    """An exact source-bound generic-fiber conclusion or operational failure.

    The declared outcome must agree with the retained evidence shape.  The
    producer establishes the mathematical conclusion once; an independently
    supplied claim may be checked with the owner-private verifier without
    re-entering a process-bound replay during deserialization.
    """

    outcome: GenericDegreeOutcome
    source: RationalPolynomialMap
    degree: int | None = Field(default=None, ge=1, le=MAX_GENERIC_DEGREE_BEZOUT_BOUND)
    evidence: GenericFiberCertificate | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_source_bound_outcome(self) -> Self:
        mathematical = {
            "GENERICALLY_FINITE",
            "NOT_DOMINANT",
            "DOMINANT_NOT_GENERICALLY_FINITE",
        }
        if self.outcome not in mathematical:
            if self.degree is not None or self.evidence is not None or not self.detail:
                raise _validation_error(
                    "operational generic-degree outcomes require only source and detail"
                )
            return self
        if self.evidence is None or self.detail:
            raise _validation_error(
                "mathematical generic-degree outcomes require exact evidence"
            )
        unit_basis = _is_unit_generic_fiber_basis(self.evidence)
        if self.outcome == "GENERICALLY_FINITE":
            if (
                unit_basis
                or not self.evidence.standard_monomials
                or self.degree is None
                or self.degree != len(self.evidence.standard_monomials)
            ):
                raise _validation_error(
                    "claimed degree does not match the standard-monomial evidence"
                )
        else:
            if self.degree is not None:
                raise _validation_error(
                    "non-finite generic-degree outcomes carry no degree"
                )
            if self.outcome == "NOT_DOMINANT":
                if not unit_basis or self.evidence.standard_monomials:
                    raise _validation_error(
                        "not-dominant outcomes carry the unit generic-fiber basis"
                    )
            elif unit_basis or self.evidence.standard_monomials:
                raise _validation_error(
                    "positive-dimensional outcomes carry a non-unit generic-fiber basis"
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        outcome: GenericDegreeOutcome,
        source: RationalPolynomialMap,
        degree: int | None,
        evidence: GenericFiberCertificate | None,
        detail: str | None,
    ) -> Self:
        """Construct a result after the owner kernel established the claim."""

        return cls.model_construct(
            outcome=outcome,
            source=source,
            degree=degree,
            evidence=evidence,
            detail=detail,
        )


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
    "JacobianResult",
    "VariablePoint",
]
