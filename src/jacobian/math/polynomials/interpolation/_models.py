"""Typed exact contracts for polynomial interpolation operations."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import encode_strict_json
from jacobian.math.polynomials.interpolation._kernel import (
    divided_difference_coefficients,
    evaluate_newton_form,
    ordinary_derivative_value,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
)


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.interpolation_contract", message)


MAX_POINTS = 32
MAX_NEWTON_POINTS = 256
_MAX_RATIONAL_DIGITS = 256
MAX_HERMITE_SYSTEM_CELLS = MAX_POINTS * (MAX_POINTS + 1)
"""Largest admitted Hermite matrix plus right-hand-side allocation."""

MAX_HERMITE_CUBIC_WORK_CELLS = MAX_POINTS**3
"""Conservative scalar-update count for exact fraction-free elimination."""

MAX_HERMITE_SCALED_ENTRY_DIGITS = MAX_POINTS * _MAX_RATIONAL_DIGITS + 64
"""Digit envelope for a row-scaled integer Hermite-system entry."""

MAX_HERMITE_INTERMEDIATE_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
"""Hadamard envelope for fraction-free minors in the maintained backend."""

MAX_HERMITE_RESULT_BYTES = 2 * 1024 * 1024
"""Aggregate canonical-result envelope for polynomial, source, and replay."""


def _require_distinct(nodes: tuple[CanonicalRational, ...]) -> None:
    if len({node.as_integer_ratio() for node in nodes}) != len(nodes):
        raise _validation_error("interpolation nodes must be pairwise distinct")


def _require_bounded(values: tuple[CanonicalRational, ...], label: str) -> None:
    for value in values:
        require_bounded_rational(
            value,
            max_digits=_MAX_RATIONAL_DIGITS,
            label=label,
        )


class OrdinaryDerivativeValue(StrictModel):
    """One explicitly indexed ordinary derivative in a complete node jet."""

    derivative_order: int = Field(
        ge=0,
        lt=MAX_POINTS,
        description=(
            "Ordinary derivative order. Within one jet these orders must be "
            "exactly 0, 1, ..., m-1 in sequence."
        ),
    )
    value: CanonicalRational = Field(
        description=(
            f"Canonical exact rational derivative value whose numerator and "
            f"denominator each have at most {_MAX_RATIONAL_DIGITS} digits."
        )
    )


class OrdinaryDerivativeJet(StrictModel):
    """A nonempty complete prefix of ordinary derivatives at one node."""

    node: CanonicalRational = Field(
        description=(
            f"Canonical exact rational node whose numerator and denominator "
            f"each have at most {_MAX_RATIONAL_DIGITS} digits."
        )
    )
    derivatives: tuple[OrdinaryDerivativeValue, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
        description=(
            "Nonempty complete prefix f(node), f'(node), ..., "
            "f^(m-1)(node), represented by derivative orders 0 through m-1."
        ),
    )

    @model_validator(mode="after")
    def require_complete_prefix_and_bounds(self) -> Self:
        orders = tuple(item.derivative_order for item in self.derivatives)
        if orders != tuple(range(len(self.derivatives))):
            raise _validation_error(
                "ordinary derivative orders must form the complete prefix "
                "0, 1, ..., m-1 in sequence"
            )
        _require_bounded((self.node,), "Hermite interpolation node")
        _require_bounded(
            tuple(item.value for item in self.derivatives),
            "Hermite derivative value",
        )
        return self


def _raw_derivative_count(value: object) -> int:
    if isinstance(value, OrdinaryDerivativeJet):
        return len(value.derivatives)
    if isinstance(value, dict):
        derivatives = value.get("derivatives")
        if isinstance(derivatives, (list, tuple)):
            return len(derivatives)
    return 0


class OrdinaryDerivativeJetTable(StrictModel):
    """One bounded materialized ordinary-derivative jet table over ``QQ``.

    Rows may be supplied in any order. Nodes must be pairwise distinct, every
    row must contain a complete derivative prefix, and the aggregate
    multiplicity ``M`` may not exceed 32.
    """

    variable: PolynomialVariable = Field(
        description=(
            "The single polynomial variable identifier for the resulting QQ polynomial."
        )
    )
    jets: tuple[OrdinaryDerivativeJet, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
        description=(
            "Nonempty derivative jets at pairwise-distinct rational nodes; row "
            "order does not affect the canonical interpolating polynomial. The "
            "sum of all derivative-prefix lengths is at most 32."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_aggregate_bound(cls, data: object) -> object:
        data = canonicalize_json_containers(data)
        if not isinstance(data, dict):
            return data
        jets = data.get("jets")
        if not isinstance(jets, (list, tuple)):
            return data
        if len(jets) > MAX_POINTS:
            raise _validation_error(
                f"ordinary derivative table exceeds the {MAX_POINTS}-node bound"
            )
        total = 0
        normalized_jets: list[object] = []
        for jet in jets:
            total += _raw_derivative_count(jet)
            if total > MAX_POINTS:
                raise _validation_error(
                    "total ordinary derivative multiplicity exceeds the "
                    f"{MAX_POINTS}-constraint bound"
                )
            if isinstance(jet, dict):
                # A before-validator sees JSON arrays as Python lists. Preserve
                # the strict tuple contract after doing the aggregate count.
                normalized_jet = dict(jet)
                derivatives = normalized_jet.get("derivatives")
                if isinstance(derivatives, list):
                    normalized_jet["derivatives"] = tuple(derivatives)
                normalized_jets.append(normalized_jet)
            else:
                normalized_jets.append(jet)
        normalized = dict(data)
        normalized["jets"] = tuple(normalized_jets)
        return normalized

    @model_validator(mode="after")
    def require_distinct_bounded_jets(self) -> Self:
        _require_distinct(tuple(jet.node for jet in self.jets))
        total = _total_multiplicity(self)
        if total > MAX_POINTS:
            raise _validation_error(
                "total ordinary derivative multiplicity exceeds the "
                f"{MAX_POINTS}-constraint bound"
            )
        _require_hermite_preflight(self)
        return self


def _total_multiplicity(table: OrdinaryDerivativeJetTable) -> int:
    return sum(len(jet.derivatives) for jet in table.jets)


def _factor_digits(value: int, *, exponent: int = 1) -> int:
    if exponent == 0 or abs(value) == 1:
        return 0
    return exponent * len(str(abs(value)))


def _scaled_system_row_digit_bounds(
    table: OrdinaryDerivativeJetTable,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Bound row-scaled Hermite matrix and augmented-row integer heights.

    For the derivative-``r`` row at ``a=p/q`` with target ``s/t``, multiply
    the equation by ``q^(M-1-r) t``. Its monomial-``k`` entry becomes

    ``(k)_r p^(k-r) q^(M-1-k) t``

    and its right-hand side becomes ``s q^(M-1-r)``. These formulas bound every
    integer cell before any large power or matrix entry is materialized.
    """

    multiplicity = _total_multiplicity(table)
    matrix_rows: list[int] = []
    augmented_rows: list[int] = []
    for jet in table.jets:
        node_numerator, node_denominator = jet.node.as_integer_ratio()
        for derivative in jet.derivatives:
            order = derivative.derivative_order
            target_numerator, target_denominator = derivative.value.as_integer_ratio()
            entry_bounds: list[int] = []
            for degree in range(order, multiplicity):
                node_exponent = degree - order
                if node_numerator == 0 and node_exponent > 0:
                    entry_bounds.append(1)
                    continue
                falling = prod(range(degree - order + 1, degree + 1))
                bound = (
                    _factor_digits(falling)
                    + _factor_digits(node_numerator, exponent=node_exponent)
                    + _factor_digits(
                        node_denominator,
                        exponent=multiplicity - 1 - degree,
                    )
                    + _factor_digits(target_denominator)
                )
                entry_bounds.append(max(1, bound))
            matrix_bound = max(entry_bounds)
            if target_numerator == 0:
                target_bound = 1
            else:
                target_bound = max(
                    1,
                    _factor_digits(target_numerator)
                    + _factor_digits(
                        node_denominator,
                        exponent=multiplicity - 1 - order,
                    ),
                )
            matrix_rows.append(matrix_bound)
            augmented_rows.append(max(matrix_bound, target_bound))
    return tuple(matrix_rows), tuple(augmented_rows)


def _hadamard_determinant_digits(row_digits: tuple[int, ...]) -> int:
    dimension = len(row_digits)
    euclidean_norm_digits = len(str(dimension ** ((dimension + 1) // 2)))
    return sum(row_digits) + euclidean_norm_digits + 1


def _predicted_hermite_result_bytes(
    table: OrdinaryDerivativeJetTable,
    coefficient_digits: int,
) -> int:
    multiplicity = _total_multiplicity(table)
    source_bytes = len(encode_strict_json(table.model_dump(mode="json")))
    replay_bytes = sum(
        len(encode_strict_json(jet.node.model_dump(mode="json")))
        + 2 * len(encode_strict_json(derivative.value.model_dump(mode="json")))
        + 160
        for jet in table.jets
        for derivative in jet.derivatives
    )
    polynomial_bytes = multiplicity * (2 * coefficient_digits + 192)
    leading_coefficient_bytes = 2 * coefficient_digits + 192
    return (
        source_bytes
        + replay_bytes
        + polynomial_bytes
        + leading_coefficient_bytes
        + 4_096
    )


def _require_hermite_preflight(
    table: OrdinaryDerivativeJetTable,
) -> tuple[int, int]:
    """Validate allocation, exact-work, intermediate, and result envelopes."""

    multiplicity = _total_multiplicity(table)
    system_cells = multiplicity * (multiplicity + 1)
    if system_cells > MAX_HERMITE_SYSTEM_CELLS:
        raise _validation_error(
            "Hermite linear system exceeds the "
            f"{MAX_HERMITE_SYSTEM_CELLS}-cell allocation bound"
        )
    cubic_work_cells = multiplicity**3
    if cubic_work_cells > MAX_HERMITE_CUBIC_WORK_CELLS:
        raise _validation_error(
            "Hermite fraction-free elimination exceeds the "
            f"{MAX_HERMITE_CUBIC_WORK_CELLS}-cell work bound"
        )

    matrix_rows, augmented_rows = _scaled_system_row_digit_bounds(table)
    maximum_entry_digits = max(augmented_rows)
    if maximum_entry_digits > MAX_HERMITE_SCALED_ENTRY_DIGITS:
        raise _validation_error(
            "a row-scaled Hermite system entry exceeds the "
            f"{MAX_HERMITE_SCALED_ENTRY_DIGITS}-digit bound"
        )

    intermediate_digits = _hadamard_determinant_digits(matrix_rows)
    if intermediate_digits > MAX_HERMITE_INTERMEDIATE_DIGITS:
        raise _validation_error(
            "Hermite fraction-free intermediate growth exceeds the "
            f"{MAX_HERMITE_INTERMEDIATE_DIGITS}-digit bound"
        )

    all_zero = all(
        derivative.value.as_fraction() == 0
        for jet in table.jets
        for derivative in jet.derivatives
    )
    coefficient_digits = (
        1
        if all_zero
        else max(
            intermediate_digits,
            _hadamard_determinant_digits(augmented_rows),
        )
    )
    if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise _validation_error(
            "Hermite coefficient growth exceeds the canonical "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
        )
    predicted_result_bytes = _predicted_hermite_result_bytes(table, coefficient_digits)
    if predicted_result_bytes > MAX_HERMITE_RESULT_BYTES:
        raise _validation_error(
            "the complete Hermite result exceeds the "
            f"{MAX_HERMITE_RESULT_BYTES}-byte aggregate result bound"
        )
    return coefficient_digits, predicted_result_bytes


class HermiteInterpolationRequest(StrictModel):
    table: OrdinaryDerivativeJetTable = Field(
        description=(
            "Materialized ordinary-derivative jet table whose total "
            "multiplicity determines the unique degree bound."
        )
    )


class HermiteConstraintReplay(StrictModel):
    """One exact replay row for an ordinary-derivative constraint."""

    node: CanonicalRational
    derivative_order: int = Field(ge=0, lt=MAX_POINTS)
    expected: CanonicalRational
    computed: CanonicalRational


def _polynomial_coefficients(
    polynomial: RationalPolynomial,
    multiplicity: int,
) -> tuple[Fraction, ...]:
    coefficients = [Fraction(0)] * multiplicity
    for term in polynomial.polynomial.terms:
        exponent = term.exponents[0]
        if exponent >= multiplicity:
            raise _validation_error("Hermite polynomial degree must be less than M")
        coefficients[exponent] = term.coefficient.as_fraction()
    return tuple(coefficients)


class HermiteInterpolationResult(StrictModel):
    """Unique degree-``< M`` interpolant with complete source-bound replay."""

    source: OrdinaryDerivativeJetTable = Field(
        description="Retained source table that binds the polynomial and replay."
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "Canonical sparse polynomial in QQ[source.variable] of degree less "
            "than total_multiplicity."
        )
    )
    total_multiplicity: int = Field(
        ge=1,
        le=MAX_POINTS,
        description="Sum of all retained derivative-prefix lengths.",
    )
    degree: int | None = Field(
        default=None,
        ge=0,
        lt=MAX_POINTS,
        description=(
            "Exact polynomial degree, or null for the zero polynomial whose "
            "sparse term family is empty."
        ),
    )
    leading_coefficient: CanonicalRational = Field(
        description=(
            "Exact leading coefficient; canonical zero 0/1 for the zero polynomial."
        )
    )
    replay: tuple[HermiteConstraintReplay, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
        description=(
            "Every retained derivative constraint exactly once, sorted by "
            "rational node and then derivative order."
        ),
    )
    method: Literal["FLINT_FMPQ_HERMITE_VANDERMONDE_FFLU"] = (
        "FLINT_FMPQ_HERMITE_VANDERMONDE_FFLU"
    )

    @model_validator(mode="after")
    def require_unique_source_bound_interpolant(self) -> Self:
        multiplicity = _total_multiplicity(self.source)
        if self.total_multiplicity != multiplicity:
            raise _validation_error(
                "total_multiplicity must equal the source jet length"
            )
        if self.polynomial.variables != (self.source.variable,):
            raise _validation_error(
                "Hermite polynomial must use exactly the source table variable"
            )
        coefficient_bound, _ = _require_hermite_preflight(self.source)
        for term in self.polynomial.polynomial.terms:
            require_bounded_rational(
                term.coefficient,
                max_digits=coefficient_bound,
                label="Hermite polynomial coefficient",
            )
        coefficients = _polynomial_coefficients(self.polynomial, multiplicity)
        nonzero_degrees = tuple(
            degree for degree, coefficient in enumerate(coefficients) if coefficient
        )
        expected_degree = max(nonzero_degrees) if nonzero_degrees else None
        if self.degree != expected_degree:
            raise _validation_error("degree must equal the exact polynomial degree")
        expected_leading = (
            Fraction(0) if expected_degree is None else coefficients[expected_degree]
        )
        if self.leading_coefficient.as_fraction() != expected_leading:
            raise _validation_error(
                "leading_coefficient must match the canonical polynomial"
            )

        expected_rows = tuple(
            (
                jet.node.as_fraction(),
                derivative.derivative_order,
                derivative.value.as_fraction(),
            )
            for jet in sorted(
                self.source.jets, key=lambda item: item.node.as_fraction()
            )
            for derivative in jet.derivatives
        )
        actual_rows = tuple(
            (
                item.node.as_fraction(),
                item.derivative_order,
                item.expected.as_fraction(),
            )
            for item in self.replay
        )
        if actual_rows != expected_rows:
            raise _validation_error(
                "replay must cover every source derivative constraint in "
                "canonical node/order order"
            )
        for item in self.replay:
            computed = ordinary_derivative_value(
                coefficients,
                item.node.as_fraction(),
                item.derivative_order,
            )
            if item.computed.as_fraction() != computed:
                raise _validation_error(
                    "replay computed value does not match the returned polynomial"
                )
            if item.computed.as_fraction() != item.expected.as_fraction():
                raise _validation_error(
                    "returned polynomial does not satisfy a source derivative "
                    "constraint"
                )
        if (
            len(encode_strict_json(self.model_dump(mode="json")))
            > MAX_HERMITE_RESULT_BYTES
        ):
            raise _validation_error(
                "complete Hermite result exceeds the aggregate result bound"
            )
        return self


class InterpolationSamples(StrictModel):
    """One bounded graph of a rational-valued function on distinct nodes."""

    nodes: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_NEWTON_POINTS,
    )
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_NEWTON_POINTS,
    )

    @model_validator(mode="after")
    def require_well_formed_samples(self) -> Self:
        if len(self.nodes) != len(self.values):
            raise _validation_error("nodes and values must have the same length")
        _require_distinct(self.nodes)
        _require_bounded(self.nodes, "interpolation node")
        _require_bounded(self.values, "interpolation value")
        for coefficient in divided_difference_coefficients(self.nodes, self.values):
            require_bounded_rational(
                CanonicalRational.from_fraction(coefficient),
                max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="derived Newton coefficient",
            )
        return self


class DividedDifferencesRequest(StrictModel):
    samples: InterpolationSamples


class NewtonFormRequest(StrictModel):
    samples: InterpolationSamples


class NewtonForm(StrictModel):
    """A directly evaluable Newton-basis polynomial over ``QQ``."""

    nodes: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_NEWTON_POINTS,
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_NEWTON_POINTS,
    )

    @model_validator(mode="after")
    def require_basis_shape(self) -> Self:
        if len(self.nodes) != len(self.coefficients):
            raise _validation_error(
                "Newton nodes and coefficients must have the same length"
            )
        _require_distinct(self.nodes)
        _require_bounded(self.nodes, "Newton node")
        return self


class NewtonEvaluateRequest(StrictModel):
    newton_form: NewtonForm
    evaluation_point: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_point(self) -> Self:
        require_bounded_rational(
            self.evaluation_point,
            max_digits=_MAX_RATIONAL_DIGITS,
            label="evaluation point",
        )
        require_bounded_rational(
            CanonicalRational.from_fraction(
                evaluate_newton_form(
                    self.newton_form.nodes,
                    self.newton_form.coefficients,
                    self.evaluation_point,
                )
            ),
            max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="derived Newton evaluation",
        )
        return self


class DividedDifferencesResult(StrictModel):
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_NEWTON_POINTS,
    )
    method: str = "NEWTON_DIVIDED_DIFFERENCES"


class NewtonEvaluateResult(StrictModel):
    result: CanonicalRational
    method: str = "NEWTON_HORNER"


__all__ = [
    "MAX_HERMITE_CUBIC_WORK_CELLS",
    "MAX_HERMITE_INTERMEDIATE_DIGITS",
    "MAX_HERMITE_RESULT_BYTES",
    "MAX_HERMITE_SCALED_ENTRY_DIGITS",
    "MAX_HERMITE_SYSTEM_CELLS",
    "DividedDifferencesRequest",
    "DividedDifferencesResult",
    "HermiteConstraintReplay",
    "HermiteInterpolationRequest",
    "HermiteInterpolationResult",
    "InterpolationSamples",
    "NewtonEvaluateRequest",
    "NewtonEvaluateResult",
    "NewtonForm",
    "NewtonFormRequest",
    "OrdinaryDerivativeJet",
    "OrdinaryDerivativeJetTable",
    "OrdinaryDerivativeValue",
]
