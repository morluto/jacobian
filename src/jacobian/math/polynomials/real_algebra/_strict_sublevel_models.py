"""Typed source-bound contracts for exact strict polynomial sublevel measure."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd, lcm
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
)

# SymPy clears QQ denominators before its Vincent-Akritas-Strzebonski exact
# real-root isolation.  The controlling height is therefore the primitive
# integer height of f-t and f+t, not one input coefficient in isolation.  These
# degree and height caps form a conservative fixed backend envelope while the
# backend exposes no exact step counter.  Exact root comparisons are likewise
# bounded by two lists of at most degree roots in these same number fields.
MAX_STRICT_SUBLEVEL_DEGREE = 16
MAX_STRICT_SUBLEVEL_TERMS = MAX_STRICT_SUBLEVEL_DEGREE + 1
MAX_STRICT_SUBLEVEL_INPUT_DIGITS = 64
MAX_STRICT_SUBLEVEL_BOUNDARY_HEIGHT_DIGITS = 256
MAX_STRICT_SUBLEVEL_BOUNDARIES = 2 * MAX_STRICT_SUBLEVEL_DEGREE
MAX_STRICT_SUBLEVEL_COMPONENTS = MAX_STRICT_SUBLEVEL_BOUNDARIES + 1

# Joint isolation-work admission for the two level polynomials.  VAS isolation
# is strongly degree- and height-sensitive; ``degree**5 * sum(height_digits)``
# is a deliberately conservative measured proxy over the fixed exact backend.
# A degree-10 polynomial with twenty nearby level boundaries costs 1.6M units
# and about 2.5 seconds including canonical result construction; a close-root
# degree-8, 41-digit case costs 2.69M units and about 9 seconds, so it is
# rejected. The
# envelope still admits degree 16 at unit height and progressively taller
# inputs at lower degree.
MAX_STRICT_SUBLEVEL_ISOLATION_WORK = 2_100_000

# By the rational-root theorem, every rational level root has numerator and
# denominator bounded by the primitive level-polynomial height.  A sum of all
# 2d possible rational boundaries and both rational scope endpoints therefore
# fits this conservative digit budget.  Irrational roots stay as source-bound
# references and do not expand the rational part or serialized result.
MAX_STRICT_SUBLEVEL_MEASURE_DIGITS = (
    MAX_STRICT_SUBLEVEL_BOUNDARIES * MAX_STRICT_SUBLEVEL_BOUNDARY_HEIGHT_DIGITS
    + 2 * MAX_STRICT_SUBLEVEL_INPUT_DIGITS
    + 2
)

LevelEquation = Literal["F_MINUS_THRESHOLD", "F_PLUS_THRESHOLD"]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.strict_sublevel_{reason}", message)


def _bound_raw_rational(
    value: object,
    *,
    max_digits: int,
    label: str,
) -> None:
    """Reject oversized decimal components before canonical-rational parsing."""

    if isinstance(value, CanonicalRational):
        components: tuple[object, object] = (value.num, value.den)
    elif isinstance(value, Mapping):
        components = (value.get("num"), value.get("den"))
    else:
        return
    for component in components:
        if (
            isinstance(component, str)
            and len(component) - component.startswith("-") > max_digits
        ):
            raise _validation_error(
                "rational_height", f"{label} exceeds the {max_digits}-digit bound"
            )


def _bound_raw_terms(terms: tuple[object, ...] | list[object]) -> None:
    """Bound univariate raw terms before nested polynomial parsing."""

    for term in terms:
        coefficient: object
        if isinstance(term, RationalPolynomialTerm):
            coefficient = term.coefficient
        elif isinstance(term, Mapping):
            coefficient = term.get("coefficient")
            exponents = term.get("exponents")
            if isinstance(exponents, (list, tuple)) and len(exponents) != 1:
                raise _validation_error(
                    "term_shape",
                    "strict sublevel measure requires exactly one exponent per term",
                )
        else:
            continue
        _bound_raw_rational(
            coefficient,
            max_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
            label="strict sublevel polynomial coefficient",
        )


def _prepare_raw_polynomial(value: object) -> object:
    """Bound and normalize one JSON polynomial before nested construction."""

    if isinstance(value, RationalPolynomial):
        variables: object = value.variables
        terms: object = value.polynomial.terms
    elif isinstance(value, Mapping):
        variables = value.get("variables")
        sparse = value.get("polynomial")
        terms = sparse.get("terms") if isinstance(sparse, Mapping) else None
    else:
        return value

    if isinstance(variables, (list, tuple)) and len(variables) > 1:
        raise _validation_error(
            "variable_count", "strict sublevel measure requires one polynomial variable"
        )
    if not isinstance(terms, (list, tuple)):
        return value
    if len(terms) > MAX_STRICT_SUBLEVEL_TERMS:
        raise _validation_error(
            "term_budget",
            "strict sublevel polynomial exceeds the "
            f"{MAX_STRICT_SUBLEVEL_TERMS}-term operation budget",
        )
    _bound_raw_terms(terms)

    if not isinstance(value, Mapping):
        return value
    prepared = dict(value)
    if isinstance(variables, list):
        prepared["variables"] = tuple(variables)
    if isinstance(sparse, Mapping):
        prepared_sparse = dict(sparse)
        prepared_terms: list[object] = []
        for term in terms:
            if isinstance(term, Mapping):
                prepared_term = dict(term)
                exponents = prepared_term.get("exponents")
                if isinstance(exponents, list):
                    prepared_term["exponents"] = tuple(exponents)
                prepared_terms.append(prepared_term)
            else:
                prepared_terms.append(term)
        prepared_sparse["terms"] = tuple(prepared_terms)
        prepared["polynomial"] = prepared_sparse
    return prepared


def _polynomial_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (term.exponents[0] for term in polynomial.polynomial.terms),
        default=0,
    )


def _level_polynomial_height_digits(
    polynomial: RationalPolynomial,
    threshold: CanonicalRational,
    *,
    subtract_threshold: bool,
) -> int:
    """Return primitive integer height digits of ``f-t`` or ``f+t``.

    This bounded Fraction-only preflight mirrors denominator clearing without
    asking the root backend to expand anything.  The polynomial is known to be
    univariate before this helper is called.
    """

    coefficients = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }
    signed_threshold = threshold.as_fraction()
    if subtract_threshold:
        signed_threshold = -signed_threshold
    coefficients[0] = coefficients.get(0, 0) + signed_threshold
    nonzero = tuple(value for value in coefficients.values() if value)
    if not nonzero:
        return 1

    common_denominator = 1
    for coefficient in nonzero:
        common_denominator = lcm(common_denominator, coefficient.denominator)
    integer_coefficients = tuple(
        coefficient.numerator * (common_denominator // coefficient.denominator)
        for coefficient in nonzero
    )
    content = 0
    for integer_coefficient in integer_coefficients:
        content = gcd(content, abs(integer_coefficient))
    return max(
        len(str(abs(integer_coefficient // content)))
        for integer_coefficient in integer_coefficients
    )


class StrictSublevelMeasureRequest(StrictModel):
    """Exact strict ``|f(x)| < threshold`` measure on a rational scope."""

    polynomial: RationalPolynomial
    threshold: CanonicalRational
    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="before")
    @classmethod
    def bound_raw_request(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        prepared["polynomial"] = _prepare_raw_polynomial(prepared.get("polynomial"))
        for field, label in (
            ("threshold", "strict sublevel threshold"),
            ("lower", "strict sublevel lower scope endpoint"),
            ("upper", "strict sublevel upper scope endpoint"),
        ):
            _bound_raw_rational(
                prepared.get(field),
                max_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
                label=label,
            )
        return prepared


class LevelRootReference(StrictModel):
    """One distinct real root of a retained level polynomial.

    ``root_index`` is zero-based in the increasing list of distinct real roots
    of the named equation.  Together with the result's source polynomial and
    threshold, it is an exact real-algebraic number definition.
    """

    equation: LevelEquation
    root_index: int = Field(
        ge=0,
        lt=MAX_STRICT_SUBLEVEL_DEGREE,
        strict=True,
    )
    multiplicity: int = Field(
        ge=1,
        le=MAX_STRICT_SUBLEVEL_DEGREE,
        strict=True,
    )


class ScopeEndpoint(StrictModel):
    kind: Literal["SCOPE_ENDPOINT"] = "SCOPE_ENDPOINT"
    value: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_scope_value(self) -> Self:
        require_bounded_rational(
            self.value,
            max_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
            label="strict sublevel scope endpoint",
        )
        return self


class LevelRootEndpoint(StrictModel):
    kind: Literal["LEVEL_ROOT"] = "LEVEL_ROOT"
    root: LevelRootReference


StrictSublevelEndpoint = Annotated[
    ScopeEndpoint | LevelRootEndpoint,
    Field(discriminator="kind"),
]


class StrictSublevelComponent(StrictModel):
    """One connected component with explicit strict/scope endpoint membership."""

    left: StrictSublevelEndpoint
    right: StrictSublevelEndpoint
    left_included: StrictBool
    right_included: StrictBool


class AlgebraicMeasureRootTerm(StrictModel):
    """An integer multiple of one retained-source irrational level root."""

    root: LevelRootReference
    coefficient: Literal[-1, 1]

    @field_validator("coefficient", mode="before")
    @classmethod
    def require_strict_incidence_coefficient(cls, value: object) -> object:
        if type(value) is not int:
            raise _validation_error(
                "root_coefficient",
                "measure root coefficient must be the integer -1 or 1",
            )
        return value


class SourceBoundAlgebraicMeasure(StrictModel):
    """The exact value ``rational_part + sum(coefficient * root)``.

    Root terms are consolidated and sorted in the retained level-root
    coordinates.  This is the canonical endpoint-incidence ledger for the
    decomposition, rather than a potentially exponential primitive-element
    expansion.
    """

    rational_part: CanonicalRational
    root_terms: tuple[AlgebraicMeasureRootTerm, ...] = Field(
        default=(),
        max_length=MAX_STRICT_SUBLEVEL_BOUNDARIES,
    )

    @model_validator(mode="after")
    def require_canonical_root_ledger(self) -> Self:
        require_bounded_rational(
            self.rational_part,
            max_digits=MAX_STRICT_SUBLEVEL_MEASURE_DIGITS,
            label="strict sublevel rational measure part",
        )
        keys = tuple(
            (term.root.equation, term.root.root_index) for term in self.root_terms
        )
        if len(set(keys)) != len(keys):
            raise _validation_error(
                "duplicate_roots", "measure root terms must identify distinct roots"
            )
        if keys != tuple(sorted(keys)):
            raise _validation_error(
                "root_order",
                "measure root terms must use canonical equation/index order",
            )
        return self


class StrictSublevelMeasureResult(StrictModel):
    """Complete source-bound strict sublevel decomposition and exact measure."""

    source_polynomial: RationalPolynomial
    threshold: CanonicalRational
    lower: CanonicalRational
    upper: CanonicalRational
    components: tuple[StrictSublevelComponent, ...] = Field(
        default=(),
        max_length=MAX_STRICT_SUBLEVEL_COMPONENTS,
    )
    measure: SourceBoundAlgebraicMeasure

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        prepared["source_polynomial"] = _prepare_raw_polynomial(
            prepared.get("source_polynomial")
        )
        for field, label in (
            ("threshold", "strict sublevel threshold"),
            ("lower", "strict sublevel lower scope endpoint"),
            ("upper", "strict sublevel upper scope endpoint"),
        ):
            _bound_raw_rational(
                prepared.get(field),
                max_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
                label=label,
            )

        raw_components = prepared.get("components")
        if isinstance(raw_components, (list, tuple)):
            if len(raw_components) > MAX_STRICT_SUBLEVEL_COMPONENTS:
                raise _validation_error(
                    "component_bound",
                    "strict sublevel result exceeds the "
                    f"{MAX_STRICT_SUBLEVEL_COMPONENTS}-component bound",
                )
            for component in raw_components:
                if not isinstance(component, Mapping):
                    continue
                for side in ("left", "right"):
                    endpoint = component.get(side)
                    if isinstance(endpoint, Mapping) and endpoint.get("kind") == (
                        "SCOPE_ENDPOINT"
                    ):
                        _bound_raw_rational(
                            endpoint.get("value"),
                            max_digits=MAX_STRICT_SUBLEVEL_INPUT_DIGITS,
                            label="strict sublevel scope endpoint",
                        )
            if isinstance(raw_components, list):
                prepared["components"] = tuple(raw_components)

        raw_measure = prepared.get("measure")
        if isinstance(raw_measure, SourceBoundAlgebraicMeasure):
            rational_part: object = raw_measure.rational_part
            root_terms: object = raw_measure.root_terms
        elif isinstance(raw_measure, Mapping):
            measure = dict(raw_measure)
            rational_part = measure.get("rational_part")
            root_terms = measure.get("root_terms")
            if isinstance(root_terms, list):
                measure["root_terms"] = tuple(root_terms)
                root_terms = measure["root_terms"]
            prepared["measure"] = measure
        else:
            return prepared
        _bound_raw_rational(
            rational_part,
            max_digits=MAX_STRICT_SUBLEVEL_MEASURE_DIGITS,
            label="strict sublevel rational measure part",
        )
        if isinstance(root_terms, (list, tuple)) and (
            len(root_terms) > MAX_STRICT_SUBLEVEL_BOUNDARIES
        ):
            raise _validation_error(
                "root_term_bound",
                "strict sublevel measure exceeds the "
                f"{MAX_STRICT_SUBLEVEL_BOUNDARIES}-root-term bound",
            )
        return prepared

    @classmethod
    def _from_kernel(
        cls,
        source_polynomial: RationalPolynomial,
        threshold: CanonicalRational,
        lower: CanonicalRational,
        upper: CanonicalRational,
        *,
        components: tuple[StrictSublevelComponent, ...],
        measure: SourceBoundAlgebraicMeasure,
    ) -> Self:
        """Build a result after the admitted exact kernel established it.

        Public parsing checks the bounded structural representation only.
        Re-running root isolation to authenticate an independently supplied
        claim belongs to the owner-private verifier, not result validation.
        """

        return cls.model_construct(
            source_polynomial=source_polynomial,
            threshold=threshold,
            lower=lower,
            upper=upper,
            components=components,
            measure=measure,
        )


__all__ = [
    "AlgebraicMeasureRootTerm",
    "LevelRootEndpoint",
    "LevelRootReference",
    "ScopeEndpoint",
    "SourceBoundAlgebraicMeasure",
    "StrictSublevelComponent",
    "StrictSublevelEndpoint",
    "StrictSublevelMeasureRequest",
    "StrictSublevelMeasureResult",
]
