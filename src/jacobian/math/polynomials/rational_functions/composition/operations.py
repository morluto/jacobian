"""Exact composition of bounded rational coordinate maps."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, NoReturn

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    sparse_rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundWorkCategory,
    FractionBound,
    PolynomialBound,
    RationalFunctionBoundLimits,
    _add_polynomials,
    _fraction_bound,
    _multiply_polynomials,
    _one_polynomial,
    _polynomial_backend_conversion_work_units,
    _recognition_work_units,
    _remove_guaranteed_common_monomial,
    _total_degree_term_bound,
    _validate_canonical_result_bound,
    _zero_polynomial,
)
from jacobian.math.polynomials.rational_functions.composition._models import (
    RationalFunctionMapComposition,
    guard_key,
)
from jacobian.math.polynomials.rational_functions.gradient._kernel import (
    _normalize_fraction,
)
from jacobian.math.polynomials.rational_functions.gradient.operations import (
    _recognize_source,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_COMPOSITION_WORK = 25_000_000
MAX_COMPOSITION_TERMS = 65_536
MAX_COMPOSITION_BITS = 8_388_608
MAX_COMPOSITION_RAW_EXPONENT = 32_768


def _reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("outer", "inner"),
        code=f"rational_function_map.compose.{reason}",
        message=message,
    )


def _reject_undefined_outer_denominator() -> NoReturn:
    raise OperationDomainValidationError(
        location=("outer", "components"),
        code="rational_function_map.compose.undefined_outer_denominator",
        message="an outer denominator vanishes identically after substitution",
    )


def _component_identity(component: RationalFunction) -> bytes:
    return encode_strict_json(component.model_dump(mode="json"))


def _equal_inner_substitution_vanishes(
    polynomial: SparseRationalPolynomial,
    inner_components: tuple[RationalFunction, ...],
    ledger: _Ledger,
) -> bool:
    """Return whether scalar-equivalent inner coordinates cancel the polynomial."""

    if not polynomial.terms or len(inner_components) != len(
        polynomial.terms[0].exponents
    ):
        return False
    identities: list[tuple[bytes, object]] = []
    factors: list[Fraction] = []
    for component in inner_components:
        numerator = component.numerator
        denominator = encode_strict_json(component.denominator.model_dump(mode="json"))
        key: tuple[bytes, object]
        if not numerator.terms:
            key = (denominator, ())
            factor = Fraction(1)
        else:
            pivot = numerator.terms[0].coefficient.as_fraction()
            pivot_bits = _bits_for_fraction(pivot)
            source_bits = sum(
                _bits_for_fraction(term.coefficient.as_fraction())
                for term in numerator.terms
            )
            estimated = source_bits + pivot_bits
            estimated += sum(
                _bits_for_fraction(term.coefficient.as_fraction()) + pivot_bits + 1
                for term in numerator.terms
            )
            if ledger.work + estimated > MAX_COMPOSITION_WORK:
                identities.append((denominator, _component_identity(component)))
                factors.append(Fraction(1))
                continue
            ledger.charge("multiplication", estimated)
            shape = tuple(
                (
                    term.exponents,
                    term.coefficient.as_fraction() / pivot,
                )
                for term in numerator.terms
            )
            key = (denominator, shape)
            factor = pivot
        identities.append(key)
        factors.append(factor)
        ledger.charge("recognition", max(1, len(numerator.terms)))
    representative: dict[tuple[bytes, object], int] = {}
    for index, identity in enumerate(identities):
        representative.setdefault(identity, index)
    collapsed: dict[tuple[int, ...], Fraction] = {}
    for term in polynomial.terms:
        exponents = [0] * len(term.exponents)
        coefficient = term.coefficient.as_fraction()
        for axis, degree in enumerate(term.exponents):
            representative_axis = representative[identities[axis]]
            exponents[representative_axis] += degree
            if degree:
                coefficient_bits = _bits_for_fraction(coefficient)
                factor_bits = _bits_for_fraction(factors[axis])
                ledger.charge(
                    "multiplication",
                    degree * (coefficient_bits + factor_bits + 1),
                )
            coefficient *= factors[axis] ** degree
        slot = tuple(exponents)
        ledger.charge("addition", _bits_for_fraction(coefficient) + 1)
        collapsed[slot] = collapsed.get(slot, Fraction(0)) + coefficient
    return all(coefficient == 0 for coefficient in collapsed.values())


class _Ledger:
    def __init__(self) -> None:
        self.work = 0
        self.limits = RationalFunctionBoundLimits(
            raw_terms=4_096,
            raw_digits=4_096,
            result_exponent=64,
            result_terms=256,
            result_digits=128,
            label="Rational map composition",
            reject=_reject,
        )

    def charge(self, category: BoundWorkCategory, amount: int) -> None:
        request_checkpoint(f"during rational map composition {category}")
        self.work += amount
        if self.work > MAX_COMPOSITION_WORK:
            _reject(
                "work_budget",
                "rational map composition exceeds its aggregate coefficient-work bound",
            )


class _Allocation:
    """Bound exact sparse terms and coefficient bits retained by the result."""

    def __init__(self) -> None:
        self.terms = 0
        self.bits = 0

    def charge(self, terms: int, bits: int) -> None:
        request_checkpoint("during rational map composition allocation")
        self.terms += terms
        self.bits += bits
        if self.terms > MAX_COMPOSITION_TERMS:
            _reject(
                "allocation",
                "composition source, guards, or complete output exceeds "
                f"{MAX_COMPOSITION_TERMS} polynomial terms",
            )
        if self.bits > MAX_COMPOSITION_BITS:
            _reject(
                "allocation",
                "composition source, guards, or complete output exceeds "
                f"{MAX_COMPOSITION_BITS} coefficient bits",
            )


def _support_bound(polynomial: PolynomialBound) -> int:
    """Use the shared coordinate and total-degree support bound."""
    return _total_degree_term_bound(polynomial)


def _check_raw_exponents(polynomial: PolynomialBound) -> None:
    if any(degree > MAX_COMPOSITION_RAW_EXPONENT for degree in polynomial.degrees):
        _reject(
            "intermediate_exponent",
            "substitution exceeds the sparse polynomial exponent envelope",
        )


@dataclass(frozen=True)
class _SubstitutionBound:
    numerator: PolynomialBound
    denominator: PolynomialBound


@dataclass(frozen=True)
class _PreparedPolynomial:
    """A backend-free sparse polynomial retained for execution."""

    terms: tuple[tuple[Fraction, tuple[int, ...]], ...]


@dataclass(frozen=True)
class _PreparedComposition:
    numerator: _PreparedPolynomial | None
    numerator_denominator: _PreparedPolynomial | None
    denominator: _PreparedPolynomial | None
    denominator_denominator: _PreparedPolynomial | None


def _bits_for_fraction(value: Fraction) -> int:
    return value.numerator.bit_length() + value.denominator.bit_length()


def _charge_source(
    value: RationalFunction, ledger: _Ledger, allocation: _Allocation
) -> None:
    for polynomial in (value.numerator, value.denominator):
        allocation.charge(
            len(polynomial.terms),
            sum(
                _bits_for_fraction(term.coefficient.as_fraction())
                for term in polynomial.terms
            ),
        )
    ledger.charge(
        "source_conversion",
        (len(value.numerator.terms) + len(value.denominator.terms))
        * (len(value.variables) + 1),
    )


def _power(
    source: PolynomialBound,
    exponent: int,
    variable_count: int,
    ledger: _Ledger,
) -> PolynomialBound:
    if exponent == 0:
        return _one_polynomial(variable_count)
    if source.is_zero:
        return _zero_polynomial(variable_count)
    result = _one_polynomial(variable_count)
    for _ in range(exponent):
        result = _multiply_polynomials(result, source, ledger)
    return result


def _substitute_bound(
    polynomial: SparseRationalPolynomial,
    inner_bounds: tuple[FractionBound, ...],
    ledger: _Ledger,
) -> _SubstitutionBound:
    """Bound a cleared-denominator substitution before any CAS expansion."""
    variable_count = len(inner_bounds[0].numerator.degrees) if inner_bounds else 0
    if not polynomial.terms:
        return _SubstitutionBound(
            numerator=_zero_polynomial(variable_count),
            denominator=_one_polynomial(variable_count),
        )
    powers = tuple(
        max(term.exponents[axis] for term in polynomial.terms)
        for axis in range(len(inner_bounds))
    )
    denominator = _one_polynomial(variable_count)
    for bound, exponent in zip(inner_bounds, powers, strict=True):
        denominator = _multiply_polynomials(
            denominator,
            _power(bound.denominator, exponent, variable_count, ledger),
            ledger,
        )
        _check_raw_exponents(denominator)
    numerator: PolynomialBound | None = None
    for term in polynomial.terms:
        product = _one_polynomial(variable_count)
        coefficient = term.coefficient.as_fraction()
        # A zero inner coordinate annihilates this monomial when its exponent
        # is positive. The denominator is still formed from its canonical one.
        for bound, exponent, common_power in zip(
            inner_bounds, term.exponents, powers, strict=True
        ):
            if bound.is_zero and exponent:
                product = _zero_polynomial(variable_count)
                break
            product = _multiply_polynomials(
                product,
                _power(bound.numerator, exponent, variable_count, ledger),
                ledger,
            )
            _check_raw_exponents(product)
            product = _multiply_polynomials(
                product,
                _power(
                    bound.denominator,
                    common_power - exponent,
                    variable_count,
                    ledger,
                ),
                ledger,
            )
            _check_raw_exponents(product)
        if product.is_zero:
            continue
        product = replace(
            product,
            coefficient_digits=product.coefficient_digits
            + max(1, len(str(abs(coefficient)))),
            rational_content=product.rational_content * coefficient,
        )
        numerator = (
            product
            if numerator is None
            else _add_polynomials(numerator, product, ledger)
        )
    if numerator is None:
        numerator = _zero_polynomial(variable_count)
    return _SubstitutionBound(numerator=numerator, denominator=denominator)


def _monomial_inner(value: RationalFunction) -> tuple[Fraction, tuple[int, ...]] | None:
    if len(value.denominator.terms) != 1 or value.denominator.terms[0].exponents != (
        0,
    ) * len(value.variables):
        return None
    if len(value.numerator.terms) != 1:
        return None
    term = value.numerator.terms[0]
    return term.coefficient.as_fraction(), term.exponents


def _substitute_monomial(
    polynomial: SparseRationalPolynomial,
    monomials: tuple[tuple[Fraction, tuple[int, ...]], ...],
    ledger: _Ledger,
    *,
    charge: bool = True,
) -> _PreparedPolynomial:
    if not polynomial.terms:
        return _PreparedPolynomial(())
    combined: dict[tuple[int, ...], Fraction] = {}
    for source_term in polynomial.terms:
        coefficient = source_term.coefficient.as_fraction()
        exponents = [0] * len(monomials[0][1]) if monomials else []
        for scalar, monomial_exponents, exponent in zip(
            (item[0] for item in monomials),
            (item[1] for item in monomials),
            source_term.exponents,
            strict=True,
        ):
            coefficient *= scalar**exponent
            for axis, power in enumerate(monomial_exponents):
                exponents[axis] += power * exponent
        exponent_tuple = tuple(exponents)
        if any(exponent > MAX_COMPOSITION_RAW_EXPONENT for exponent in exponent_tuple):
            _reject(
                "intermediate_exponent",
                "substitution exceeds the sparse polynomial exponent envelope",
            )
        combined[exponent_tuple] = (
            combined.get(exponent_tuple, Fraction(0)) + coefficient
        )
        if charge:
            ledger.charge("multiplication", len(monomials) + len(exponent_tuple) + 1)
    return _PreparedPolynomial(
        tuple(
            (coefficient, exponents)
            for exponents, coefficient in sorted(combined.items(), reverse=True)
            if coefficient
        )
    )


def _prepared_to_sparse(value: _PreparedPolynomial) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coefficient),
                exponents=exponents,
            )
            for coefficient, exponents in value.terms
        )
    )


def _monic_guard(value: RationalFunction) -> RationalPolynomial | None:
    if not value.numerator.terms:
        _reject_undefined_outer_denominator()
    if len(value.numerator.terms) == 1 and not any(value.numerator.terms[0].exponents):
        return None
    polynomial = sparse_rational_polynomial_to_sympy(value.numerator, value.variables)
    return rational_polynomial_from_sympy(polynomial.monic(), value.variables)


def _constant_fraction(value: RationalFunction) -> Fraction:
    if value.variables:
        raise AssertionError("constant evaluator requires an empty source axis")
    if not value.numerator.terms:
        return Fraction(0)
    return value.numerator.terms[0].coefficient.as_fraction()


def _constant_substitute_polynomial(
    polynomial: SparseRationalPolynomial,
    values: tuple[Fraction, ...],
    ledger: _Ledger,
) -> Fraction:
    result = Fraction(0)
    for term in polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        for value, exponent in zip(values, term.exponents, strict=True):
            ledger.charge("multiplication", exponent + 1)
            coefficient *= value**exponent
        ledger.charge("addition", 1)
        result += coefficient
    return result


def _constant_map_composition(
    outer: RationalFunctionMap,
    inner: RationalFunctionMap,
    ledger: _Ledger,
    allocation: _Allocation,
) -> RationalFunctionMapComposition:
    for component in (*outer.components, *inner.components):
        request_checkpoint("before constant-map source recognition")
        _recognize_source(component)
    values = tuple(_constant_fraction(component) for component in inner.components)
    guards: tuple[RationalPolynomial, ...] = ()
    composites: list[RationalFunction] = []
    for outer_component in outer.components:
        numerator = _constant_substitute_polynomial(
            outer_component.numerator, values, ledger
        )
        denominator = _constant_substitute_polynomial(
            outer_component.denominator, values, ledger
        )
        if denominator == 0:
            raise OperationDomainValidationError(
                location=("outer", "components"),
                code="rational_function_map.compose.undefined_outer_denominator",
                message="an outer denominator vanishes identically after substitution",
            )
        result_value = numerator / denominator
        result_digits = max(
            len(str(abs(result_value.numerator))), len(str(result_value.denominator))
        )
        if result_digits > 128:
            _reject(
                "result_height",
                "constant composite exceeds the 128-digit canonical coefficient bound",
            )
        ledger.charge("normalization", result_digits)
        allocation.charge(1, _bits_for_fraction(result_value))
        composites.append(
            RationalFunction._from_kernel(
                variables=(),
                numerator=SparseRationalPolynomial(
                    terms=()
                    if numerator == 0
                    else (
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational.from_fraction(result_value),
                            exponents=(),
                        ),
                    )
                ),
                denominator=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1), exponents=()
                        ),
                    )
                ),
            )
        )
    return RationalFunctionMapComposition(
        outer=outer,
        inner=inner,
        composite=RationalFunctionMap(
            source_variables=inner.source_variables,
            target_coordinates=outer.target_coordinates,
            components=tuple(composites),
        ),
        construction_locus_guard=guards,
    )


def _prepared_poly_to_sympy(
    value: _PreparedPolynomial, variables: tuple[str, ...]
) -> Any:
    from jacobian.math.polynomials._conversions import (
        sparse_rational_polynomial_to_sympy,
    )

    return sparse_rational_polynomial_to_sympy(_prepared_to_sparse(value), variables)


def compose_maps(  # noqa: C901
    outer: RationalFunctionMap,
    inner: RationalFunctionMap,
) -> RationalFunctionMapComposition:
    """Compose two canonical rational maps on their retained construction locus."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return compose_maps(outer, inner)
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before rational map composition admission")
    if inner.target_coordinates != outer.source_variables:
        raise OperationDomainValidationError(
            location=("inner", "target_coordinates"),
            code="rational_function_map.compose.axis_mismatch",
            message="inner target coordinates must equal outer source variables",
        )
    ledger = _Ledger()
    allocation = _Allocation()
    for value in (*outer.components, *inner.components):
        _charge_source(value, ledger, allocation)

    outer_bounds = tuple(
        _fraction_bound(component, ledger) for component in outer.components
    )
    inner_bounds = tuple(
        _fraction_bound(component, ledger) for component in inner.components
    )
    for component_bound in (*outer_bounds, *inner_bounds):
        # This is caller-authored semantic recognition; charge it after the
        # complete source representation has been admitted.
        ledger.charge("recognition", _recognition_work_units(component_bound))

    # Empty source axes contain only exact constants and cannot be represented
    # by SymPy Poly. Handle that degenerate field after source admission.
    if not inner.source_variables:
        return _constant_map_composition(outer, inner, ledger, allocation)

    monomials = tuple(_monomial_inner(component) for component in inner.components)
    use_monomial_path = bool(monomials) and all(item is not None for item in monomials)
    guards: list[RationalPolynomial] = []
    prepared: list[_PreparedComposition] = []

    # Inner denominators are construction guards, including denominators that
    # happen not to occur in a particular outer coordinate.
    seen_inner_guards: set[bytes] = set()
    for component in inner.components:
        denominator = component.denominator
        if denominator.terms and any(denominator.terms[0].exponents):
            guard = RationalPolynomial(
                variables=inner.source_variables,
                polynomial=denominator,
            )
            key = guard_key(guard)
            if key in seen_inner_guards:
                continue
            seen_inner_guards.add(key)
            guards.append(guard)
            allocation.charge(
                len(guard.polynomial.terms),
                sum(
                    _bits_for_fraction(t.coefficient.as_fraction())
                    for t in guard.polynomial.terms
                ),
            )

    for outer_component in outer.components:
        outer_numerator_bound = _substitute_bound(
            outer_component.numerator, inner_bounds, ledger
        )
        outer_denominator_bound = _substitute_bound(
            outer_component.denominator, inner_bounds, ledger
        )
        raw_numerator_bound = _multiply_polynomials(
            outer_numerator_bound.numerator,
            outer_denominator_bound.denominator,
            ledger,
        )
        _check_raw_exponents(raw_numerator_bound)
        raw_denominator_bound = _multiply_polynomials(
            outer_numerator_bound.denominator,
            outer_denominator_bound.numerator,
            ledger,
        )
        if _equal_inner_substitution_vanishes(
            outer_component.denominator, inner.components, ledger
        ):
            _reject_undefined_outer_denominator()
        _check_raw_exponents(raw_denominator_bound)
        if raw_denominator_bound.is_zero:
            _reject_undefined_outer_denominator()
        result_bound = _remove_guaranteed_common_monomial(
            FractionBound(raw_numerator_bound, raw_denominator_bound)
        )
        ledger.charge(
            "source_conversion",
            _polynomial_backend_conversion_work_units(outer_numerator_bound.numerator)
            + _polynomial_backend_conversion_work_units(
                outer_numerator_bound.denominator
            )
            + _polynomial_backend_conversion_work_units(
                outer_denominator_bound.numerator
            )
            + _polynomial_backend_conversion_work_units(
                outer_denominator_bound.denominator
            ),
        )
        digits = _validate_canonical_result_bound(result_bound, ledger)
        denominator_is_unit = all(
            degree == 0 for degree in result_bound.denominator.degrees
        )
        result_terms = sum(
            polynomial.terms if denominator_is_unit else _support_bound(polynomial)
            for polynomial in (result_bound.numerator, result_bound.denominator)
        )
        allocation.charge(result_terms, result_terms * 8 * digits)
        if not (
            len(outer_component.denominator.terms) == 1
            and not any(outer_component.denominator.terms[0].exponents)
        ):
            guard_digits = _validate_canonical_result_bound(
                _remove_guaranteed_common_monomial(
                    FractionBound(
                        outer_denominator_bound.numerator,
                        outer_denominator_bound.denominator,
                    )
                ),
                ledger,
            )
            guard_terms = _support_bound(outer_denominator_bound.numerator)
            allocation.charge(guard_terms, guard_terms * 8 * guard_digits)
        if use_monomial_path:
            monomial_values = tuple(item for item in monomials if item is not None)
            for polynomial in (
                outer_component.numerator,
                outer_component.denominator,
            ):
                ledger.charge(
                    "multiplication",
                    len(polynomial.terms)
                    * (len(monomial_values) + len(inner.source_variables) + 1),
                )
            prepared.append(
                _PreparedComposition(
                    numerator=None,
                    numerator_denominator=None,
                    denominator=None,
                    denominator_denominator=None,
                )
            )
        else:
            prepared.append(
                _PreparedComposition(
                    numerator=None,
                    numerator_denominator=None,
                    denominator=None,
                    denominator_denominator=None,
                )
            )

    # #2880's source bounds are cached above; no caller-side canonical check is
    # performed until every substitution bound and retained allocation is known.
    # Every source, substitution bound, output bound and guard allocation has
    # been admitted. Only now construct backend polynomials and normalize.
    require_canonical = tuple(
        component for component in (*outer.components, *inner.components)
    )
    recognized: set[bytes] = set()
    for component in require_canonical:
        identity = _component_identity(component)
        if identity in recognized:
            continue
        recognized.add(identity)
        request_checkpoint("before composition source recognition")
        _recognize_source(component)
    outer_components = require_canonical[: len(outer.components)]
    inner_components = require_canonical[len(outer.components) :]
    if use_monomial_path:
        monomial_values = tuple(item for item in monomials if item is not None)
        prepared = [
            _PreparedComposition(
                numerator=_substitute_monomial(
                    component.numerator, monomial_values, ledger, charge=False
                ),
                numerator_denominator=_substitute_monomial(
                    SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=(0,) * len(outer.source_variables),
                            ),
                        )
                    ),
                    monomial_values,
                    ledger,
                    charge=False,
                ),
                denominator=_substitute_monomial(
                    component.denominator, monomial_values, ledger, charge=False
                ),
                denominator_denominator=_substitute_monomial(
                    SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(num=1, den=1),
                                exponents=(0,) * len(outer.source_variables),
                            ),
                        )
                    ),
                    monomial_values,
                    ledger,
                    charge=False,
                ),
            )
            for component in outer_components
        ]
    xvars = inner.source_variables
    inner_num = []
    for component in inner_components:
        request_checkpoint("before rational map composition inner numerator conversion")
        inner_num.append(
            sparse_rational_polynomial_to_sympy(component.numerator, xvars)
        )
    inner_den = []
    for component in inner_components:
        request_checkpoint(
            "before rational map composition inner denominator conversion"
        )
        inner_den.append(
            sparse_rational_polynomial_to_sympy(component.denominator, xvars)
        )

    def substitute(polynomial: SparseRationalPolynomial) -> tuple[Any, Any]:
        from sympy import Poly

        request_checkpoint("before rational map composition substitution")
        xgens = symbols_for_variables(xvars)
        if not polynomial.terms:
            one = Poly(1, *xgens, domain="QQ")
            return Poly(0, *xgens, domain="QQ"), one
        powers = tuple(
            max(term.exponents[axis] for term in polynomial.terms)
            for axis in range(len(inner_components))
        )
        common_denominator = Poly(1, *xgens, domain="QQ")
        for denominator, exponent in zip(inner_den, powers, strict=True):
            common_denominator *= denominator**exponent
        numerator = Poly(0, *xgens, domain="QQ")
        for term in polynomial.terms:
            request_checkpoint("during rational map composition substitution term")
            value = Poly(term.coefficient.as_fraction(), *xgens, domain="QQ")
            for num, den, exponent, max_exponent in zip(
                inner_num, inner_den, term.exponents, powers, strict=True
            ):
                value *= num**exponent * den ** (max_exponent - exponent)
            numerator += value
        return numerator, common_denominator

    composites: list[RationalFunction] = []
    composite_rows: dict[bytes, RationalFunction] = {}
    for outer_component, prepared_component in zip(
        outer_components, prepared, strict=True
    ):
        request_checkpoint("before rational map composition output row")
        identity = _component_identity(outer_component)
        cached = composite_rows.get(identity)
        if cached is not None:
            composites.append(cached)
            continue
        if use_monomial_path:
            assert (
                prepared_component.numerator is not None
                and prepared_component.numerator_denominator is not None
                and prepared_component.denominator is not None
                and prepared_component.denominator_denominator is not None
            )
            p_num = _prepared_poly_to_sympy(prepared_component.numerator, xvars)
            p_den = _prepared_poly_to_sympy(
                prepared_component.numerator_denominator, xvars
            )
            q_num = _prepared_poly_to_sympy(prepared_component.denominator, xvars)
            q_den = _prepared_poly_to_sympy(
                prepared_component.denominator_denominator, xvars
            )
        else:
            p_num, p_den = substitute(outer_component.numerator)
            q_num, q_den = substitute(outer_component.denominator)
        if q_num.is_zero:
            _reject_undefined_outer_denominator()
        denominator_value = _normalize_fraction(q_num, q_den, xvars)
        outer_guard = _monic_guard(denominator_value)
        if outer_guard is not None:
            guards.append(outer_guard)
        value = _normalize_fraction(p_num * q_den, p_den * q_num, xvars)
        composite_rows[identity] = value
        composites.append(value)
    guards = sorted(
        {guard_key(guard): guard for guard in guards}.values(), key=guard_key
    )
    result = RationalFunctionMapComposition(
        outer=outer,
        inner=inner,
        composite=RationalFunctionMap(
            source_variables=inner.source_variables,
            target_coordinates=outer.target_coordinates,
            components=tuple(composites),
        ),
        construction_locus_guard=tuple(guards),
    )
    request_checkpoint("after rational map composition construction")
    return result


__all__ = ["compose_maps"]
