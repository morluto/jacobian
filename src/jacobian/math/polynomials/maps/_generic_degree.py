"""Exact replay of generic-fiber Gröbner certificates."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

from jacobian.math.polynomials.maps._models import (
    MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_BITS,
    MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_OPERATIONS,
    MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_PRODUCTS,
    MAX_GENERIC_FIBER_REPLAY_PRODUCTS,
    MAX_GENERIC_FIBER_REPLAY_REDUCTION_STEPS,
    MAX_GENERIC_FIBER_REPLAY_SOURCE_TERMS,
    MAX_GENERIC_FIBER_STANDARD_MONOMIALS,
)

if TYPE_CHECKING:
    from jacobian.math.polynomials.maps._models import GenericFiberCertificate
    from jacobian.math.polynomials.maps.values import RationalPolynomialMap
    from jacobian.math.polynomials.values import RationalFunction

MathematicalOutcome = Literal[
    "GENERICALLY_FINITE",
    "NOT_DOMINANT",
    "DOMINANT_NOT_GENERICALLY_FINITE",
]


class GenericFiberReplayLimitError(ValueError):
    """A certificate replay exceeded its declared finite work envelope."""


class _ReplayBudget:
    """Cumulative declared-work counters charged during one certificate replay."""

    def __init__(self) -> None:
        self.reduction_steps = 0
        self.coefficient_operations = 0
        self.coefficient_products = 0

    def charge_reduction_steps(self, steps: int) -> None:
        self.reduction_steps += steps
        if self.reduction_steps > MAX_GENERIC_FIBER_REPLAY_REDUCTION_STEPS:
            raise GenericFiberReplayLimitError(
                "generic-fiber replay exceeds the declared reduction-step bound"
            )

    def charge_coefficient_operations(self, operations: int) -> None:
        self.coefficient_operations += operations
        if (
            self.coefficient_operations
            > MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_OPERATIONS
        ):
            raise GenericFiberReplayLimitError(
                "generic-fiber replay exceeds the declared coefficient-operation bound"
            )

    def charge_coefficient_products(self, products: int) -> None:
        self.coefficient_products += products
        if self.coefficient_products > MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_PRODUCTS:
            raise GenericFiberReplayLimitError(
                "generic-fiber reconstruction exceeds the declared "
                "coefficient-product bound"
            )


def _parameter_field(parameter_symbols: tuple[Any, ...]) -> Any:
    from sympy import QQ

    return QQ.frac_field(*parameter_symbols)


def _replay_ring(source_count: int, *, field: Any) -> Any:
    """Build one lexicographic source-variable ring over the parameter field."""

    from sympy import Dummy
    from sympy.polys.rings import ring as make_polynomial_ring

    source_symbols = tuple(
        Dummy(f"source_{index + 1}") for index in range(source_count)
    )
    return make_polynomial_ring(source_symbols, field, "lex")[0]


def _integer_bits(value: Any) -> int:
    return abs(int(value)).bit_length()


def _require_bounded_intermediate(polynomial: Any) -> None:
    terms = polynomial.terms()
    if len(terms) > MAX_GENERIC_FIBER_REPLAY_SOURCE_TERMS:
        raise GenericFiberReplayLimitError(
            "generic-fiber replay intermediate exceeds the declared term bound"
        )
    bits = 0
    for _monomial, coefficient in terms:
        for part in (coefficient.numer, coefficient.denom):
            for value in part.coeffs():
                bits = max(bits, _integer_bits(value))
    if bits > MAX_GENERIC_FIBER_REPLAY_COEFFICIENT_BITS:
        raise GenericFiberReplayLimitError(
            "generic-fiber replay intermediate exceeds the declared "
            "coefficient-height bound"
        )


def _fraction_field_element(
    coefficient: RationalFunction,
    *,
    field: Any,
) -> Any:
    """Materialize one canonical rational function as a fraction-field element."""

    parameter_ring = field.field.ring

    def part(terms: Any) -> Any:
        if not terms:
            return parameter_ring.zero
        return parameter_ring.from_dict(
            {
                tuple(term.exponents): Fraction(*term.coefficient.as_integer_ratio())
                for term in terms
            }
        )

    return field.field.new(part(coefficient.numerator.terms)) / field.field.new(
        part(coefficient.denominator.terms)
    )


def _certificate_polynomial(
    polynomial: Any,
    *,
    field: Any,
    ring: Any,
) -> Any:
    terms = {
        tuple(term.source_exponents): _fraction_field_element(
            term.coefficient,
            field=field,
        )
        for term in polynomial.terms
    }
    return ring.from_dict(terms)


def _source_generators(
    source: RationalPolynomialMap,
    *,
    source_variable_order: tuple[str, ...],
    field: Any,
    ring: Any,
) -> tuple[Any, ...]:
    canonical_positions = {
        variable: index for index, variable in enumerate(source.input_variables)
    }
    permutation = tuple(
        canonical_positions[variable] for variable in source_variable_order
    )
    zero_exponents = (0,) * len(ring.gens)
    generators = []
    for row, component in enumerate(source.output_polynomials):
        terms: dict[tuple[int, ...], Fraction] = {}
        for term in component.polynomial.terms:
            exponents = tuple(term.exponents[index] for index in permutation)
            terms[exponents] = Fraction(*term.coefficient.as_integer_ratio())
        generator = ring.from_dict(terms)
        generators.append(generator - ring.from_dict({zero_exponents: field.gens[row]}))
    return tuple(generators)


def _leading_exponent(polynomial: Any) -> tuple[int, ...]:
    if polynomial.is_zero:
        raise ValueError("generic-fiber basis polynomials must be nonzero")
    monomial, _coefficient = polynomial.LT
    return tuple(int(exponent) for exponent in monomial)


def _divides(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _shift_polynomial(polynomial: Any, shift: tuple[int, ...]) -> Any:
    monomial = polynomial.ring.from_dict({tuple(shift): polynomial.ring.domain.one})
    return polynomial * monomial


def _require_reduced_monic_basis(
    basis: tuple[Any, ...],
) -> tuple[tuple[int, ...], ...]:
    leading_exponents = tuple(_leading_exponent(polynomial) for polynomial in basis)
    if leading_exponents != tuple(sorted(leading_exponents)):
        raise ValueError(
            "generic-fiber basis must be ordered by ascending leading monomial"
        )
    for index, polynomial in enumerate(basis):
        if polynomial.LT[1] != 1:
            raise ValueError("generic-fiber basis polynomials must be monic")
        other_leads = tuple(
            leading
            for other_index, leading in enumerate(leading_exponents)
            if other_index != index
        )
        for exponents, _coefficient in polynomial.terms():
            if any(_divides(leading, exponents) for leading in other_leads):
                raise ValueError("generic-fiber basis must be reduced")
    return leading_exponents


def _reduce(
    polynomial: Any,
    basis: tuple[Any, ...],
    *,
    budget: _ReplayBudget,
) -> Any:
    budget.charge_reduction_steps(len(polynomial.terms()) * len(basis))
    _quotients, remainder = polynomial.div(list(basis))
    _require_bounded_intermediate(remainder)
    return remainder


def _require_buchberger_criterion(
    basis: tuple[Any, ...],
    leading_exponents: tuple[tuple[int, ...], ...],
    *,
    budget: _ReplayBudget,
) -> None:
    for (left_index, left), (right_index, right) in combinations(
        enumerate(basis),
        2,
    ):
        left_leading = leading_exponents[left_index]
        right_leading = leading_exponents[right_index]
        least_common_multiple = tuple(
            max(a, b) for a, b in zip(left_leading, right_leading, strict=True)
        )
        left_shift = tuple(
            common - exponent
            for common, exponent in zip(
                least_common_multiple,
                left_leading,
                strict=True,
            )
        )
        right_shift = tuple(
            common - exponent
            for common, exponent in zip(
                least_common_multiple,
                right_leading,
                strict=True,
            )
        )
        budget.charge_coefficient_operations(len(left.terms()) + len(right.terms()))
        s_polynomial = _shift_polynomial(left, left_shift) - _shift_polynomial(
            right,
            right_shift,
        )
        _require_bounded_intermediate(s_polynomial)
        if not _reduce(s_polynomial, basis, budget=budget).is_zero:
            raise ValueError("generic-fiber basis fails Buchberger's criterion")


def _has_pure_power(
    leading_exponents: tuple[tuple[int, ...], ...],
    variable_index: int,
) -> bool:
    return any(
        exponents[variable_index] > 0
        and all(
            exponent == 0
            for index, exponent in enumerate(exponents)
            if index != variable_index
        )
        for exponents in leading_exponents
    )


def _standard_monomials(
    leading_exponents: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    variable_count = len(leading_exponents[0])
    if not all(
        _has_pure_power(leading_exponents, variable_index)
        for variable_index in range(variable_count)
    ):
        return None
    origin = (0,) * variable_count
    found = {origin}
    frontier = [origin]
    while frontier:
        current = frontier.pop()
        for variable_index in range(variable_count):
            shifted = list(current)
            shifted[variable_index] += 1
            candidate = tuple(shifted)
            if candidate in found or any(
                _divides(leading, candidate) for leading in leading_exponents
            ):
                continue
            if len(found) >= MAX_GENERIC_FIBER_STANDARD_MONOMIALS:
                raise ValueError(
                    "generic-fiber quotient exceeds the standard-monomial bound"
                )
            found.add(candidate)
            frontier.append(candidate)
    return tuple(sorted(found))


def enumerate_standard_monomials(
    leading_exponents: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Return the bounded standard-monomial complement for one leading ideal."""

    try:
        return _standard_monomials(leading_exponents)
    except ValueError as exc:
        raise GenericFiberReplayLimitError(str(exc)) from exc


def _materialize_certificate(
    source: RationalPolynomialMap,
    certificate: GenericFiberCertificate,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Check the certificate skeleton against its source and materialize it."""

    expected_parameters = tuple(
        f"t{index + 1}" for index in range(len(source.output_polynomials))
    )
    if certificate.target_parameters != expected_parameters:
        raise ValueError("generic target parameters must follow component order")
    if set(certificate.source_variable_order) != set(source.input_variables):
        raise ValueError(
            "generic-fiber source variable order must permute the source axis"
        )
    if len(certificate.basis_from_source) != len(source.output_polynomials):
        raise ValueError(
            "generic-fiber transformation rows must match source generators"
        )

    from sympy import Dummy

    parameter_symbols = tuple(
        Dummy(f"target_{index + 1}")
        for index in range(len(certificate.target_parameters))
    )
    field = _parameter_field(parameter_symbols)
    ring = _replay_ring(len(certificate.source_variable_order), field=field)
    basis = tuple(
        _certificate_polynomial(polynomial, field=field, ring=ring)
        for polynomial in certificate.basis
    )
    transformation = tuple(
        tuple(
            _certificate_polynomial(polynomial, field=field, ring=ring)
            for polynomial in row
        )
        for row in certificate.basis_from_source
    )
    source_generators = _source_generators(
        source,
        source_variable_order=certificate.source_variable_order,
        field=field,
        ring=ring,
    )

    replay_products = sum(
        len(source_generator.terms()) * len(transformation[row][column].terms())
        for row, source_generator in enumerate(source_generators)
        for column in range(len(basis))
    )
    if replay_products > MAX_GENERIC_FIBER_REPLAY_PRODUCTS:
        raise GenericFiberReplayLimitError(
            "generic-fiber reconstruction exceeds the declared replay-product bound"
        )
    return ring, basis, transformation, source_generators


def _require_reconstruction(
    ring: Any,
    basis: tuple[Any, ...],
    transformation: tuple[Any, ...],
    source_generators: tuple[Any, ...],
    *,
    budget: _ReplayBudget,
) -> None:
    """Verify ``basis = source_generators * basis_from_source`` exactly."""

    zero = ring.zero
    for column, polynomial in enumerate(basis):
        reconstructed = zero
        for row, generator in enumerate(source_generators):
            factor = transformation[row][column]
            budget.charge_coefficient_products(
                len(generator.terms()) * len(factor.terms())
            )
            reconstructed = reconstructed + generator * factor
        if reconstructed != polynomial:
            raise ValueError(
                "generic-fiber basis does not reconstruct from the source generators"
            )


def require_certificate_reconstructs_from_source(
    source: RationalPolynomialMap,
    certificate: GenericFiberCertificate,
) -> None:
    """Bind one certificate to its stated source map within declared bounds.

    Raises unless ``basis[j]`` equals the exact combination of this source's
    generic-fiber generators named by ``basis_from_source[:, j]``, so evidence
    computed for one map cannot be presented against another.
    """

    ring, basis, transformation, source_generators = _materialize_certificate(
        source,
        certificate,
    )
    _require_reconstruction(
        ring,
        basis,
        transformation,
        source_generators,
        budget=_ReplayBudget(),
    )


def validate_generic_fiber_certificate(
    source: RationalPolynomialMap,
    certificate: GenericFiberCertificate,
) -> tuple[MathematicalOutcome, int | None]:
    """Replay the generic-fiber ideal, Gröbner basis, and quotient dimension."""

    ring, basis, transformation, source_generators = _materialize_certificate(
        source,
        certificate,
    )

    budget = _ReplayBudget()
    leading_exponents = _require_reduced_monic_basis(basis)
    _require_buchberger_criterion(basis, leading_exponents, budget=budget)
    if any(
        not _reduce(generator, basis, budget=budget).is_zero
        for generator in source_generators
    ):
        raise ValueError("source generic-fiber generators do not reduce to zero")
    _require_reconstruction(
        ring,
        basis,
        transformation,
        source_generators,
        budget=budget,
    )

    zero_exponents = (0,) * len(ring.gens)
    if leading_exponents == (zero_exponents,):
        if certificate.standard_monomials:
            raise ValueError("the unit generic-fiber ideal has no standard monomials")
        return "NOT_DOMINANT", None

    standard_monomials = enumerate_standard_monomials(leading_exponents)
    if standard_monomials is None:
        if certificate.standard_monomials:
            raise ValueError(
                "a positive-dimensional generic fiber has no finite monomial basis"
            )
        return "DOMINANT_NOT_GENERICALLY_FINITE", None
    if certificate.standard_monomials != standard_monomials:
        raise ValueError(
            "standard monomials do not match the generic-fiber leading ideal"
        )
    if not standard_monomials:
        raise ValueError("a proper zero-dimensional quotient must have positive degree")
    return "GENERICALLY_FINITE", len(standard_monomials)


__all__ = [
    "GenericFiberReplayLimitError",
    "enumerate_standard_monomials",
    "require_certificate_reconstructs_from_source",
    "validate_generic_fiber_certificate",
]
