"""Exact replay of generic-fiber Gröbner certificates."""

from __future__ import annotations

from itertools import combinations, product
from typing import TYPE_CHECKING, Any, Literal

from jacobian.math.polynomials.maps._models import (
    MAX_GENERIC_FIBER_REPLAY_PRODUCTS,
    MAX_GENERIC_FIBER_STANDARD_MONOMIAL_CANDIDATES,
    MAX_GENERIC_FIBER_STANDARD_MONOMIALS,
)

if TYPE_CHECKING:
    from sympy import Poly

    from jacobian.math.polynomials.maps._models import GenericFiberCertificate
    from jacobian.math.polynomials.maps.values import RationalPolynomialMap
    from jacobian.math.polynomials.values import (
        RationalFunction,
        SparseRationalPolynomial,
    )

MathematicalOutcome = Literal[
    "GENERICALLY_FINITE",
    "NOT_DOMINANT",
    "DOMINANT_NOT_GENERICALLY_FINITE",
]


class GenericFiberReplayLimitError(ValueError):
    """A certificate replay exceeded its declared finite work envelope."""


def _sparse_expression(
    polynomial: SparseRationalPolynomial,
    variables: tuple[Any, ...],
) -> Any:
    from sympy import Add, Integer, Mul, Pow, Rational

    terms: list[Any] = []
    for term in polynomial.terms:
        coefficient = Rational(*term.coefficient.as_integer_ratio())
        factors = [
            Pow(variable, exponent)
            for variable, exponent in zip(
                variables,
                term.exponents,
                strict=True,
            )
            if exponent
        ]
        terms.append(Mul(coefficient, *factors))
    return Add(*terms) if terms else Integer(0)


def _coefficient_expression(
    coefficient: RationalFunction,
    parameters: tuple[Any, ...],
) -> Any:
    numerator = _sparse_expression(coefficient.numerator, parameters)
    denominator = _sparse_expression(coefficient.denominator, parameters)
    return numerator / denominator


def _certificate_polynomial(
    polynomial: Any,
    *,
    source_symbols: tuple[Any, ...],
    parameter_symbols: tuple[Any, ...],
    field: Any,
) -> Poly:
    from sympy import Poly

    return Poly.from_dict(
        {
            term.source_exponents: _coefficient_expression(
                term.coefficient,
                parameter_symbols,
            )
            for term in polynomial.terms
        },
        *source_symbols,
        domain=field,
    )


def _source_generators(
    source: RationalPolynomialMap,
    *,
    source_variable_order: tuple[str, ...],
    source_symbols: tuple[Any, ...],
    parameter_symbols: tuple[Any, ...],
    field: Any,
) -> tuple[Poly, ...]:
    from sympy import Poly, Rational

    canonical_positions = {
        variable: index for index, variable in enumerate(source.input_variables)
    }
    permutation = tuple(
        canonical_positions[variable] for variable in source_variable_order
    )
    generators: list[Poly] = []
    for component, parameter in zip(
        source.output_polynomials,
        parameter_symbols,
        strict=True,
    ):
        terms: dict[tuple[int, ...], Any] = {}
        for term in component.polynomial.terms:
            exponents = tuple(term.exponents[index] for index in permutation)
            terms[exponents] = Rational(*term.coefficient.as_integer_ratio())
        zero = (0,) * len(source_symbols)
        terms[zero] = terms.get(zero, 0) - parameter
        generators.append(Poly.from_dict(terms, *source_symbols, domain=field))
    return tuple(generators)


def _leading_exponent(polynomial: Poly) -> tuple[int, ...]:
    terms = polynomial.terms(order="lex")
    if not terms:
        raise ValueError("generic-fiber basis polynomials must be nonzero")
    return tuple(int(exponent) for exponent in terms[0][0])


def _divides(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _shift_polynomial(polynomial: Poly, shift: tuple[int, ...]) -> Poly:
    from sympy import Poly

    return Poly.from_dict(
        {
            tuple(a + b for a, b in zip(exponents, shift, strict=True)): coefficient
            for exponents, coefficient in polynomial.terms()
        },
        *polynomial.gens,
        domain=polynomial.domain,
    )


def _require_reduced_monic_basis(
    basis: tuple[Poly, ...],
) -> tuple[tuple[int, ...], ...]:
    leading_exponents = tuple(_leading_exponent(polynomial) for polynomial in basis)
    if leading_exponents != tuple(sorted(leading_exponents)):
        raise ValueError(
            "generic-fiber basis must be ordered by ascending leading monomial"
        )
    for index, polynomial in enumerate(basis):
        terms = polynomial.terms(order="lex")
        if terms[0][1] != 1:
            raise ValueError("generic-fiber basis polynomials must be monic")
        other_leads = tuple(
            leading
            for other_index, leading in enumerate(leading_exponents)
            if other_index != index
        )
        for exponents, _coefficient in terms:
            if any(_divides(leading, exponents) for leading in other_leads):
                raise ValueError("generic-fiber basis must be reduced")
    return leading_exponents


def _reduce(polynomial: Poly, basis: tuple[Poly, ...]) -> Poly:
    from sympy.polys.polytools import reduced

    _quotients, remainder = reduced(
        polynomial,
        basis,
        *polynomial.gens,
        order="lex",
        domain=polynomial.domain,
    )
    return remainder


def _require_buchberger_criterion(
    basis: tuple[Poly, ...],
    leading_exponents: tuple[tuple[int, ...], ...],
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
        s_polynomial = _shift_polynomial(left, left_shift) - _shift_polynomial(
            right,
            right_shift,
        )
        if not _reduce(s_polynomial, basis).is_zero:
            raise ValueError("generic-fiber basis fails Buchberger's criterion")


def _standard_monomials(
    leading_exponents: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    variable_count = len(leading_exponents[0])
    pure_power_bounds: list[int] = []
    for variable_index in range(variable_count):
        pure_powers = [
            exponents[variable_index]
            for exponents in leading_exponents
            if exponents[variable_index] > 0
            and all(
                exponent == 0
                for index, exponent in enumerate(exponents)
                if index != variable_index
            )
        ]
        if not pure_powers:
            return None
        pure_power_bounds.append(min(pure_powers))
    candidate_count = 1
    for bound in pure_power_bounds:
        candidate_count *= bound
    if candidate_count > MAX_GENERIC_FIBER_STANDARD_MONOMIAL_CANDIDATES:
        raise ValueError(
            "generic-fiber quotient exceeds the standard-monomial candidate bound"
        )
    standard_monomials = tuple(
        exponents
        for exponents in product(*(range(bound) for bound in pure_power_bounds))
        if not any(_divides(leading, exponents) for leading in leading_exponents)
    )
    if len(standard_monomials) > MAX_GENERIC_FIBER_STANDARD_MONOMIALS:
        raise ValueError("generic-fiber quotient exceeds the standard-monomial bound")
    return standard_monomials


def enumerate_standard_monomials(
    leading_exponents: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Return the bounded standard-monomial complement for one leading ideal."""

    try:
        return _standard_monomials(leading_exponents)
    except ValueError as exc:
        raise GenericFiberReplayLimitError(str(exc)) from exc


def validate_generic_fiber_certificate(
    source: RationalPolynomialMap,
    certificate: GenericFiberCertificate,
) -> tuple[MathematicalOutcome, int | None]:
    """Replay the generic-fiber ideal, Gröbner basis, and quotient dimension."""

    from sympy import QQ, Dummy

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

    source_symbols = tuple(
        Dummy(f"source_{index + 1}")
        for index in range(len(certificate.source_variable_order))
    )
    parameter_symbols = tuple(
        Dummy(f"target_{index + 1}")
        for index in range(len(certificate.target_parameters))
    )
    field = QQ.frac_field(*parameter_symbols)
    basis = tuple(
        _certificate_polynomial(
            polynomial,
            source_symbols=source_symbols,
            parameter_symbols=parameter_symbols,
            field=field,
        )
        for polynomial in certificate.basis
    )
    transformation = tuple(
        tuple(
            _certificate_polynomial(
                polynomial,
                source_symbols=source_symbols,
                parameter_symbols=parameter_symbols,
                field=field,
            )
            for polynomial in row
        )
        for row in certificate.basis_from_source
    )
    source_generators = _source_generators(
        source,
        source_variable_order=certificate.source_variable_order,
        source_symbols=source_symbols,
        parameter_symbols=parameter_symbols,
        field=field,
    )

    replay_products = sum(
        len(source_generator.terms()) * len(transformation[row][column].terms())
        for row, source_generator in enumerate(source_generators)
        for column in range(len(basis))
    )
    if replay_products > MAX_GENERIC_FIBER_REPLAY_PRODUCTS:
        raise ValueError("generic-fiber reconstruction exceeds the replay-work bound")

    leading_exponents = _require_reduced_monic_basis(basis)
    _require_buchberger_criterion(basis, leading_exponents)
    if any(not _reduce(generator, basis).is_zero for generator in source_generators):
        raise ValueError("source generic-fiber generators do not reduce to zero")
    zero = basis[0] * 0
    for column, polynomial in enumerate(basis):
        reconstructed = zero
        for row, generator in enumerate(source_generators):
            reconstructed += generator * transformation[row][column]
        if reconstructed != polynomial:
            raise ValueError(
                "generic-fiber basis does not reconstruct from the source generators"
            )

    zero_exponents = (0,) * len(source_symbols)
    if leading_exponents == (zero_exponents,):
        if certificate.standard_monomials:
            raise ValueError("the unit generic-fiber ideal has no standard monomials")
        return "NOT_DOMINANT", None

    standard_monomials = _standard_monomials(leading_exponents)
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
    "validate_generic_fiber_certificate",
]
