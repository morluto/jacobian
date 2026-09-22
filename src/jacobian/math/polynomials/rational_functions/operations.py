"""Exact native operations on canonical rational functions."""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionResult,
    PartialFractionsResult,
    PartialFractionTerm,
    RationalFunctionFactorPower,
    require_hermite_reduction_budget,
    require_partial_fraction_budget,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

from .structured_operations import (
    formal_antiderivative,
    global_residues,
    logarithmic_differential,
    rational_primitive,
    residue_at_infinity,
)


def _hermite_parts(function: RationalFunction) -> tuple[Any, Any]:
    """Return the zero-constant rational part and square-free remainder."""

    from sympy import Poly, cancel, fraction
    from sympy.integrals.rationaltools import ratint_ratpart

    (variable,) = symbols_for_variables(function.variables)
    source = cancel(rational_function_to_sympy(function))
    numerator_expression, denominator_expression = fraction(source)
    numerator = Poly(numerator_expression, variable, domain="QQ")
    denominator = Poly(denominator_expression, variable, domain="QQ")
    polynomial_part, proper_numerator = numerator.div(denominator)
    rational_part = polynomial_part.integrate().as_expr()

    if proper_numerator.is_zero:
        remainder = 0
    else:
        repeated_pole_part, remainder = ratint_ratpart(
            proper_numerator,
            denominator,
            variable,
        )
        rational_part += repeated_pole_part

    rational_part = cancel(rational_part)
    remainder = cancel(remainder)
    return rational_part, remainder


def hermite_reduction(
    function: RationalFunction,
) -> tuple[RationalFunction, RationalFunction]:
    """Compute canonical ``f = R' + H`` over the admitted subset of ``QQ(x)``.

    The current native envelope matches the catalog operation: numerator degree
    at most 6 and denominator degree at most 3 for nonpolynomial sources.
    Polynomial inputs allow degree 63 and 128-digit rational components when
    primitive denominators fit the same carrier. ``H`` is proper with square-free
    denominator. ``R`` has zero additive constant, so the pair is unique.
    """

    try:
        require_hermite_reduction_budget(function)
    except OperationResourceAdmissionError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.rational_function_admission", message=str(exc)
        ) from exc
    return _hermite_reduction_admitted(function)


def _hermite_reduction_admitted(
    function: RationalFunction,
) -> tuple[RationalFunction, RationalFunction]:
    """Compute after the shared owner admission has succeeded."""

    if all(not any(term.exponents) for term in function.denominator.terms):
        primitive = SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(
                        term.coefficient.as_fraction() / (term.exponents[0] + 1)
                    ),
                    exponents=(term.exponents[0] + 1,),
                )
                for term in function.numerator.terms
            )
        )
        return (
            RationalFunction(
                variables=function.variables,
                numerator=primitive,
                denominator=function.denominator,
            ),
            RationalFunction(
                variables=function.variables,
                numerator=SparseRationalPolynomial(),
                denominator=function.denominator,
            ),
        )
    rational_part, remainder = _hermite_parts(function)
    return (
        rational_function_from_sympy(rational_part, function.variables),
        rational_function_from_sympy(remainder, function.variables),
    )


def verify_hermite_reduction(claim: HermiteReductionResult) -> bool:
    """Verify the canonical Hermite decomposition against its source function."""
    try:
        rational_part, remainder = hermite_reduction(claim.function)
        has_primitive = not remainder.numerator.terms
        return (
            rational_part == claim.rational_part
            and remainder == claim.remainder
            and claim.rational_primitive_status
            == ("RATIONAL_PRIMITIVE" if has_primitive else "NO_RATIONAL_PRIMITIVE")
            and claim.rational_primitive == (rational_part if has_primitive else None)
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def _run_partial_fraction_admission(function: RationalFunction) -> None:
    try:
        require_partial_fraction_budget(function)
    except OperationResourceAdmissionError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.rational_function_admission", message=str(exc)
        ) from exc


def _sum_partial_fraction_terms(
    terms: tuple[PartialFractionTerm, ...],
    variables: tuple[str, ...],
) -> RationalFunction:
    """Replay a sum of exact ``numerator / factor**exponent`` terms."""

    from sympy import cancel

    expression = 0
    for term in terms:
        numerator = rational_polynomial_to_sympy(term.numerator).as_expr()
        factor = rational_polynomial_to_sympy(term.factor).as_expr()
        expression += numerator / factor**term.exponent
    return rational_function_from_sympy(cancel(expression), variables)


def _partial_fractions_admitted(function: RationalFunction) -> PartialFractionsResult:
    """Compute the factor-power profile after shared owner admission."""

    from jacobian.math.polynomials._elementary_kernel import (
        rational_partial_fraction_decomposition,
    )
    from jacobian.math.polynomials.operations import polynomial_factorization

    variables = function.variables
    numerator = RationalPolynomial(variables=variables, polynomial=function.numerator)
    denominator = RationalPolynomial(
        variables=variables, polynomial=function.denominator
    )
    decomposition = rational_partial_fraction_decomposition(numerator, denominator)
    factorization = polynomial_factorization(denominator)
    factors = tuple(
        RationalFunctionFactorPower(factor=record.factor, exponent=record.multiplicity)
        for record in factorization.factors
    )
    terms = tuple(
        PartialFractionTerm(
            factor=term.denominator_factor,
            exponent=term.denominator_exponent,
            numerator=term.numerator,
        )
        for term in decomposition.terms
    )
    reconstructed = RationalFunction._from_kernel(
        variables=variables,
        numerator=decomposition.reconstruction_numerator.polynomial,
        denominator=decomposition.reconstruction_denominator.polynomial,
    )
    if reconstructed != function:
        raise RuntimeError("partial-fraction replay did not reconstruct the source")

    _, hermite_remainder = hermite_reduction(function)
    simple_pole_part = _sum_partial_fraction_terms(
        tuple(term for term in terms if term.exponent == 1), variables
    )
    hermite_agreement = simple_pole_part == hermite_remainder
    if not hermite_agreement:
        raise RuntimeError("partial fractions disagree with Hermite reduction")

    return PartialFractionsResult._from_kernel(
        function=function,
        polynomial_part=decomposition.polynomial_part,
        factors=factors,
        terms=terms,
        reconstructed=reconstructed,
        hermite_remainder=hermite_remainder,
        hermite_agreement=hermite_agreement,
    )


def partial_fractions(function: RationalFunction) -> PartialFractionsResult:
    """Compute the canonical partial-fraction profile of ``f`` in ``QQ(x)``.

    Return the unique reduced ``f = A + sum n_(i,k)/g_i^k`` decomposition over
    ``QQ`` with the monic irreducible factor-power profile of the reduced
    denominator, a common-denominator replay, and the independently computed
    Hermite remainder. ``g_i`` are monic irreducible over ``QQ`` and every
    ``n_(i,k)`` has degree below ``g_i``. This does not split irreducibles into
    algebraic roots.
    """

    _run_partial_fraction_admission(function)
    return _partial_fractions_admitted(function)


def verify_partial_fractions(claim: PartialFractionsResult) -> bool:
    """Verify a partial-fraction profile against its source function."""

    try:
        return partial_fractions(claim.function) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


__all__ = [
    "formal_antiderivative",
    "global_residues",
    "hermite_reduction",
    "logarithmic_differential",
    "partial_fractions",
    "rational_primitive",
    "residue_at_infinity",
    "verify_hermite_reduction",
    "verify_partial_fractions",
]
