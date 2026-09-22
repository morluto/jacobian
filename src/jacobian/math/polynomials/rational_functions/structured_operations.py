"""Exact structured dlog and residue transforms."""

from __future__ import annotations

from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction

from ._models import require_partial_fraction_budget
from .structured_models import (
    FormalAntiderivativeResult,
    GlobalResidueResult,
    LogarithmicDifferentialResult,
    LogarithmicDifferentialTerm,
    RationalPrimitiveResult,
    ResidueRow,
)


def _admit(f: RationalFunction) -> None:
    if not isinstance(f, RationalFunction):
        raise OperationDomainValidationError(
            location=("function",),
            code="polynomial.rational_function_type",
            message="function must be a canonical rational-function value",
        )
    try:
        require_partial_fraction_budget(f)
    except OperationResourceAdmissionError:
        raise
    except (ValueError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("function",),
            code="polynomial.rational_function_admission",
            message=str(exc),
        ) from exc


def logarithmic_differential(
    function: RationalFunction,
) -> LogarithmicDifferentialResult:
    from .operations import hermite_reduction, partial_fractions

    _admit(function)
    _, remainder = hermite_reduction(function)
    if not remainder.numerator.terms:
        return LogarithmicDifferentialResult(
            source=function,
            terms=(),
            reconstructed=rational_function_from_sympy(0, function.variables),
        )
    profile = partial_fractions(remainder)
    terms = tuple(
        LogarithmicDifferentialTerm(factor=t.factor, numerator=t.numerator)
        for t in profile.terms
        if t.exponent == 1
    )
    expression = sum(
        (
            rational_polynomial_to_sympy(t.numerator).as_expr()
            / rational_polynomial_to_sympy(t.factor).as_expr()
            for t in terms
        ),
        0,
    )
    reconstructed = rational_function_from_sympy(expression, function.variables)
    if reconstructed != remainder:
        raise RuntimeError("structured logarithmic differential failed reconstruction")
    return LogarithmicDifferentialResult(
        source=remainder, terms=terms, reconstructed=reconstructed
    )


def formal_antiderivative(
    function: RationalFunction,
) -> FormalAntiderivativeResult:
    _admit(function)
    from .operations import hermite_reduction

    rational_part, remainder = hermite_reduction(function)
    log_part = logarithmic_differential(remainder)
    return FormalAntiderivativeResult(
        source=function, rational_part=rational_part, logarithmic_part=log_part
    )


def rational_primitive(function: RationalFunction) -> RationalPrimitiveResult:
    from .operations import hermite_reduction

    _admit(function)
    rational_part, remainder = hermite_reduction(function)
    return RationalPrimitiveResult(
        source=function,
        status="RATIONAL_PRIMITIVE"
        if not remainder.numerator.terms
        else "NO_RATIONAL_PRIMITIVE",
        rational_part=rational_part,
        remainder=remainder,
    )


def residue_at_infinity(function: RationalFunction) -> CanonicalRational:
    _admit(function)
    import sympy as sp

    (x,) = symbols_for_variables(function.variables)
    t = sp.symbols("t")
    value = -sp.residue(
        rational_function_to_sympy(function).subs(x, 1 / t) / t**2, t, 0
    )
    return CanonicalRational.from_fraction(
        Fraction(int(sp.numer(value)), int(sp.denom(value)))
    )


def global_residues(function: RationalFunction) -> GlobalResidueResult:
    _admit(function)
    import sympy as sp

    (x,) = symbols_for_variables(function.variables)
    expression = sp.cancel(rational_function_to_sympy(function))
    num, den = sp.fraction(expression)
    _, factors = sp.factor_list(den, x)
    rows = []
    finite_sum = Fraction(0)
    for factor, multiplicity in factors:
        poly = sp.Poly(factor, x, domain="QQ")
        fp = rational_polynomial_from_sympy(poly, function.variables)
        derivative = rational_polynomial_from_sympy(poly.diff(), function.variables)
        numerator_poly = sp.Poly(num, x, domain="QQ")
        np = rational_polynomial_from_sympy(
            numerator_poly.rem(poly), function.variables
        )
        for root_index in range(poly.degree()):
            rows.append(
                ResidueRow(
                    factor=fp,
                    root_index=root_index,
                    pole_order=int(multiplicity),
                    numerator_at_root=np,
                    derivative_factor=derivative,
                )
            )
        # Sum of all residues belonging to a factor is rational and can be
        # obtained from the exact global identity; retain no approximations.
    inf = residue_at_infinity(function).as_fraction()
    # finite residue sum is -Res_infinity, an exact identity of rational forms.
    finite_sum = -inf
    return GlobalResidueResult(
        source=function,
        finite_poles=tuple(rows),
        residue_at_infinity=CanonicalRational.from_fraction(inf),
        finite_residue_sum=CanonicalRational.from_fraction(finite_sum),
        total_residue=CanonicalRational(num=0, den=1),
    )


__all__ = [
    "formal_antiderivative",
    "global_residues",
    "logarithmic_differential",
    "rational_primitive",
    "residue_at_infinity",
]
