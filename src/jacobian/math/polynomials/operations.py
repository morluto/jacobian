"""Exact polynomial operations on canonical values and SymPy kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._models import (
    _MAX_DISCRIMINANT_DEGREE,
    _MAX_ELIMINATION_DEGREE_SUM,
    _MAX_GCD_DEGREE,
    _MAX_GCD_TERMS,
    _MAX_GROEBNER_COEFFICIENT_DIGITS,
    _MAX_GROEBNER_EXPONENT,
    _MAX_INVARIANT_TERMS,
    _MAX_SQUARE_FREE_EXPONENT,
    _MAX_UNIVARIATE_INVARIANT_DEGREE_SUM,
    MAX_GROEBNER_GENERATORS,
    PolynomialBezoutIdentity,
    PolynomialDiscriminantResult,
    PolynomialFactorizationResult,
    PolynomialGcdResult,
    PolynomialGroebnerBasisResult,
    PolynomialGroebnerBudget,
    PolynomialInvariantValue,
    PolynomialIrreducibleFactor,
    PolynomialResultantResult,
    PolynomialScalarValue,
    PolynomialSquareFreeDecompositionResult,
    PolynomialSquareFreeFactor,
    PolynomialValue,
    _degree,
    _validation_error,
)
from jacobian.math.polynomials._multiply_kernel import (
    rational_polynomial_multiply as multiply,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalFunction,
    RationalPolynomial,
    require_polynomial_budget,
)

if TYPE_CHECKING:
    from sympy import Poly

__all__ = [
    "derivative",
    "discriminant",
    "divide",
    "evaluate",
    "factorization",
    "gcdex",
    "groebner_basis",
    "hermite_reduction",
    "integral",
    "multiply",
    "partial_fractions",
    "polynomial_discriminant",
    "polynomial_factorization",
    "polynomial_gcd",
    "polynomial_groebner_basis",
    "polynomial_resultant",
    "polynomial_square_free_decomposition",
    "resultant",
    "square_free_decomposition",
]

MAX_OPERATION_OUTPUT_TERMS = 1_024


def _poly(value: Poly) -> Poly:
    from sympy import Poly

    if not isinstance(value, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    return value


def gcdex(left: Poly, right: Poly) -> tuple[Poly, Poly, Poly]:
    """Return the exact extended-GCD tuple for two compatible polynomials."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_gcdex(_poly(left), _poly(right))


def resultant(left: Poly, right: Poly, generator: Any) -> Any:
    """Return the exact resultant in the supplied common generator."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_resultant(_poly(left), _poly(right), generator)


def derivative(polynomial: Poly) -> Poly:
    """Return the formal derivative of a polynomial."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_derivative(_poly(polynomial))


def discriminant(polynomial: Poly, generator: Any) -> Any:
    """Return the discriminant in the supplied generator."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_discriminant(_poly(polynomial), generator)


def divide(left: Poly, right: Poly) -> tuple[Poly, Poly, Poly]:
    """Return quotient, remainder, and exact reconstruction."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_division(_poly(left), _poly(right))


def evaluate(polynomial: Poly, point: Any) -> Any:
    """Evaluate a polynomial at one exact backend-native point."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_evaluate(_poly(polynomial), point)


def factorization(source: Poly) -> tuple[Any, tuple[tuple[Poly, int], ...], Poly]:
    """Return coefficient, monic irreducible factors, and reconstruction."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_factorization(_poly(source))


def groebner_basis(
    generators: tuple[Poly, ...],
    variables: tuple[Any, ...],
    monomial_order: str,
) -> tuple[Poly, ...]:
    """Return a reduced Gröbner basis over ``QQ``."""

    from jacobian.math.polynomials import _sympy

    canonical_generators = tuple(_poly(generator) for generator in generators)
    if any(not generator.domain.is_QQ for generator in canonical_generators):
        raise ValueError("Gröbner basis generators must use the QQ domain")
    return _sympy.polynomial_groebner_basis(
        canonical_generators,
        variables,
        monomial_order,
    )


def integral(polynomial: Poly) -> Poly:
    """Return the formal antiderivative with zero constant term."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_integral(_poly(polynomial))


def hermite_reduction(
    function: RationalFunction,
) -> tuple[RationalFunction, RationalFunction]:
    """Reduce one admitted canonical rational function modulo derivatives."""

    from jacobian.math.polynomials.rational_functions.operations import (
        hermite_reduction as _hermite_reduction,
    )

    return _hermite_reduction(function)


def partial_fractions(expression: Any, generator: Any) -> Any:
    """Return an exact univariate partial-fraction decomposition."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_partial_fractions(expression, generator)


def square_free_decomposition(
    source: Poly,
) -> tuple[Any, tuple[tuple[Poly, int], ...], Poly]:
    """Return coefficient, monic square-free factors, and reconstruction."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_square_free_decomposition(_poly(source))


class PolynomialOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _run_admission[ResultT](admission: Callable[[], ResultT]) -> ResultT:
    """Expose owner admission as a typed native-domain failure."""

    try:
        return admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.admission", message=str(exc)
        ) from exc


def _admit_gcd(left: RationalPolynomial, right: RationalPolynomial) -> None:
    if left.variables != right.variables:
        raise _validation_error("polynomials must use the same ordered variables")
    if len(left.variables) != 1:
        raise _validation_error("Bézout GCD currently supports one variable over QQ")
    for polynomial in (left, right):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
        )
    if not left.polynomial.terms and not right.polynomial.terms:
        raise _validation_error(
            "gcd(0, 0) is undefined: zero has no monic normalization"
        )


def _admit_resultant(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_variable: str,
) -> None:
    if left.variables != right.variables:
        raise _validation_error("polynomials must use the same ordered variables")
    if elimination_variable not in left.variables:
        raise _validation_error("elimination variable must belong to the declared ring")
    univariate = len(left.variables) == 1
    maximum_exponent = (
        _MAX_UNIVARIATE_INVARIANT_DEGREE_SUM
        if univariate
        else _MAX_ELIMINATION_DEGREE_SUM
    )
    for polynomial in (left, right):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_INVARIANT_TERMS,
            maximum_exponent=maximum_exponent,
        )
    index = left.variables.index(elimination_variable)
    degree_sum = _degree(left, index) + _degree(right, index)
    degree_limit = (
        _MAX_UNIVARIATE_INVARIANT_DEGREE_SUM
        if univariate
        else _MAX_ELIMINATION_DEGREE_SUM
    )
    if degree_sum > degree_limit:
        raise _validation_error("Sylvester degree exceeds the resultant budget")
    if (
        univariate
        and _resultant_component_digit_bound(left, right)
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise _validation_error("resultant scalar exceeds the canonical digit budget")


def _admit_discriminant(polynomial: RationalPolynomial, variable: str) -> None:
    if variable not in polynomial.variables:
        raise _validation_error(
            "discriminant variable must belong to the declared ring"
        )
    univariate = len(polynomial.variables) == 1
    maximum_exponent = (
        _MAX_UNIVARIATE_INVARIANT_DEGREE_SUM
        if univariate
        else _MAX_SQUARE_FREE_EXPONENT
    )
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_INVARIANT_TERMS,
        maximum_exponent=maximum_exponent,
    )
    if not univariate and (
        _degree(polynomial, polynomial.variables.index(variable))
        > _MAX_DISCRIMINANT_DEGREE
    ):
        raise _validation_error("main-variable degree exceeds the discriminant budget")
    if (
        univariate
        and _discriminant_component_digit_bound(polynomial)
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise _validation_error(
            "discriminant scalar exceeds the canonical digit budget"
        )


def _coefficient_bounds(polynomial: RationalPolynomial) -> tuple[int, int]:
    terms = polynomial.polynomial.terms
    denominator_digits = sum(len(term.coefficient.den) for term in terms)
    cleared_height_digits = max(
        (
            len(term.coefficient.num.lstrip("-"))
            + denominator_digits
            - len(term.coefficient.den)
            for term in terms
        ),
        default=1,
    )
    return denominator_digits, cleared_height_digits


def _resultant_component_digit_bound(
    left: RationalPolynomial, right: RationalPolynomial
) -> int:
    left_degree = _degree(left, 0)
    right_degree = _degree(right, 0)
    left_denominator, left_height = _coefficient_bounds(left)
    right_denominator, right_height = _coefficient_bounds(right)
    left_norm_digits = left_height + len(str(len(left.polynomial.terms) or 1))
    right_norm_digits = right_height + len(str(len(right.polynomial.terms) or 1))
    numerator_digits = right_degree * left_norm_digits + left_degree * right_norm_digits
    denominator_digits = (
        right_degree * left_denominator + left_degree * right_denominator
    )
    return max(1, numerator_digits, denominator_digits)


def _discriminant_component_digit_bound(polynomial: RationalPolynomial) -> int:
    degree = _degree(polynomial, 0)
    denominator_digits, height_digits = _coefficient_bounds(polynomial)
    derivative_growth_digits = len(str(max(1, degree)))
    numerator_digits = max(1, 2 * degree - 1) * (
        height_digits
        + derivative_growth_digits
        + len(str(len(polynomial.polynomial.terms) or 1))
    )
    denominator_result_digits = max(0, 2 * degree - 2) * denominator_digits
    return max(1, numerator_digits, denominator_result_digits)


def _admit_square_free(polynomial: RationalPolynomial) -> None:
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_GCD_TERMS,
        maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
    )


def _admit_factorization(polynomial: RationalPolynomial) -> None:
    if len(polynomial.variables) != 1:
        raise _validation_error("factorization currently supports one variable over QQ")
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_GCD_TERMS,
        maximum_exponent=_MAX_GCD_DEGREE,
    )


def _admit_groebner(
    generators: tuple[RationalPolynomial, ...],
    monomial_order: str,
) -> None:
    if not generators or len(generators) > MAX_GROEBNER_GENERATORS:
        raise _validation_error("ideal generator count is outside the operation budget")
    if monomial_order not in {"lex", "grlex", "grevlex"}:
        raise _validation_error("monomial order must be lex, grlex, or grevlex")
    variables = generators[0].variables
    if any(generator.variables != variables for generator in generators):
        raise _validation_error("all ideal generators must use the same ordered ring")
    if (
        sum(len(generator.polynomial.terms) for generator in generators)
        > _MAX_INVARIANT_TERMS
    ):
        raise _validation_error(
            f"ideal generators exceed the {_MAX_INVARIANT_TERMS}-term aggregate budget"
        )
    for generator in generators:
        require_polynomial_budget(
            generator,
            maximum_terms=MAX_POLYNOMIAL_TERMS,
            maximum_exponent=_MAX_GROEBNER_EXPONENT,
            maximum_coefficient_digits=_MAX_GROEBNER_COEFFICIENT_DIGITS,
            label="ideal generator",
        )
        if any(
            sum(term.exponents) > _MAX_GROEBNER_EXPONENT
            for term in generator.polynomial.terms
        ):
            raise _validation_error(
                f"ideal generator exceeds total degree {_MAX_GROEBNER_EXPONENT}"
            )


def _result_polynomial(poly: object, variables: tuple[str, ...]) -> RationalPolynomial:
    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=MAX_OPERATION_OUTPUT_TERMS,
        )
    except ValueError as exc:
        if "term operation budget" in str(exc):
            raise PolynomialOutputBudgetError(str(exc)) from exc
        raise


def _invariant_value(
    expression: Any,
    remaining_variables: tuple[str, ...],
) -> PolynomialInvariantValue:
    from sympy import QQ, Poly

    if not remaining_variables:
        return PolynomialScalarValue(value=rational_from_sympy(expression))
    return PolynomialValue(
        value=_result_polynomial(
            Poly(expression, *symbols_for_variables(remaining_variables), domain=QQ),
            remaining_variables,
        )
    )


def _flint_univariate(polynomial: RationalPolynomial) -> Any:
    from flint import fmpq, fmpq_poly

    degree = _degree(polynomial, 0)
    coefficients = [fmpq(0)] * (degree + 1)
    for term in polynomial.polynomial.terms:
        numerator, denominator = term.coefficient.as_integer_ratio()
        coefficients[term.exponents[0]] = fmpq(numerator, denominator)
    return fmpq_poly(coefficients)


def _canonical_rational_from_flint(value: Any) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(int(value.p), int(value.q))


def polynomial_gcd(
    left: RationalPolynomial, right: RationalPolynomial
) -> PolynomialGcdResult:
    """Compute the monic GCD and Bézout identity of two canonical polynomials."""

    _run_admission(lambda: _admit_gcd(left, right))
    left_sympy = rational_polynomial_to_sympy(left)
    right_sympy = rational_polynomial_to_sympy(right)
    left_multiplier, right_multiplier, gcd = gcdex(left_sympy, right_sympy)
    variables = left.variables
    return PolynomialGcdResult(
        gcd=_result_polynomial(gcd, variables),
        bezout=PolynomialBezoutIdentity(
            left_multiplier=_result_polynomial(left_multiplier, variables),
            right_multiplier=_result_polynomial(right_multiplier, variables),
        ),
    )


def polynomial_resultant(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_variable: str,
) -> PolynomialResultantResult:
    """Compute the exact resultant in one declared canonical ring variable."""

    _run_admission(lambda: _admit_resultant(left, right, elimination_variable))
    variables = left.variables
    if len(variables) == 1:
        left_flint = _flint_univariate(left)
        right_flint = _flint_univariate(right)
        value = left_flint.resultant(right_flint)
        return PolynomialResultantResult(
            elimination_variable=elimination_variable,
            resultant=PolynomialScalarValue(
                value=_canonical_rational_from_flint(value)
            ),
        )
    elimination_index = variables.index(elimination_variable)
    generator = symbols_for_variables(variables)[elimination_index]
    value = resultant(
        rational_polynomial_to_sympy(left),
        rational_polynomial_to_sympy(right),
        generator,
    )
    remaining_variables = tuple(
        variable for variable in variables if variable != elimination_variable
    )
    return PolynomialResultantResult(
        elimination_variable=elimination_variable,
        resultant=_invariant_value(value, remaining_variables),
    )


def polynomial_discriminant(
    polynomial: RationalPolynomial, variable: str
) -> PolynomialDiscriminantResult:
    """Compute the exact discriminant in one canonical ring variable."""

    _run_admission(lambda: _admit_discriminant(polynomial, variable))
    variables = polynomial.variables
    if len(variables) == 1:
        value = _flint_univariate(polynomial).discriminant()
        return PolynomialDiscriminantResult(
            variable=variable,
            discriminant=PolynomialScalarValue(
                value=_canonical_rational_from_flint(value)
            ),
        )
    variable_index = variables.index(variable)
    generator = symbols_for_variables(variables)[variable_index]
    value = discriminant(rational_polynomial_to_sympy(polynomial), generator)
    remaining_variables = tuple(name for name in variables if name != variable)
    return PolynomialDiscriminantResult(
        variable=variable,
        discriminant=_invariant_value(value, remaining_variables),
    )


def polynomial_square_free_decomposition(
    polynomial: RationalPolynomial,
) -> PolynomialSquareFreeDecompositionResult:
    """Compute the canonical square-free decomposition of a polynomial."""

    _run_admission(lambda: _admit_square_free(polynomial))
    source = rational_polynomial_to_sympy(polynomial)
    coefficient, canonical_factors, reconstructed = square_free_decomposition(source)
    factors = tuple(
        PolynomialSquareFreeFactor(
            factor=_result_polynomial(factor, polynomial.variables),
            multiplicity=multiplicity,
        )
        for factor, multiplicity in sorted(canonical_factors, key=lambda item: item[1])
    )
    return PolynomialSquareFreeDecompositionResult._from_kernel(
        polynomial=polynomial,
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=_result_polynomial(reconstructed, polynomial.variables),
    )


def _irreducible_factor_sort_key(
    record: PolynomialIrreducibleFactor,
) -> tuple[int, int, tuple[tuple[tuple[int, ...], str, str], ...]]:
    return (
        record.multiplicity,
        max(
            (sum(term.exponents) for term in record.factor.polynomial.terms),
            default=0,
        ),
        tuple(
            (term.exponents, term.coefficient.num, term.coefficient.den)
            for term in record.factor.polynomial.terms
        ),
    )


def polynomial_factorization(
    polynomial: RationalPolynomial,
) -> PolynomialFactorizationResult:
    """Compute a canonical exact univariate factorization."""

    _run_admission(lambda: _admit_factorization(polynomial))
    source = rational_polynomial_to_sympy(polynomial)
    coefficient, canonical_factors, reconstructed = factorization(source)
    factors = tuple(
        sorted(
            (
                PolynomialIrreducibleFactor(
                    factor=_result_polynomial(factor, polynomial.variables),
                    multiplicity=multiplicity,
                )
                for factor, multiplicity in canonical_factors
            ),
            key=_irreducible_factor_sort_key,
        )
    )
    return PolynomialFactorizationResult._from_kernel(
        polynomial=polynomial,
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=_result_polynomial(reconstructed, polynomial.variables),
    )


def polynomial_groebner_basis(
    generators: tuple[RationalPolynomial, ...],
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex",
    resource_budget: PolynomialGroebnerBudget | None = None,
) -> PolynomialGroebnerBasisResult:
    """Compute one complete reduced basis inside the isolated worker."""

    budget = resource_budget or PolynomialGroebnerBudget()
    _run_admission(lambda: _admit_groebner(generators, monomial_order))

    variables = generators[0].variables
    basis_polynomials = tuple(
        _result_polynomial(polynomial, variables)
        for polynomial in groebner_basis(
            tuple(rational_polynomial_to_sympy(generator) for generator in generators),
            symbols_for_variables(variables),
            monomial_order,
        )
    )
    if len(basis_polynomials) > budget.maximum_basis_polynomials:
        raise PolynomialOutputBudgetError(
            "Gröbner basis exceeds the requested polynomial-count limit"
        )
    if (
        sum(len(polynomial.polynomial.terms) for polynomial in basis_polynomials)
        > budget.maximum_output_terms
    ):
        raise PolynomialOutputBudgetError(
            "Gröbner basis exceeds the requested aggregate term limit"
        )
    return PolynomialGroebnerBasisResult(
        variables=variables,
        monomial_order=monomial_order,
        basis=basis_polynomials,
    )
