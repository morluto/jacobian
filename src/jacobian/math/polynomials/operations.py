"""Exact polynomial operations on canonical values and SymPy kernels."""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._bezout_kernel import (
    MAX_BEZOUT_SOURCE_COEFFICIENT_DIGITS,
    bounded_bezout,
    require_bezout_source_shape,
)
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._gcd_verification import verify_gcd_relation
from jacobian.math.polynomials._invariant_admission import (
    InvariantPlan,
    discriminant_plan,
    require_invariant_source,
    resultant_plan,
)
from jacobian.math.polynomials._models import (
    _MAX_DISCRIMINANT_DEGREE,
    _MAX_ELIMINATION_DEGREE_SUM,
    _MAX_GCD_DEGREE,
    _MAX_GCD_TERMS,
    _MAX_GROEBNER_COEFFICIENT_DIGITS,
    _MAX_GROEBNER_EXPONENT,
    _MAX_INVARIANT_TERMS,
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
    _irreducible_factor_sort_key,
    _validation_error,
)
from jacobian.math.polynomials._multiply_kernel import (
    rational_polynomial_multiply as multiply,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

if TYPE_CHECKING:
    from sympy import Poly

    from jacobian.math.polynomials._square_free import Plan as SquareFreePlan

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
    "verify_polynomial_discriminant",
    "verify_polynomial_factorization",
    "verify_polynomial_gcd",
    "verify_polynomial_resultant",
    "verify_polynomial_square_free_decomposition",
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
    require_bezout_source_shape(left)
    require_bezout_source_shape(right)
    if left.variables != right.variables:
        raise _validation_error("polynomials must use the same ordered variables")
    if len(left.variables) != 1:
        raise _validation_error("Bézout GCD currently supports one variable over QQ")
    for polynomial in (left, right):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
            maximum_coefficient_digits=MAX_BEZOUT_SOURCE_COEFFICIENT_DIGITS,
        )
    if not left.polynomial.terms and not right.polynomial.terms:
        raise _validation_error(
            "gcd(0, 0) is undefined: zero has no monic normalization"
        )


def _admit_resultant(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_variable: str,
) -> InvariantPlan | None:
    require_invariant_source(left)
    require_invariant_source(right)
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

    return None if univariate else resultant_plan(left, right, index)


def _admit_discriminant(
    polynomial: RationalPolynomial, variable: str
) -> InvariantPlan | None:
    require_invariant_source(polynomial)
    if variable not in polynomial.variables:
        raise _validation_error(
            "discriminant variable must belong to the declared ring"
        )
    univariate = len(polynomial.variables) == 1
    maximum_exponent = (
        _MAX_UNIVARIATE_INVARIANT_DEGREE_SUM if univariate else _MAX_DISCRIMINANT_DEGREE
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

    return (
        None
        if univariate
        else discriminant_plan(polynomial, polynomial.variables.index(variable))
    )


def _coefficient_bounds(polynomial: RationalPolynomial) -> tuple[int, int]:
    terms = polynomial.polynomial.terms
    denominators = {term.coefficient.den for term in terms}
    common_denominator = 1
    for denominator in denominators:
        factor = denominator // gcd(common_denominator, denominator)
        # Bound the allocation before multiplying. If clearing would exceed
        # the scalar envelope, retain a conservative product bound instead.
        if (
            common_denominator.bit_length() + factor.bit_length()
            > 4 * MAX_CANONICAL_RATIONAL_DIGITS
        ):
            denominator_digits = sum(
                len(format_canonical_integer(d)) for d in denominators if d != 1
            )
            return denominator_digits, max(
                (
                    len(format_canonical_integer(abs(t.coefficient.num)))
                    + denominator_digits
                    for t in terms
                ),
                default=1,
            )
        common_denominator *= factor
    denominator_digits = (
        0
        if common_denominator == 1
        else len(format_canonical_integer(common_denominator))
    )
    cleared_height_digits = max(
        (
            len(format_canonical_integer(abs(term.coefficient.num)))
            + (
                0
                if common_denominator == term.coefficient.den
                else len(
                    format_canonical_integer(common_denominator // term.coefficient.den)
                )
            )
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


def _admit_square_free(polynomial: RationalPolynomial) -> SquareFreePlan:
    from jacobian.math.polynomials._square_free import admit

    return admit(polynomial)


def _lift_square_free_polynomial(
    polynomial: RationalPolynomial, dilation: int
) -> RationalPolynomial:
    if dilation == 1:
        return polynomial
    return RationalPolynomial(
        variables=polynomial.variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=term.coefficient,
                    exponents=(term.exponents[0] * dilation,),
                )
                for term in polynomial.polynomial.terms
            )
        ),
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
    left_multiplier, right_multiplier, common = bounded_bezout(left, right)
    return PolynomialGcdResult(
        left=left,
        right=right,
        gcd=common,
        bezout=PolynomialBezoutIdentity(
            left_multiplier=left_multiplier,
            right_multiplier=right_multiplier,
        ),
    )


def polynomial_resultant(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_variable: str,
) -> PolynomialResultantResult:
    """Compute the exact resultant in one declared canonical ring variable."""

    plan = _run_admission(lambda: _admit_resultant(left, right, elimination_variable))
    variables = left.variables
    if len(variables) == 1:
        left_flint = _flint_univariate(left)
        right_flint = _flint_univariate(right)
        value = left_flint.resultant(right_flint)
        return PolynomialResultantResult(
            left=left,
            right=right,
            elimination_variable=elimination_variable,
            resultant=PolynomialScalarValue(
                value=_canonical_rational_from_flint(value)
            ),
        )
    if plan is None:
        raise RuntimeError("multivariate invariant admission did not provide a plan")
    return PolynomialResultantResult(
        left=left,
        right=right,
        elimination_variable=elimination_variable,
        resultant=PolynomialValue(value=plan.execute()),
    )


def polynomial_discriminant(
    polynomial: RationalPolynomial, variable: str
) -> PolynomialDiscriminantResult:
    """Compute the exact discriminant in one canonical ring variable."""

    plan = _run_admission(lambda: _admit_discriminant(polynomial, variable))
    variables = polynomial.variables
    if len(variables) == 1:
        flint_polynomial = _flint_univariate(polynomial)
        if flint_polynomial.is_zero():
            return PolynomialDiscriminantResult(
                polynomial=polynomial,
                variable=variable,
                discriminant=PolynomialScalarValue(
                    value=CanonicalRational.from_integer_ratio(0, 1)
                ),
            )
        value = flint_polynomial.discriminant()
        return PolynomialDiscriminantResult(
            polynomial=polynomial,
            variable=variable,
            discriminant=PolynomialScalarValue(
                value=_canonical_rational_from_flint(value)
            ),
        )
    if plan is None:
        raise RuntimeError("multivariate invariant admission did not provide a plan")
    return PolynomialDiscriminantResult(
        polynomial=polynomial,
        variable=variable,
        discriminant=PolynomialValue(value=plan.execute()),
    )


def polynomial_square_free_decomposition(
    polynomial: RationalPolynomial,
) -> PolynomialSquareFreeDecompositionResult:
    """Compute the canonical square-free decomposition of a polynomial."""

    admission = _run_admission(lambda: _admit_square_free(polynomial))
    if admission.kind != "compact":
        from jacobian.math.polynomials._square_free import compute

        return compute(admission)
    dilation = admission.dilation
    backend_source = polynomial
    if dilation != 1:
        leading, constant = polynomial.polynomial.terms
        backend_source = RationalPolynomial(
            variables=polynomial.variables,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=leading.coefficient, exponents=(1,)
                    ),
                    constant,
                )
            ),
        )
    source = rational_polynomial_to_sympy(backend_source)
    coefficient, canonical_factors, reconstructed = square_free_decomposition(source)
    factors = tuple(
        PolynomialSquareFreeFactor(
            factor=_lift_square_free_polynomial(
                _result_polynomial(factor, polynomial.variables), dilation
            ),
            multiplicity=multiplicity,
        )
        for factor, multiplicity in sorted(canonical_factors, key=lambda item: item[1])
    )
    lifted_reconstruction = _lift_square_free_polynomial(
        _result_polynomial(reconstructed, polynomial.variables), dilation
    )
    if lifted_reconstruction != polynomial:
        raise RuntimeError("square-free exponent lifting did not reconstruct input")
    return PolynomialSquareFreeDecompositionResult._from_kernel(
        polynomial=polynomial,
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=lifted_reconstruction,
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


def verify_polynomial_gcd(claim: PolynomialGcdResult) -> bool:
    """Verify a monic GCD and any exact Bézout identity for retained operands.

    Resource refusal during bounded candidate replay propagates rather than
    returning a mathematical negative.
    """

    return verify_gcd_relation(claim)


def verify_polynomial_resultant(claim: PolynomialResultantResult) -> bool:
    """Verify a resultant against its retained operands and elimination variable."""

    if not isinstance(claim, PolynomialResultantResult):
        return False
    try:
        return (
            polynomial_resultant(claim.left, claim.right, claim.elimination_variable)
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_polynomial_discriminant(claim: PolynomialDiscriminantResult) -> bool:
    """Verify a discriminant against its retained polynomial and variable."""

    if not isinstance(claim, PolynomialDiscriminantResult):
        return False
    try:
        return polynomial_discriminant(claim.polynomial, claim.variable) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_polynomial_square_free_decomposition(
    claim: PolynomialSquareFreeDecompositionResult,
) -> bool:
    """Verify square-free factors and reconstruction against their source."""

    if not isinstance(claim, PolynomialSquareFreeDecompositionResult):
        return False
    try:
        return polynomial_square_free_decomposition(claim.polynomial) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_polynomial_factorization(claim: PolynomialFactorizationResult) -> bool:
    """Verify factorization claims against their retained source polynomial."""

    if not isinstance(claim, PolynomialFactorizationResult):
        return False
    try:
        return polynomial_factorization(claim.polynomial) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


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
