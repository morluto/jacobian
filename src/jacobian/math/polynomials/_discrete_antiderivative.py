"""Selected-variable rational discrete antiderivatives."""

from fractions import Fraction
from math import comb, gcd, lgamma, log

from pydantic import Field

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

# The current deterministic pure-Python triangular kernel performs one exact
# rational update per charged unit. Keep the admitted envelope below the
# sparse carrier's maximum so a valid but pathological high-degree source
# cannot monopolize a native call while still retaining ordinary low-degree
# polynomial sums.
MAX_DISCRETE_ANTIDERIVATIVE_WORK = 1_000_000


class RationalDiscreteAntiderivativeRequest(StrictModel):
    """Bind one polynomial axis for the zero-based finite-difference inverse."""

    polynomial: RationalPolynomial = Field(
        description=(
            "Canonical sparse polynomial over QQ. Its declared variable axis is "
            "retained unchanged in both returned polynomials. The current kernel "
            f"admits at most {MAX_DISCRETE_ANTIDERIVATIVE_WORK} exact rational "
            "updates for the triangular solve and reconstruction."
        )
    )
    variable: PolynomialVariable = Field(
        description=(
            "Declared polynomial axis along which Q(x+1)-Q(x)=P(x) is solved; "
            "all other axes remain coefficient parameters."
        )
    )


class RationalDiscreteAntiderivativeResult(StrictModel):
    """A normalized antiderivative and its exact finite-difference reconstruction."""

    source: RationalPolynomial
    variable: PolynomialVariable
    antiderivative: RationalPolynomial
    reconstructed_difference: RationalPolynomial


def _parse_native_polynomial(source: RationalPolynomial) -> RationalPolynomial:
    """Reparse nested values for native callers bypassing Pydantic validation."""

    try:
        payload = source.model_dump(mode="python", warnings=False)
        return RationalPolynomial.model_validate(payload)
    except (AttributeError, RecursionError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.polynomial_structure",
            message="polynomial contains malformed canonical nested values",
        ) from exc


def _decimal_digits_upper(value: int) -> int:
    """Return a cheap upper bound on the decimal digits of a nonnegative int."""

    if value == 0:
        return 1
    # log10(2) < 30103 / 100000; the extra two digits make this bound safe
    # without converting a caller-sized integer to a decimal string.
    return (value.bit_length() * 30103) // 100000 + 2


def _factorial_digits_upper(degree: int) -> int:
    """Bound decimal digits of ``(degree + 1)!`` without materializing it."""

    return max(1, int(lgamma(degree + 2) / log(10)) + 2)


def _exceeds_canonical_digits(value: Fraction) -> bool:
    limit = 10**MAX_CANONICAL_RATIONAL_DIGITS
    return abs(value.numerator) >= limit or value.denominator >= limit


def _admit_closed_form_coefficients(values: tuple[Fraction, ...]) -> None:
    if any(_exceeds_canonical_digits(value) for value in values):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.intermediate_growth",
            message=(
                "discrete-antiderivative intermediate coefficients exceed the "
                "exact-output bound"
            ),
        )


def _admit_linear_group(coefficients: dict[int, Fraction]) -> None:
    """Admit degree-1 slices from the exact closed-form inverse.

    For ``P(x) = a x + b`` the zero-based inverse is ``Q(x) = (a/2) x^2 +
    (b - a/2) x``.  The triangular factorial/binomial envelope overcounts this
    by several digits, so a source coefficient already at the 32768-digit
    carrier limit would be refused even when ``Q`` stays representable.
    """

    linear = coefficients.get(1, Fraction())
    constant = coefficients.get(0, Fraction())
    quadratic = linear / 2
    linear_term = constant - quadratic
    _admit_closed_form_coefficients((quadratic, linear_term, linear))


def _admit_quadratic_group(coefficients: dict[int, Fraction]) -> None:
    """Admit degree-2 slices from the exact closed-form inverse.

    For ``P(x) = a x^2 + b x + c`` the zero-based inverse is
    ``Q(x) = (a/3) x^3 + ((b-a)/2) x^2 + (a/6 - b/2 + c) x``.  Every
    denominator is at most 6, so a source coefficient already at the digit
    limit stays representable even when the triangular envelope does not.
    """

    quadratic = coefficients.get(2, Fraction())
    linear = coefficients.get(1, Fraction())
    constant = coefficients.get(0, Fraction())
    cubic_term = quadratic / 3
    quadratic_term = (linear - quadratic) / 2
    linear_term = quadratic / 6 - linear / 2 + constant
    _admit_closed_form_coefficients(
        (cubic_term, quadratic_term, linear_term, quadratic, linear)
    )


def _admit_cubic_group(coefficients: dict[int, Fraction]) -> None:
    """Admit degree-3 slices from the exact closed-form inverse.

    For ``P(x) = a x^3`` the zero-based inverse is
    ``Q(x) = a (x^4 - 2 x^3 + x^2) / 4``.  Superposition with the quadratic
    inverse keeps every denominator at most 12, so a source coefficient at
    the digit limit stays representable.
    """

    cubic = coefficients.get(3, Fraction())
    quadratic = coefficients.get(2, Fraction())
    linear = coefficients.get(1, Fraction())
    constant = coefficients.get(0, Fraction())
    quartic_term = cubic / 4
    cubic_term = -cubic / 2 + quadratic / 3
    quadratic_term = cubic / 4 - quadratic / 2 + linear / 2
    linear_term = quadratic / 6 - linear / 2 + constant
    _admit_closed_form_coefficients(
        (
            quartic_term,
            cubic_term,
            quadratic_term,
            linear_term,
            cubic,
            quadratic,
            linear,
        )
    )


def _admit_group(
    coefficients: dict[int, Fraction],
    *,
    maximum_degree: int,
) -> None:
    """Preflight one coefficient-parameter slice before exact expansion.

    The triangular solve uses only additions, integer binomial factors, and
    divisions by integers through ``maximum_degree + 1``.  Clearing all input
    denominators and the factorial envelope gives a conservative bound for
    every residual, antiderivative coefficient, and reconstructed coefficient.
    """

    if maximum_degree == 0:
        return
    if maximum_degree == 1:
        _admit_linear_group(coefficients)
        return
    if maximum_degree == 2:
        _admit_quadratic_group(coefficients)
        return
    if maximum_degree == 3:
        _admit_cubic_group(coefficients)
        return

    common_denominator = 1
    maximum_numerator_digits = 1
    for coefficient in coefficients.values():
        maximum_numerator_digits = max(
            maximum_numerator_digits,
            _decimal_digits_upper(abs(coefficient.numerator)),
        )
        denominator = coefficient.denominator
        factor = denominator // gcd(common_denominator, denominator)
        common_digits = _decimal_digits_upper(common_denominator)
        factor_digits = _decimal_digits_upper(factor)
        if common_digits + factor_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationResourceAdmissionError(
                location=("polynomial",),
                code="polynomial.discrete_antiderivative.intermediate_growth",
                message=(
                    "common coefficient denominators exceed the "
                    "exact-intermediate bound"
                ),
            )
        common_denominator *= factor

    common_denominator_digits = _decimal_digits_upper(common_denominator)
    factorial_digits = _factorial_digits_upper(maximum_degree)
    binomial_digits = (maximum_degree + 1) * 30103 // 100000 + 2
    term_count_digits = _decimal_digits_upper(len(coefficients))
    denominator_digits = common_denominator_digits + factorial_digits
    coefficient_digits = (
        maximum_numerator_digits
        + common_denominator_digits
        + factorial_digits
        + (2 * binomial_digits)
        + term_count_digits
    )
    if max(denominator_digits, coefficient_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.intermediate_growth",
            message=(
                "discrete-antiderivative intermediate coefficients exceed the "
                "exact-output bound"
            ),
        )


def _admit_coefficients(coefficients: dict[tuple[int, ...], Fraction]) -> None:
    limit = 10**MAX_CANONICAL_RATIONAL_DIGITS
    if any(
        abs(value.numerator) >= limit or value.denominator >= limit
        for value in coefficients.values()
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.coefficient_growth",
            message="discrete-antiderivative coefficients exceed the exact-output bound",
        )


def _polynomial(
    variables: tuple[str, ...], coefficients: dict[tuple[int, ...], Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(coefficients.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _checkpoint_work(charged: int, stage: str) -> int:
    charged += 1
    if charged % 64 == 0:
        request_checkpoint(stage)
    return charged


def _solve_slices(
    groups: dict[tuple[int, ...], dict[int, Fraction]],
    variable_index: int,
) -> tuple[dict[tuple[int, ...], Fraction], int]:
    answer: dict[tuple[int, ...], Fraction] = {}
    charged = 0
    request_checkpoint("during discrete antiderivative solve")
    for other, coefficients in groups.items():
        residual = dict(coefficients)
        for degree in range(max(residual, default=-1), -1, -1):
            charged = _checkpoint_work(charged, "during discrete antiderivative solve")
            leading = residual.get(degree, Fraction())
            if not leading:
                continue
            antiderivative_coefficient = leading / (degree + 1)
            exponent = (
                *other[:variable_index],
                degree + 1,
                *other[variable_index:],
            )
            answer[exponent] = antiderivative_coefficient
            for lower_degree in range(degree + 1):
                charged = _checkpoint_work(
                    charged, "during discrete antiderivative solve"
                )
                residual[lower_degree] = residual.get(
                    lower_degree, Fraction()
                ) - antiderivative_coefficient * comb(degree + 1, lower_degree)
    return answer, charged


def _reconstruct_difference(
    answer: dict[tuple[int, ...], Fraction],
    variable_index: int,
    charged: int,
) -> dict[tuple[int, ...], Fraction]:
    reconstructed: dict[tuple[int, ...], Fraction] = {}
    for exponents, coefficient in answer.items():
        degree = exponents[variable_index]
        for lower_degree in range(degree):
            charged = _checkpoint_work(
                charged, "during discrete antiderivative reconstruction"
            )
            target = list(exponents)
            target[variable_index] = lower_degree
            key = tuple(target)
            reconstructed[key] = reconstructed.get(
                key, Fraction()
            ) + coefficient * comb(degree, lower_degree)
    return reconstructed


def _compute_discrete_antiderivative(
    source: RationalPolynomial,
    variable: PolynomialVariable,
) -> RationalDiscreteAntiderivativeResult:
    if not isinstance(source, RationalPolynomial):
        raise OperationDomainValidationError(
            location=(),
            code="polynomial.discrete_antiderivative.polynomial_type",
            message="polynomial must be a RationalPolynomial",
        )
    source = _parse_native_polynomial(source)
    if type(variable) is not str or not variable.isidentifier():
        raise OperationDomainValidationError(
            location=("variable",),
            code="polynomial.discrete_antiderivative.variable_type",
            message="variable must be a valid polynomial axis name",
        )
    if variable not in source.variables:
        raise OperationDomainValidationError(
            location=("variable",),
            code="polynomial.discrete_antiderivative.variable_axis",
            message="selected variable must belong to the polynomial axis",
        )
    variable_index = source.variables.index(variable)
    output_bound = sum(
        term.exponents[variable_index] + 1 for term in source.polynomial.terms
    )
    maximum_degree = max(
        (term.exponents[variable_index] for term in source.polynomial.terms),
        default=0,
    )
    if output_bound > MAX_POLYNOMIAL_TERMS or maximum_degree >= MAX_POLYNOMIAL_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.output_bound",
            message="discrete antiderivative exceeds the sparse support or exponent bound",
        )
    groups: dict[tuple[int, ...], dict[int, Fraction]] = {}
    for term in source.polynomial.terms:
        other = (
            *term.exponents[:variable_index],
            *term.exponents[variable_index + 1 :],
        )
        groups.setdefault(other, {})[term.exponents[variable_index]] = (
            term.coefficient.as_fraction()
        )
    work = sum(
        (degree + 1) * (degree + 2) // 2 + (degree + 1) ** 2
        for coefficients in groups.values()
        for degree in (max(coefficients, default=-1),)
    )
    if work > MAX_DISCRETE_ANTIDERIVATIVE_WORK:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.discrete_antiderivative.work_bound",
            message="discrete-antiderivative work exceeds the admitted bound",
        )
    for coefficients in groups.values():
        _admit_group(
            coefficients,
            maximum_degree=max(coefficients, default=0),
        )
    answer, charged = _solve_slices(groups, variable_index)
    reconstructed = _reconstruct_difference(answer, variable_index, charged)
    request_checkpoint("after discrete antiderivative computation")
    _admit_coefficients(answer)
    _admit_coefficients(reconstructed)
    antiderivative = _polynomial(source.variables, answer)
    difference = _polynomial(source.variables, reconstructed)
    if difference != source:
        raise RuntimeError("discrete antiderivative reconstruction failed")
    return RationalDiscreteAntiderivativeResult(
        source=source,
        variable=variable,
        antiderivative=antiderivative,
        reconstructed_difference=difference,
    )


def rational_discrete_antiderivative(
    polynomial: RationalPolynomial,
    variable: PolynomialVariable,
) -> RationalDiscreteAntiderivativeResult:
    """Compute the normalized finite-difference inverse on one polynomial axis."""

    return _compute_discrete_antiderivative(polynomial, variable)


__all__ = ["rational_discrete_antiderivative"]
