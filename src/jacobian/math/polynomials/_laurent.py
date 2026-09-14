"""Exact sparse rational Laurent-polynomial multiplication."""

from fractions import Fraction

from pydantic import ValidationError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
)

# Aggregate coefficient-digit envelope for one serialized Laurent product,
# matching the canonical 10 MiB encoded-output limit.
MAX_LAURENT_RESULT_DIGITS = 10 * 1024 * 1024


class RationalLaurentMultiplyRequest(StrictModel):
    left: RationalLaurentPolynomial
    right: RationalLaurentPolynomial


def _fraction_component_digits(value: Fraction) -> int:
    if not value:
        return 1
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _monomial_coefficient_digits(
    left: RationalLaurentPolynomial, right: RationalLaurentPolynomial
) -> (
    tuple[
        int,
        RationalLaurentPolynomialTerm,
        RationalLaurentPolynomial,
        tuple[Fraction, ...],
    ]
    | None
):
    """Admit monomial scale products once, with cancellation checkpoints."""

    for monomial, other in ((left, right), (right, left)):
        if len(monomial.terms) != 1:
            continue
        factor = monomial.terms[0].coefficient.as_fraction()
        scaled: list[Fraction] = []
        height = 1
        aggregate_digits = 0
        overflow = MAX_CANONICAL_RATIONAL_DIGITS + 1
        for index, term in enumerate(other.terms):
            if index % 128 == 0:
                request_checkpoint("during Laurent monomial height admission")
            product = factor * term.coefficient.as_fraction()
            scaled.append(product)
            if not product:
                continue
            numerator_digits = _integer_digits(product.numerator)
            denominator_digits = _integer_digits(product.denominator)
            if (
                numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
                or denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS
            ):
                return overflow, monomial.terms[0], other, tuple(scaled)
            aggregate_digits += _laurent_encoded_digits(term.exponents, product)
            if aggregate_digits > MAX_LAURENT_RESULT_DIGITS:
                return overflow, monomial.terms[0], other, tuple(scaled)
            height = max(height, numerator_digits, denominator_digits)
        return height, monomial.terms[0], other, tuple(scaled)
    return None


def _integer_digits(value: int) -> int:
    return 1 if value == 0 else len(format_canonical_integer(abs(value)))


# Fixed canonical-JSON overhead per term: the ``coefficient``/``num``/``den``/
# ``exponents`` key names, quotes, braces, and separators. Every other counted
# character is a digit, sign, or exponent separator, so this constant plus the
# component digit counts upper-bounds the term's encoded size.
_LAURENT_TERM_JSON_OVERHEAD = 64
# Enclosing ``{"variables":[...],"terms":[...]}`` field overhead for the result.
_LAURENT_RESULT_JSON_OVERHEAD = 64


def _laurent_encoded_digits(exponents: tuple[int, ...], value: Fraction) -> int:
    """Bound one term's canonical-JSON encoded size in ASCII characters.

    The aggregate envelope is an encoded-size ceiling, not a coefficient-digit
    count: the canonical JSON also encodes every term's keys, quotes, commas,
    and exponent array. Counting those keeps the bound a true signed-byte
    bound instead of admitting a result that cannot fit the claimed limit.
    """

    if not value:
        return 0
    return (
        # Sign plus digits for the numerator; digits for the denominator.
        1
        + _integer_digits(value.numerator)
        + _integer_digits(value.denominator)
        # Sign plus digits plus a separating comma for each exponent.
        + sum(2 + _integer_digits(exponent) for exponent in exponents)
        + _LAURENT_TERM_JSON_OVERHEAD
    )


def _maximum_coefficient_digits(
    left: RationalLaurentPolynomial, right: RationalLaurentPolynomial
) -> tuple[int, dict[tuple[int, ...], Fraction]]:
    """Bound one collected coefficient before exact convolution begins.

    Each output coefficient collects only the pairs whose exponents sum to one
    output exponent. Its denominator is the LCM of those pairs' denominators
    and never combines unrelated supports, so bound per collision group rather
    than from the operand-wide LCM.

    Returns the bound together with the collected exponent-to-Fraction
    convolution so the kernel can reuse it instead of recomputing every
    exact product.
    """

    groups: dict[tuple[int, ...], Fraction] = {}
    group_widths: dict[tuple[int, ...], int] = {}
    running_aggregate_digits = 0
    pairs = 0
    for left_term in left.terms:
        left_coefficient = left_term.coefficient
        for right_term in right.terms:
            if pairs % 128 == 0:
                request_checkpoint("during Laurent coefficient-height admission")
            exponent = tuple(
                a + b
                for a, b in zip(left_term.exponents, right_term.exponents, strict=True)
            )
            # Accumulate the exact signed pair coefficient and bound the
            # reduced running group after every addition.  The denominator is
            # held to the canonical envelope so a group of unrelated
            # denominators cannot grow unbounded, while the numerator is given
            # a transient allowance: a later contribution in the same group can
            # cancel it, so a temporary overshoot is not yet an output bound.
            pair = Fraction(
                left_coefficient.num * right_term.coefficient.num,
                left_coefficient.den * right_term.coefficient.den,
            )
            current = groups.get(exponent)
            result = pair if current is None else current + pair
            if (
                _integer_digits(result.denominator) > 2 * MAX_CANONICAL_RATIONAL_DIGITS
                or _integer_digits(result.numerator) > 2 * MAX_CANONICAL_RATIONAL_DIGITS
            ):
                return MAX_CANONICAL_RATIONAL_DIGITS + 1, groups
            # Track the aggregate encoded output width while collecting, so
            # output growth is refused before the whole convolution is
            # materialized.
            width = _laurent_encoded_digits(exponent, result)
            running_aggregate_digits += width - group_widths.get(exponent, 0)
            if running_aggregate_digits > MAX_LAURENT_RESULT_DIGITS:
                return MAX_CANONICAL_RATIONAL_DIGITS + 1, groups
            group_widths[exponent] = width
            groups[exponent] = result
            pairs += 1
    height = 1
    aggregate_digits = 0
    for index, (exponents, total) in enumerate(groups.items()):
        if index % 128 == 0:
            request_checkpoint("during Laurent coefficient-height scan")
        if total == 0:
            continue
        numerator_digits = _integer_digits(total.numerator)
        denominator_digits = _integer_digits(total.denominator)
        if (
            numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
            or denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            return MAX_CANONICAL_RATIONAL_DIGITS + 1, groups
        # The term's keys, quotes, commas, and exponent array count toward the
        # serialized output alongside both retained coefficient components.
        aggregate_digits += _laurent_encoded_digits(exponents, total)
        if aggregate_digits > MAX_LAURENT_RESULT_DIGITS:
            return MAX_CANONICAL_RATIONAL_DIGITS + 1, groups
        height = max(height, numerator_digits, denominator_digits)
    return height, groups


def _result_from_coefficients(
    variables: tuple[str, ...], coefficients: dict[tuple[int, ...], Fraction]
) -> RationalLaurentPolynomial:
    terms: list[RationalLaurentPolynomialTerm] = []
    for index, (exponents, value) in enumerate(
        sorted(coefficients.items(), reverse=True)
    ):
        if index % 128 == 0:
            request_checkpoint("during Laurent result construction")
        if not value:
            continue
        component_digits = max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        if component_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationResourceAdmissionError(
                location=("right", "terms"),
                code="polynomial.laurent.output_growth",
                message="a reduced Laurent coefficient exceeds the exact-output bound",
            )
        terms.append(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(
                    num=value.numerator, den=value.denominator
                ),
                exponents=exponents,
            )
        )
    return RationalLaurentPolynomial(variables=variables, terms=tuple(terms))


def _admit_operand(
    operand: RationalLaurentPolynomial, *, location: str
) -> RationalLaurentPolynomial:
    if not isinstance(operand, RationalLaurentPolynomial):
        raise OperationDomainValidationError(
            location=(location,),
            code="polynomial.laurent.operand_type",
            message="Laurent factors must be canonical rational Laurent polynomials",
        )
    raw_terms = getattr(operand, "terms", None)
    if type(raw_terms) is not tuple:
        raise OperationDomainValidationError(
            location=(location, "terms"),
            code="polynomial.laurent.operand_shape",
            message="Laurent factors must expose canonical term fields",
        )
    if len(raw_terms) > MAX_POLYNOMIAL_TERMS:
        raise OperationDomainValidationError(
            location=(location, "terms"),
            code="polynomial.laurent.term_bound",
            message=f"Laurent factors admit at most {MAX_POLYNOMIAL_TERMS} terms",
        )
    admitted_terms: list[RationalLaurentPolynomialTerm] = []
    try:
        for index, term in enumerate(raw_terms):
            if index % 128 == 0:
                request_checkpoint("during Laurent operand reconstruction")
            admitted_terms.append(
                RationalLaurentPolynomialTerm.model_validate(
                    {
                        "coefficient": {
                            "num": getattr(term.coefficient, "num", None),
                            "den": getattr(term.coefficient, "den", None),
                        },
                        "exponents": getattr(term, "exponents", None),
                    }
                )
            )
        return RationalLaurentPolynomial(
            domain=getattr(operand, "domain", "QQ"),
            variables=getattr(operand, "variables", ()),
            terms=tuple(admitted_terms),
        )
    except (AttributeError, TypeError) as error:
        raise OperationDomainValidationError(
            location=(location,),
            code="polynomial.laurent.operand_shape",
            message="Laurent factors must expose canonical term fields",
        ) from error
    except ValidationError as error:
        detail = error.errors()[0]
        raise OperationDomainValidationError(
            location=(location, *tuple(detail.get("loc", ()))),
            code=str(detail["type"]),
            message=str(detail["msg"]),
        ) from error


def rational_laurent_multiply(
    left: RationalLaurentPolynomial, right: RationalLaurentPolynomial
) -> RationalLaurentPolynomial:
    left = _admit_operand(left, location="left")
    right = _admit_operand(right, location="right")
    if left.variables != right.variables:
        raise OperationDomainValidationError(
            location=("right", "variables"),
            code="polynomial.laurent.axis_mismatch",
            message="Laurent factors must use the same ordered variable axis",
        )
    if not left.terms or not right.terms:
        return RationalLaurentPolynomial(variables=left.variables, terms=())
    request_checkpoint("before Laurent semantic admission")
    # Signed extrema bound every convolution exponent without excluding
    # products whose opposite-sign source exponents cancel.
    if any(
        min(term.exponents[index] for term in left.terms)
        + min(term.exponents[index] for term in right.terms)
        < -MAX_POLYNOMIAL_EXPONENT
        or max(term.exponents[index] for term in left.terms)
        + max(term.exponents[index] for term in right.terms)
        > MAX_POLYNOMIAL_EXPONENT
        for index in range(len(left.variables))
    ):
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.exponent_growth",
            message="product exponent may exceed the Laurent representation bound",
        )
    work = len(left.terms) * len(right.terms)
    if work > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.convolution_bound",
            message=f"sparse convolution requires {work} term products; maximum is {MAX_POLYNOMIAL_TERMS}",
        )
    monomial_scale = _monomial_coefficient_digits(left, right)
    collected: dict[tuple[int, ...], Fraction] | None = None
    if monomial_scale is not None:
        coefficient_digits = monomial_scale[0]
    else:
        coefficient_digits, collected = _maximum_coefficient_digits(left, right)
    if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="polynomial.laurent.coefficient_growth",
            message="Laurent convolution coefficients exceed the exact-output bound",
        )
    request_checkpoint("after Laurent semantic admission")

    # A monomial only shifts the other support and scales its coefficients;
    # avoid paying for a general pair convolution in that common case.
    if monomial_scale is not None:
        monomial, source, scaled_coefficients = monomial_scale[1:]
        monomial_coefficients: dict[tuple[int, ...], Fraction] = {}
        for index, (term, product) in enumerate(
            zip(source.terms, scaled_coefficients, strict=True)
        ):
            if index % 128 == 0:
                request_checkpoint("during Laurent monomial multiplication")
            exponents = tuple(
                a + b for a, b in zip(monomial.exponents, term.exponents, strict=True)
            )
            if any(abs(exponent) > MAX_POLYNOMIAL_EXPONENT for exponent in exponents):
                raise OperationResourceAdmissionError(
                    location=("right", "terms"),
                    code="polynomial.laurent.exponent_growth",
                    message="product exponent exceeds the Laurent representation bound",
                )
            monomial_coefficients[exponents] = product
        result = _result_from_coefficients(left.variables, monomial_coefficients)
        request_checkpoint("after Laurent monomial result construction")
        return result

    coefficients: dict[tuple[int, ...], Fraction] = collected or {}
    if collected is None:
        pairs = 0
        for left_term in left.terms:
            for right_term in right.terms:
                if pairs % 128 == 0:
                    request_checkpoint("during Laurent convolution")
                exponents = tuple(
                    a + b
                    for a, b in zip(
                        left_term.exponents, right_term.exponents, strict=True
                    )
                )
                coefficients[exponents] = coefficients.get(exponents, Fraction()) + (
                    left_term.coefficient.as_fraction()
                    * right_term.coefficient.as_fraction()
                )
                pairs += 1
    result = _result_from_coefficients(left.variables, coefficients)
    request_checkpoint("after Laurent result construction")
    return result


def _run(request: RationalLaurentMultiplyRequest) -> RationalLaurentPolynomial:
    return rational_laurent_multiply(request.left, request.right)
