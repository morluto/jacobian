"""Exact sparse rational Laurent-polynomial multiplication."""

from fractions import Fraction

from pydantic import ValidationError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
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
        for index, term in enumerate(other.terms):
            if index % 128 == 0:
                request_checkpoint("during Laurent monomial height admission")
            product = factor * term.coefficient.as_fraction()
            scaled.append(product)
            digits = _fraction_component_digits(product)
            if digits > height:
                height = digits
        return height, monomial.terms[0], other, tuple(scaled)
    return None


def _operand_coefficient_digits(
    terms: tuple[RationalLaurentPolynomialTerm, ...],
) -> tuple[int, bool]:
    height = 1
    shared_denominator: int | None = None
    for index, term in enumerate(terms):
        if index % 128 == 0:
            request_checkpoint("during Laurent coefficient-height admission")
        digits = canonical_rational_component_digits(term.coefficient)
        if digits > height:
            height = digits
        denominator = term.coefficient.den
        if shared_denominator is None:
            shared_denominator = denominator
        elif denominator != shared_denominator:
            shared_denominator = 0
    return height, shared_denominator is not None and shared_denominator != 0


def _maximum_coefficient_digits(
    left: RationalLaurentPolynomial, right: RationalLaurentPolynomial
) -> int:
    """Bound one collected coefficient before exact convolution begins."""

    collisions = min(len(left.terms), len(right.terms))
    left_digits, left_shared = _operand_coefficient_digits(left.terms)
    right_digits, right_shared = _operand_coefficient_digits(right.terms)
    addition_digits = len(str(collisions)) if collisions > 1 else 0
    product_digits = left_digits + right_digits
    if left_shared and right_shared:
        return product_digits + addition_digits
    return collisions * product_digits + addition_digits


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
    coefficient_digits = (
        monomial_scale[0]
        if monomial_scale is not None
        else _maximum_coefficient_digits(left, right)
    )
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

    coefficients: dict[tuple[int, ...], Fraction] = {}
    pairs = 0
    for left_term in left.terms:
        for right_term in right.terms:
            if pairs % 128 == 0:
                request_checkpoint("during Laurent convolution")
            exponents = tuple(
                a + b
                for a, b in zip(left_term.exponents, right_term.exponents, strict=True)
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
