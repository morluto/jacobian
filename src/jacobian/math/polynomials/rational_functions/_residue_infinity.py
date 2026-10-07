"""Bounded extraction of the Laurent coefficient at infinity over QQ(x)."""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from time import monotonic

from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    current_request_execution,
    execution_deadline,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_RESIDUE_PRIVATE_BITS = 262_144
MAX_RESIDUE_LIMB_WORK = 1 << 37
MAX_RESIDUE_STORAGE_BITS = 1 << 29
RESIDUE_WALL_SECONDS = 60.0
_SCALAR_LIMIT = 10**MAX_CANONICAL_RATIONAL_DIGITS


def _resource(reason: str) -> None:
    raise OperationResourceAdmissionError(
        location=("function",),
        code=f"rational_function.residue_{reason}",
        message=f"residue extraction exceeds its {reason} envelope",
    )


@dataclass
class _Budget:
    work: int = 0

    def charge(self, operations: int, bits: int) -> None:
        request_checkpoint("during infinity-residue admission")
        if bits > MAX_RESIDUE_PRIVATE_BITS:
            _resource("intermediate_height")
        self.work += operations * max(1, (bits + 63) // 64) ** 2
        if self.work > MAX_RESIDUE_LIMB_WORK:
            _resource("work")

    def multiply(self, left: int, right: int) -> int:
        self.charge(1, left.bit_length() + right.bit_length())
        return left * right


def _admit_source(function: object) -> RationalFunction:
    """Bound native containers before structural revalidation and arithmetic."""
    try:
        if (
            not isinstance(function, RationalFunction)
            or type(function.domain) is not str
            or function.domain != "QQ"
            or type(function.variables) is not tuple
            or len(function.variables) != 1
            or type(function.variables[0]) is not str
            or len(function.variables[0]) > 32
        ):
            raise ValueError("residue requires one bounded univariate source")
        for part in (function.numerator, function.denominator):
            if (
                type(part) is not SparseRationalPolynomial
                or type(part.terms) is not tuple
                or len(part.terms) > MAX_RATIONAL_FUNCTION_TERMS
            ):
                raise ValueError("invalid residue polynomial containers")
            for term in part.terms:
                if (
                    type(term) is not RationalPolynomialTerm
                    or type(term.exponents) is not tuple
                    or len(term.exponents) != 1
                    or type(term.exponents[0]) is not int
                    or not 0
                    <= term.exponents[0]
                    <= MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
                    or type(term.coefficient) is not CanonicalRational
                    or any(
                        type(value) is not int
                        or abs(value).bit_length()
                        > 4 * MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
                        for value in (term.coefficient.num, term.coefficient.den)
                    )
                ):
                    raise ValueError("invalid residue scalar or exponent structure")
        # The carrier deliberately parses presentations without a symbolic
        # GCD. The Laurent coefficient depends only on their ratio, so common
        # factor cancellation is not a hypothesis of this computation.
        return RationalFunction.model_validate(function.model_dump())
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("function",),
            code="rational_function.residue_source",
            message="residue requires a structurally canonical univariate rational presentation",
        ) from exc


def _clear(
    terms: tuple[RationalPolynomialTerm, ...], budget: _Budget
) -> tuple[int, tuple[int, ...]]:
    common = 1
    for term in terms:
        denominator = term.coefficient.den
        budget.charge(1, common.bit_length() + denominator.bit_length())
        common = budget.multiply(common // gcd(common, denominator), denominator)
    coefficients = tuple(
        budget.multiply(abs(term.coefficient.num), common // term.coefficient.den)
        for term in terms
    )
    return common, coefficients


def _admit_division(function: RationalFunction, steps: int) -> None:
    """Prove all monic long-division updates before extracting a coefficient.

    For A=L*N and B=M*D integral, LC(B)=M. After j leading-term
    eliminations, every partial remainder coefficient has denominator dividing
    L*M**j and common numerator height at most H*C**j, where
    H=max|A_i| and C=M+max|nonleading B_i|. Reduction cannot enlarge these
    common bounds. Unreduced Fraction products/additions fit twice the larger
    scalar width; sparse updates and gcds receive a squared-limb work charge.
    """
    budget = _Budget()
    left_den, left = _clear(function.numerator.terms, budget)
    right_den, right = _clear(function.denominator.terms, budget)
    height = max(left)
    growth = right_den + max(right[1:], default=0)
    denominator = left_den
    for _ in range(steps):
        height = budget.multiply(height, growth)
        denominator = budget.multiply(denominator, right_den)
        if height >= _SCALAR_LIMIT or denominator >= _SCALAR_LIMIT:
            _resource("coefficient_height")
    bits = 2 * max(height.bit_length(), denominator.bit_length()) + 4
    # Include structural/scalar admission, source construction, elimination,
    # result construction, and exact encoding. No quotient is retained.
    cells = function.numerator.terms[0].exponents[0] + 1
    operations = 64 * (cells + len(function.denominator.terms))
    operations += 16 * steps * len(function.denominator.terms)
    budget.charge(operations, bits)
    if 16 * cells * bits > MAX_RESIDUE_STORAGE_BITS:
        _resource("storage")


def exact_residue_at_infinity(function: RationalFunction) -> CanonicalRational:
    """Return minus the x^-1 coefficient using an admitted monic remainder."""
    if current_request_execution() is None:
        with request_execution(monotonic()):
            return exact_residue_at_infinity(function)
    execution_deadline(RESIDUE_WALL_SECONDS)
    source = _admit_source(function)
    request_checkpoint("after infinity-residue structural admission")
    numerator, denominator = source.numerator.terms, source.denominator.terms
    degree = denominator[0].exponents[0]
    if not numerator or degree == 0:
        return CanonicalRational(num=0, den=1)
    target = degree - 1
    highest = numerator[0].exponents[0]
    if highest < degree:
        coefficient = next(
            (
                term.coefficient.as_fraction()
                for term in numerator
                if term.exponents[0] == target
            ),
            Fraction(0),
        )
        return CanonicalRational.from_fraction(-coefficient)
    _admit_division(source, highest - degree + 1)
    remainder = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in numerator
        if term.exponents[0] >= target
    }
    tail = tuple(
        (term.exponents[0], term.coefficient.as_fraction()) for term in denominator[1:]
    )
    for exponent in range(highest, degree - 1, -1):
        request_checkpoint("during infinity-residue long division")
        leading = remainder.pop(exponent, Fraction(0))
        if not leading:
            continue
        for power, coefficient in tail:
            index = exponent - degree + power
            # Lower terms can never affect this or any later leading term,
            # nor the target coefficient, so do not construct them.
            if index < target:
                continue
            value = remainder.get(index, Fraction(0)) - leading * coefficient
            if value:
                remainder[index] = value
            else:
                remainder.pop(index, None)
    result = CanonicalRational.from_fraction(-remainder.get(target, Fraction(0)))
    request_checkpoint("after infinity-residue result construction")
    return result
