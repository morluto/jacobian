"""Exact separable monomial-to-Bernstein maps, with one native admission."""

from fractions import Fraction
from itertools import product
from math import comb, lcm, prod
from time import monotonic

from flint import fmpq

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import RationalBox
from jacobian.math.polynomials.bernstein.values import (
    Multidegree,
    RationalBernsteinPolynomial,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_TENSOR_COEFFICIENTS = 65536
MAX_REPRESENTATION_CHARS = 4_000_000
MAX_WEIGHTED_WORK = 20_000_000
MAX_COMPONENT_DIGITS = 8192
BERNSTEIN_WALL_SECONDS = 120.0


def _reject(message: str) -> None:
    raise OperationDomainValidationError(
        location=("polynomial", "box", "multidegree"),
        code="polynomial.bernstein_admission",
        message=message,
    )


def _digits(value: str) -> int:
    return len(value.lstrip("-"))


def _denominator_digits(value: str) -> int:
    return 0 if value == "1" else len(value)


def _admit(
    polynomial: RationalPolynomial,
    box: RationalBox,
    multidegree: Multidegree,
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    dimension = len(polynomial.variables)
    if box.variables != polynomial.variables or len(multidegree) != dimension:
        _reject(
            "box and multidegree must follow the polynomial's complete ordered axes"
        )
    if any(
        type(m) is not int or m < 0 or m >= MAX_TENSOR_COEFFICIENTS for m in multidegree
    ):
        _reject("multidegrees must be nonnegative integers within the tensor envelope")
    if any(i.lower == i.upper for i in box.intervals):
        _reject("Bernstein boxes require strictly positive width on every axis")
    size = prod(m + 1 for m in multidegree)
    if size > MAX_TENSOR_COEFFICIENTS:
        _reject("Bernstein tensor exceeds 65536 coefficients")
    terms = polynomial.polynomial.terms
    exponents = tuple(
        tuple(sorted({t.exponents[i] for t in terms})) for i in range(dimension)
    )
    degrees = tuple(max(es, default=0) for es in exponents)
    if any(e > m for e, m in zip(degrees, multidegree, strict=True)):
        _reject("multidegree must dominate every source coordinate degree")
    # A rectangular support can use separable tensor contractions.  This is
    # the actual kernel cost for dense inputs; the sparse path below retains
    # the source-term cost for genuinely sparse polynomials.
    dense = bool(terms) and len(terms) == prod(e + 1 for e in degrees)

    # A common denominator divides the product of source denominators,
    # endpoint denominator products to each coordinate degree, and all
    # binomial(m,j), j<=degree. Bound binomial(m,j) by m**j without expanding it.
    source_denominator = 1
    source_denominator_digits = 0
    for term in terms:
        if _denominator_digits(term.coefficient.den) > MAX_COMPONENT_DIGITS:
            source_denominator_digits = MAX_COMPONENT_DIGITS + 1
            break
        source_denominator = lcm(source_denominator, int(term.coefficient.den))
        if source_denominator.bit_length() > MAX_COMPONENT_DIGITS * 4:
            source_denominator_digits = MAX_COMPONENT_DIGITS + 1
            break
        source_denominator_digits = len(str(source_denominator))
    denominator = source_denominator_digits
    magnitude = max((_digits(t.coefficient.num) for t in terms), default=1)
    binomial_digits = 0
    for e, m, interval in zip(degrees, multidegree, box.intervals, strict=True):
        endpoint = max(_digits(interval.lower.num), _digits(interval.upper.num))
        denominator += e * (
            _denominator_digits(interval.lower.den)
            + _denominator_digits(interval.upper.den)
        )
        if e:
            binomial_denominator = lcm(*(comb(m, j) for j in range(e + 1)))
            denominator += len(str(binomial_denominator))
        # |a|+|b-a| < 3*10**endpoint; the binomial ratio is at most one.
        magnitude += e * (endpoint + 1)
        binomial_digits += e * len(str(max(1, m)))
    height = (
        denominator + magnitude + len(str(max(1, len(terms)))) + binomial_digits + 3
    )
    if height > MAX_COMPONENT_DIGITS:
        _reject("Bernstein rational growth exceeds the 8192-digit component envelope")
    # Kernel rational cross-products fit 4*height digits. Canonical input
    # intervals own endpoint ordering; equality detects point boxes without
    # replaying their large-rational comparisons.
    cache_size = sum(
        len(es) * (m + 1) for es, m in zip(exponents, multidegree, strict=True)
    )
    source_chars = sum(
        _digits(t.coefficient.num) + len(t.coefficient.den) + 128 for t in terms
    )
    source_chars += sum(
        _digits(q.num) + len(q.den) + 128
        for i in box.intervals
        for q in (i.lower, i.upper)
    )
    if (size + cache_size) * (
        2 * height + 64
    ) + source_chars > MAX_REPRESENTATION_CHARS:
        _reject(
            "Bernstein tensor, axis maps, and source exceed the representation budget"
        )
    # comb(k,j) uses at most j integer steps; powers and per-axis binomial
    # rows are also covered. Dense inputs are contracted one axis at a time,
    # so their work is proportional to the tensor size and axis widths rather
    # than to the product of tensor size and source support.
    if dense:
        work = size * (sum(m + 1 for m in multidegree) + 1)
    else:
        work = size * (len(terms) * (dimension + 1) + 1)
    # Each power-row entry performs bounded rational powers and cross-products;
    # each output-row entry performs one weighted term and an accumulation.
    work += sum(
        (m + 1) * sum(4 * (e + 1) + 4 for e in es)
        for es, m in zip(exponents, multidegree, strict=True)
    )
    if work * (1 + height // 64) ** 2 > MAX_WEIGHTED_WORK:
        _reject("Bernstein conversion exceeds the height-weighted arithmetic budget")
    return exponents, dense


def _contract_axis(
    values: list[fmpq],
    shape: tuple[int, ...],
    axis: int,
    rows: dict[int, tuple[fmpq, ...]],
    target_degree: int,
) -> tuple[list[fmpq], tuple[int, ...]]:
    """Contract one tensor axis with the monomial-to-Bernstein rows."""
    source_degree = shape[axis] - 1
    after = prod(shape[axis + 1 :])
    before = prod(shape[:axis])
    target_shape = (*shape[:axis], target_degree + 1, *shape[axis + 1 :])
    output: list[fmpq] = []
    for prefix in range(before):
        for target in range(target_degree + 1):
            for suffix in range(after):
                value = fmpq(0)
                for source in range(source_degree + 1):
                    value += (
                        values[(prefix * (source_degree + 1) + source) * after + suffix]
                        * rows[source][target]
                    )
                output.append(value)
    return output, target_shape


def _checkpoint(deadline: float) -> None:
    request_checkpoint("during Bernstein conversion")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError("Bernstein conversion deadline expired")


def bernstein_coefficients(
    polynomial: RationalPolynomial,
    box: RationalBox,
    multidegree: Multidegree,
) -> RationalBernsteinPolynomial:
    """Return exact tensor coordinates on a nondegenerate rational box."""
    execution = current_request_execution()
    started = monotonic() if execution is None else execution.started_at
    deadline = started + BERNSTEIN_WALL_SECONDS
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    _checkpoint(deadline)
    exponents, dense = _admit(polynomial, box, multidegree)
    _checkpoint(deadline)
    axis_maps: list[dict[int, tuple[fmpq, ...]]] = []
    for es, m, interval in zip(exponents, multidegree, box.intervals, strict=True):
        maps: dict[int, tuple[fmpq, ...]] = {}
        if not es:
            axis_maps.append(maps)
            continue
        if 0 in es:
            maps[0] = (fmpq(1),) * (m + 1)
        active = tuple(e for e in es if e != 0)
        if not active:
            axis_maps.append(maps)
            continue
        lower = fmpq(*interval.lower.as_integer_ratio())
        width = fmpq(*interval.upper.as_integer_ratio()) - lower
        for e in active:
            _checkpoint(deadline)
            power = tuple(
                comb(e, j) * lower ** (e - j) * width**j / comb(m, j)
                for j in range(e + 1)
            )
            maps[e] = tuple(
                sum((power[j] * comb(k, j) for j in range(min(e, k) + 1)), fmpq(0))
                for k in range(m + 1)
            )
        axis_maps.append(maps)
    terms = tuple(
        (fmpq(*t.coefficient.as_integer_ratio()), t.exponents)
        for t in polynomial.polynomial.terms
    )
    if dense:
        source_shape = tuple(
            e + 1 for e in tuple(max(es, default=0) for es in exponents)
        )
        source_strides = tuple(
            prod(source_shape[i + 1 :]) for i in range(len(source_shape))
        )
        values = [fmpq(0)] * prod(source_shape)
        for coefficient, powers in terms:
            offset = sum(
                power * stride
                for power, stride in zip(powers, source_strides, strict=True)
            )
            values[offset] = coefficient
        shape = source_shape
        for axis, rows in enumerate(axis_maps):
            _checkpoint(deadline)
            values, shape = _contract_axis(values, shape, axis, rows, multidegree[axis])
        coefficients = [
            CanonicalRational.model_construct(
                num=format_canonical_integer(int(value.numerator)),
                den=format_canonical_integer(int(value.denominator)),
            )
            for value in values
        ]
        result = RationalBernsteinPolynomial.model_construct(
            polynomial=polynomial,
            box=box,
            multidegree=multidegree,
            coefficients=tuple(coefficients),
        )
        _checkpoint(deadline)
        return result
    coefficients = []
    for index in product(*(range(m + 1) for m in multidegree)):
        _checkpoint(deadline)
        value = fmpq(0)
        for coefficient, powers in terms:
            contribution = coefficient
            for i, e in enumerate(powers):
                contribution *= axis_maps[i][e][index[i]]
            value += contribution
        # FLINT already returns a reduced rational. Do not repeat gcd work.
        coefficients.append(
            CanonicalRational.model_construct(
                num=format_canonical_integer(int(value.numerator)),
                den=format_canonical_integer(int(value.denominator)),
            )
        )
    result = RationalBernsteinPolynomial.model_construct(
        polynomial=polynomial,
        box=box,
        multidegree=multidegree,
        coefficients=tuple(coefficients),
    )

    _checkpoint(deadline)
    return result


def verify_bernstein_coefficients(claim: RationalBernsteinPolynomial) -> bool:
    """Verify tensor coefficients against the retained polynomial and box."""
    try:
        expected = bernstein_coefficients(
            claim.polynomial, claim.box, claim.multidegree
        )
        return expected.coefficients == claim.coefficients
    except (AttributeError, TypeError, ValueError, OperationDomainValidationError):
        return False


def _restriction_admit(
    parent: RationalBernsteinPolynomial,
    child: RationalBox,
) -> tuple[Fraction, ...]:
    if child.variables != parent.box.variables:
        _reject("restriction child must follow the parent's complete ordered axes")
    ratios: list[Fraction] = []
    max_height = 1
    for parent_interval, child_interval in zip(
        parent.box.intervals, child.intervals, strict=True
    ):
        parent_lower = parent_interval.lower.as_fraction()
        parent_upper = parent_interval.upper.as_fraction()
        child_lower = child_interval.lower.as_fraction()
        child_upper = child_interval.upper.as_fraction()
        if parent_lower >= parent_upper:
            _reject("restriction requires a nondegenerate parent on every axis")
        if child_lower >= child_upper:
            _reject("restriction requires a nondegenerate child on every axis")
        if child_lower < parent_lower or child_upper > parent_upper:
            _reject("restriction child must be contained in the parent box")
        alpha = (child_lower - parent_lower) / (parent_upper - parent_lower)
        beta = (child_upper - parent_lower) / (parent_upper - parent_lower)
        ratios.extend((alpha, beta))
        max_height = max(
            max_height,
            len(str(abs(alpha.numerator))),
            len(str(alpha.denominator)),
            len(str(abs(beta.numerator))),
            len(str(beta.denominator)),
        )
    coefficient_height = max(
        max(
            len(value.num.lstrip("-")),
            len(value.den),
        )
        for value in parent.coefficients
    )
    # Each de Casteljau level introduces one ratio cross-product and one
    # addition.  Account for both the ratio digits and the additions in the
    # output-component envelope before allocating the restricted tensor.
    degree_height = sum(m * (max_height + 1) + m for m in parent.multidegree)
    output_height = coefficient_height + degree_height + len(parent.multidegree) + 2
    if output_height > MAX_COMPONENT_DIGITS:
        _reject("Bernstein restriction rational growth exceeds the 8192-digit envelope")
    size = prod(m + 1 for m in parent.multidegree)
    work = size * (sum(m + 1 for m in parent.multidegree) + 1)
    if work * (1 + output_height // 64) ** 2 > MAX_WEIGHTED_WORK:
        _reject("Bernstein restriction exceeds the height-weighted arithmetic budget")
    return tuple(ratios)


def _split_fiber(
    values: list[Fraction],
    parameter: Fraction,
) -> tuple[list[Fraction], list[Fraction]]:
    left = [values[0]]
    right = [values[-1]]
    current = values
    for _ in range(1, len(values)):
        current = [
            (1 - parameter) * current[i] + parameter * current[i + 1]
            for i in range(len(current) - 1)
        ]
        left.append(current[0])
        right.append(current[-1])
    right.reverse()
    return left, right


def _restrict_axis(
    values: list[Fraction],
    shape: tuple[int, ...],
    axis: int,
    alpha: Fraction,
    beta: Fraction,
) -> list[Fraction]:
    width = shape[axis]
    after = prod(shape[axis + 1 :])
    before = prod(shape[:axis])
    output: list[Fraction] = []
    for prefix in range(before):
        fibers = [
            [
                values[(prefix * width + source) * after + suffix]
                for source in range(width)
            ]
            for suffix in range(after)
        ]
        restricted_fibers = []
        for fiber in fibers:
            if alpha == 0:
                segment = fiber
            else:
                _, segment = _split_fiber(fiber, alpha)
            if beta == 1:
                restricted_fibers.append(segment)
            else:
                parameter = (beta - alpha) / (1 - alpha)
                restricted, _ = _split_fiber(segment, parameter)
                restricted_fibers.append(restricted)
        for target in range(width):
            for suffix in range(after):
                output.append(restricted_fibers[suffix][target])
    return output


def _restrict_trusted(
    parent: RationalBernsteinPolynomial,
    child: RationalBox,
) -> RationalBernsteinPolynomial:
    execution = current_request_execution()
    started = monotonic() if execution is None else execution.started_at
    deadline = started + BERNSTEIN_WALL_SECONDS
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    _checkpoint(deadline)
    ratios = _restriction_admit(parent, child)
    shape = tuple(m + 1 for m in parent.multidegree)
    values = [coefficient.as_fraction() for coefficient in parent.coefficients]
    for axis, (alpha, beta) in enumerate(zip(ratios[::2], ratios[1::2], strict=True)):
        _checkpoint(deadline)
        values = _restrict_axis(values, shape, axis, alpha, beta)
    coefficients = tuple(CanonicalRational.from_fraction(value) for value in values)
    result = RationalBernsteinPolynomial.model_construct(
        polynomial=parent.polynomial,
        box=child,
        multidegree=parent.multidegree,
        coefficients=coefficients,
    )
    _checkpoint(deadline)
    return result


def restrict_bernstein(
    parent: RationalBernsteinPolynomial,
    child: RationalBox,
) -> RationalBernsteinPolynomial:
    """Restrict a source-bound Bernstein tensor to a rational subbox."""
    if not verify_bernstein_coefficients(parent):
        _reject("restriction parent coefficients do not match their source and box")
    return _restrict_trusted(parent, child)


def verify_bernstein_restriction(
    parent: RationalBernsteinPolynomial,
    claim: RationalBernsteinPolynomial,
) -> bool:
    """Verify a serialized child tensor is the exact restriction of its parent."""
    try:
        if (
            parent.polynomial != claim.polynomial
            or parent.multidegree != claim.multidegree
        ):
            return False
        if not verify_bernstein_coefficients(parent):
            return False
        expected = _restrict_trusted(parent, claim.box)
        return expected.coefficients == claim.coefficients
    except (AttributeError, TypeError, ValueError, OperationDomainValidationError):
        return False


__all__ = [
    "bernstein_coefficients",
    "restrict_bernstein",
    "verify_bernstein_coefficients",
    "verify_bernstein_restriction",
]
