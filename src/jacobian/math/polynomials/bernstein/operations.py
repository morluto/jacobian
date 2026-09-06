"""Exact separable monomial-to-Bernstein maps, with one native admission."""

from itertools import product
from math import comb, prod
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
) -> tuple[tuple[int, ...], ...]:
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

    # A common denominator divides the product of source denominators,
    # endpoint denominator products to each coordinate degree, and all
    # binomial(m,j), j<=degree. Bound binomial(m,j) by m**j without expanding it.
    denominator = sum(_denominator_digits(t.coefficient.den) for t in terms)
    magnitude = max((_digits(t.coefficient.num) for t in terms), default=1)
    binomial_digits = 0
    for e, m, interval in zip(degrees, multidegree, box.intervals, strict=True):
        endpoint = max(_digits(interval.lower.num), _digits(interval.upper.num))
        denominator += e * (
            _denominator_digits(interval.lower.den)
            + _denominator_digits(interval.upper.den)
        )
        denominator += e * (e + 1) // 2 * len(str(max(1, m)))
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
    # rows are also covered. No dense global basis-change matrix is built.
    work = size * (len(terms) * (dimension + 1) + 1)
    work += sum(
        (m + 1) * sum((e + 1) ** 2 * 8 for e in es)
        for es, m in zip(exponents, multidegree, strict=True)
    )
    if work * (1 + height // 64) ** 2 > MAX_WEIGHTED_WORK:
        _reject("Bernstein conversion exceeds the height-weighted arithmetic budget")
    return exponents


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
    exponents = _admit(polynomial, box, multidegree)
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


__all__ = ["bernstein_coefficients"]
