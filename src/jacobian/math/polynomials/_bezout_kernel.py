"""Bounded exact Bézout witnesses through maintained sparse algebra kernels."""

import re
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Any, Literal

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._models import _MAX_GCD_DEGREE, _MAX_GCD_TERMS
from jacobian.math.polynomials.values import (
    MonicPolynomial,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_CANONICAL_LIMIT = 10**MAX_CANONICAL_RATIONAL_DIGITS
MAX_BEZOUT_SOURCE_COEFFICIENT_DIGITS = 256
MAX_BEZOUT_PRIVATE_BITS = 262_144
MAX_BEZOUT_WORK = 1 << 37
MAX_BEZOUT_COEFFICIENT_STORAGE_BITS = 1 << 32
MAX_BEZOUT_RESULT_DIGITS = 9_000_000


def _reject(reason: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"), code=f"polynomial.gcd.{reason}", message=message
    )


@dataclass
class _Ledger:
    """Conservative squared 64-bit-limb policy work and exact coefficient storage."""

    work: int = 0
    coefficient_bits: int = 0

    def charge(self, count: int, bits: int = 1) -> None:
        request_checkpoint("during Bézout admission")
        if bits > MAX_BEZOUT_PRIVATE_BITS:
            _reject(
                "intermediate_height",
                "Bézout arithmetic exceeds its private scalar-bit envelope",
            )
        self.work += count * max(1, (bits + 63) // 64) ** 2
        if self.work > MAX_BEZOUT_WORK:
            _reject("work", "Bézout arithmetic exceeds its squared-limb work envelope")

    def multiply(self, left: int, right: int) -> int:
        self.charge(1, abs(left).bit_length() + abs(right).bit_length())
        return left * right

    def retain(self, bits: int) -> None:
        if self.coefficient_bits + bits > MAX_BEZOUT_COEFFICIENT_STORAGE_BITS:
            _reject(
                "storage",
                "Bézout source clearing exceeds its exact-coefficient storage envelope",
            )
        self.coefficient_bits += bits


def require_bezout_source_shape(source: object) -> None:
    """Bound native containers before owner budget traversal or shortcuts."""
    if (
        not isinstance(source, RationalPolynomial)
        or type(source) not in (RationalPolynomial, MonicPolynomial)
        or getattr(source, "domain", None) != "QQ"
        or type(getattr(source, "variables", None)) is not tuple
        or len(source.variables) != 1
        or type(source.variables[0]) is not str
        or len(source.variables[0]) > 32
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", source.variables[0]) is None
        or type(getattr(source, "polynomial", None)) is not SparseRationalPolynomial
        or type(getattr(source.polynomial, "terms", None)) is not tuple
        or len(source.polynomial.terms) > min(_MAX_GCD_TERMS, _MAX_GCD_DEGREE + 1)
    ):
        raise OperationDomainValidationError(
            location=(),
            code="polynomial.gcd.source",
            message="Bézout sources must retain canonical QQ polynomial containers and one ordered axis",
        )
    for term in source.polynomial.terms:
        coefficient = getattr(term, "coefficient", None)
        exponents = getattr(term, "exponents", None)
        if (
            type(term) is not RationalPolynomialTerm
            or type(exponents) is not tuple
            or len(exponents) != 1
            or type(exponents[0]) is not int
            or not 0 <= exponents[0] <= _MAX_GCD_DEGREE
            or type(coefficient) is not CanonicalRational
            or type(getattr(coefficient, "num", None)) is not int
            or type(getattr(coefficient, "den", None)) is not int
            or not coefficient.num
            or coefficient.den <= 0
            or max(abs(coefficient.num).bit_length(), coefficient.den.bit_length())
            > 4 * MAX_BEZOUT_SOURCE_COEFFICIENT_DIGITS
        ):
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.gcd.source",
                message="Bézout sources must retain bounded canonical scalar and exponent structure",
            )

    if type(source) is MonicPolynomial and (
        not source.polynomial.terms
        or source.polynomial.terms[0].coefficient.as_integer_ratio() != (1, 1)
    ):
        raise OperationDomainValidationError(
            location=(),
            code="polynomial.gcd.source",
            message="monic Bézout sources must retain their monic subtype invariant",
        )


def _source_terms(
    source: RationalPolynomial, ledger: _Ledger
) -> tuple[RationalPolynomialTerm, ...]:
    previous = _MAX_GCD_DEGREE + 1
    terms = source.polynomial.terms
    for term in terms:
        if (
            type(term) is not RationalPolynomialTerm
            or type(term.exponents) is not tuple
            or len(term.exponents) != 1
            or type(term.exponents[0]) is not int
            or not 0 <= term.exponents[0] < previous
            or type(term.coefficient) is not CanonicalRational
        ):
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.gcd.source",
                message="Bézout sources must have canonical ordered univariate terms",
            )
        previous = term.exponents[0]
        coefficient = term.coefficient
        if (
            type(coefficient.num) is not int
            or type(coefficient.den) is not int
            or not coefficient.num
            or coefficient.den <= 0
        ):
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.gcd.source",
                message="Bézout sources must have nonzero canonical rational coefficients",
            )
        ledger.charge(
            1, abs(coefficient.num).bit_length() + coefficient.den.bit_length()
        )
        if gcd(coefficient.num, coefficient.den) != 1:
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.gcd.source",
                message="Bézout sources must have reduced rational coefficients",
            )
    return terms


@dataclass(frozen=True)
class _Primitive:
    terms: tuple[tuple[int, int], ...]
    scale: Fraction
    norm: int

    @property
    def degree(self) -> int:
        return self.terms[0][0]


def _primitive(
    terms: tuple[RationalPolynomialTerm, ...], ledger: _Ledger
) -> _Primitive:
    denominator = 1
    for value in dict.fromkeys(t.coefficient.den for t in terms):
        ledger.charge(1, denominator.bit_length() + value.bit_length())
        denominator = ledger.multiply(denominator // gcd(denominator, value), value)
    ledger.retain(denominator.bit_length())
    integral = []
    content = 0
    for term in terms:
        ledger.charge(1, denominator.bit_length() + term.coefficient.den.bit_length())
        value = ledger.multiply(
            term.coefficient.num, denominator // term.coefficient.den
        )
        ledger.charge(1, abs(value).bit_length() + content.bit_length())
        content = gcd(content, value)
        ledger.retain(abs(value).bit_length())
        integral.append((term.exponents[0], value))
    ledger.charge(len(integral) + 1, content.bit_length() + denominator.bit_length())
    # Reserve the normalized copy before constructing it; division by content
    # cannot increase an integral coefficient's size.
    ledger.retain(sum(abs(c).bit_length() for _, c in integral) + content.bit_length())
    primitive = tuple((e, c // content) for e, c in integral)
    norm_bits = (
        max(abs(c).bit_length() for _, c in primitive) + len(primitive).bit_length()
    )
    ledger.charge(len(primitive), norm_bits)
    return _Primitive(
        primitive, Fraction(content, denominator), sum(abs(c) for _, c in primitive)
    )


def _proportional(
    left: tuple[RationalPolynomialTerm, ...],
    right: tuple[RationalPolynomialTerm, ...],
    ledger: _Ledger,
) -> bool:
    if len(left) != len(right) or any(
        a.exponents != b.exponents for a, b in zip(left, right, strict=True)
    ):
        return False
    a, b = left[0].coefficient.as_fraction(), right[0].coefficient.as_fraction()
    for first, second in zip(left, right, strict=True):
        values = (
            first.coefficient.num,
            first.coefficient.den,
            second.coefficient.num,
            second.coefficient.den,
            a.numerator,
            a.denominator,
            b.numerator,
            b.denominator,
        )
        ledger.charge(3, sum(abs(value).bit_length() for value in values))
        if first.coefficient.as_fraction() * b != second.coefficient.as_fraction() * a:
            return False
    return True


def _digits_from_bits(bits: int) -> int:
    return max(1, (bits * 30103) // 100000 + 1)


def _reserve(
    ledger: _Ledger, *, scalar_bits: int, operations: int, cells: int, digits: int
) -> None:
    ledger.charge(operations, scalar_bits)
    if (
        ledger.coefficient_bits + cells * scalar_bits
        > MAX_BEZOUT_COEFFICIENT_STORAGE_BITS
    ):
        _reject(
            "storage",
            "Bézout computation exceeds its aggregate exact-coefficient storage envelope",
        )
    if digits > MAX_BEZOUT_RESULT_DIGITS:
        _reject(
            "result_digits",
            "Bézout profile exceeds its aggregate retained scalar-digit envelope",
        )


def _ring_source(ring: Any, source: RationalPolynomial) -> Any:
    return ring.from_dict(
        {
            t.exponents: ring.domain(t.coefficient.num, t.coefficient.den)
            for t in source.polynomial.terms
        }
    )


def _value(polynomial: Any, axis: tuple[str, ...]) -> RationalPolynomial:
    request_checkpoint("during Bézout result construction")
    return RationalPolynomial(
        variables=axis,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(
                        int(c.numerator), int(c.denominator)
                    ),
                    exponents=e,
                )
                for e, c in sorted(polynomial.items(), reverse=True)
            )
        ),
    )


def _simple(
    left: RationalPolynomial,
    right: RationalPolynomial,
    kind: Literal["left", "right", "left_unit", "right_unit"],
    ledger: _Ledger,
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]:
    from sympy import QQ, Symbol
    from sympy.polys.rings import ring

    chosen = left if kind.startswith("left") else right
    source_terms = chosen.polynomial.terms
    bits = max(
        (
            abs(t.coefficient.num).bit_length() + t.coefficient.den.bit_length()
            for p in (left, right)
            for t in p.polynomial.terms
        ),
        default=1,
    )
    count = len(left.polynomial.terms) + len(right.polynomial.terms)
    _reserve(
        ledger,
        scalar_bits=4 * bits + 8,
        operations=32 * (count + 1),
        cells=16 * (count + 1),
        digits=(count + 2 * len(source_terms) + 4) * 2 * _digits_from_bits(2 * bits),
    )
    r = ring((Symbol(left.variables[0]),), QQ)[0]
    source = _ring_source(r, chosen)
    multiplier = r.ground_new(1 / source.LC)
    common = r.one if kind.endswith("unit") else source.monic()
    s, t = (multiplier, r.zero) if kind.startswith("left") else (r.zero, multiplier)
    return (
        _value(s, left.variables),
        _value(t, left.variables),
        _value(common, left.variables),
    )


def bounded_bezout(
    left: RationalPolynomial, right: RationalPolynomial
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]:
    """Admit and compute both witnesses and the monic gcd without opaque xgcd.

    For primitive integral A,B of degrees m,n, every minor of their Sylvester
    matrix is bounded by K=||A||1**n * ||B||1**m (Hadamard, column/row norms).
    Adjoining identity columns does not enlarge this bound: expand a minor
    along those columns. Fraction-free reduction therefore retains integers
    <=K, while its sparse cancellation temporaries are <=(m+n+1)*K**2.
    Source contents scale the two witnesses separately after the solve.
    """
    require_bezout_source_shape(left)
    require_bezout_source_shape(right)
    ledger = _Ledger()
    first, second = _source_terms(left, ledger), _source_terms(right, ledger)
    if not first:
        return _simple(left, right, "right", ledger)
    if not second:
        return _simple(left, right, "left", ledger)
    if second[0].exponents[0] == 0:
        return _simple(left, right, "right_unit", ledger)
    if first[0].exponents[0] == 0:
        return _simple(left, right, "left_unit", ledger)
    if _proportional(first, second, ledger):
        return _simple(left, right, "right", ledger)
    a, b = _primitive(first, ledger), _primitive(second, ledger)
    m, n = a.degree, b.degree
    if min(m, n) > 1:
        reduced = _short_reduction(a, b, left, right, ledger)
        if reduced is not None:
            return reduced
    # ceil(log2(norm)); norm1 contributes no artificial degree tax.
    h = n * (a.norm - 1).bit_length() + m * (b.norm - 1).bit_length()
    scales = (1 / a.scale, 1 / b.scale)
    scalar_scale = max(max(abs(s.numerator), s.denominator) for s in scales)
    if h + scalar_scale.bit_length() > _CANONICAL_LIMIT.bit_length() + 1:
        _reject(
            "coefficient_height",
            "Bézout witnesses can exceed the canonical scalar carrier",
        )
    bound = ledger.multiply(1 << h, scalar_scale)
    if bound >= _CANONICAL_LIMIT:
        _reject(
            "coefficient_height",
            "Bézout witnesses can exceed the canonical scalar carrier",
        )
    size = m + n
    raw_bits = 2 * h + (size + 1).bit_length() + 2 * scalar_scale.bit_length() + 8
    source_digits = sum(
        _digits_from_bits(abs(t.coefficient.num).bit_length())
        + _digits_from_bits(t.coefficient.den.bit_length())
        for p in (left, right)
        for t in p.polynomial.terms
    )
    result_digits = source_digits + 2 * (size + min(m, n) + 2) * _digits_from_bits(
        bound.bit_length()
    )
    if min(m, n) == 1:
        _reserve(
            ledger,
            scalar_bits=raw_bits,
            operations=64 * (max(m, n) + 1),
            cells=32 * (max(m, n) + 1),
            digits=result_digits,
        )
        ledger.charge((max(m, n) + 1) ** 2)
        return _linear(a, b, left.variables)
    _reserve(
        ledger,
        scalar_bits=raw_bits,
        operations=12 * size**3 + 16 * size**2 + 16 * size,
        cells=8 * size**2 + 32 * size,
        digits=result_digits,
    )
    return _fraction_free(a, b, left.variables)


def _primitive_ring(ring: Any, value: _Primitive) -> Any:
    return ring.from_dict({(e,): ring.domain(c) for e, c in value.terms})


def _linear_polynomials(
    a: _Primitive, b: _Primitive, axis: tuple[str, ...], ring_value: Any = None
) -> tuple[Any, Any, Any]:
    from sympy import QQ, Symbol
    from sympy.polys.rings import ring

    r = ring_value if ring_value is not None else ring((Symbol(axis[0]),), QQ)[0]
    f, g = _primitive_ring(r, a), _primitive_ring(r, b)
    swap = a.degree < b.degree
    high, low = (g, f) if swap else (f, g)
    request_checkpoint("before maintained linear Bézout division")
    quotient, remainder = high.div(low)
    if remainder:
        scalar = remainder[(0,)]
        s, t, common = r.ground_new(1 / scalar), -quotient.quo_ground(scalar), r.one
    else:
        s, t, common = r.zero, r.ground_new(1 / low.LC), low.monic()
    if swap:
        s, t = t, s
    if s * f + t * g != common:
        raise RuntimeError("linear Bézout reconstruction failed")
    s = s.quo_ground(QQ(a.scale.numerator, a.scale.denominator))
    t = t.quo_ground(QQ(b.scale.numerator, b.scale.denominator))
    request_checkpoint("after maintained linear Bézout division")
    return s, t, common


def _linear(
    a: _Primitive, b: _Primitive, axis: tuple[str, ...]
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]:
    s, t, common = _linear_polynomials(a, b, axis)
    return _value(s, axis), _value(t, axis), _value(common, axis)


def _fraction_free(
    a: _Primitive, b: _Primitive, axis: tuple[str, ...]
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]:
    from sympy import QQ, ZZ, Symbol
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.rings import ring

    m, n = a.degree, b.degree
    size = m + n
    rows = {}
    for index, (source, shift) in enumerate(
        (*((a, j) for j in range(n)), *((b, j) for j in range(m)))
    ):
        rows[index] = {size - 1 - e - shift: ZZ(c) for e, c in source.terms}
        rows[index][size + index] = ZZ.one
    request_checkpoint("before maintained fraction-free Bézout reduction")
    reduced, denominator, pivots = DomainMatrix(rows, (size, 2 * size), ZZ).rref_den(
        method="FF"
    )
    request_checkpoint("after maintained fraction-free Bézout reduction")
    selected = max(i for i, pivot in enumerate(pivots) if pivot < size)
    row = reduced.rep[selected]
    r = ring((Symbol(axis[0]),), QQ)[0]
    common = r.from_dict(
        {
            (size - 1 - column,): QQ(value, denominator)
            for column, value in row.items()
            if column < size
        }
    )
    s = r.from_dict(
        {
            (j,): QQ(row.get(size + j, 0), denominator)
            for j in range(n)
            if row.get(size + j, 0)
        }
    )
    t = r.from_dict(
        {
            (j,): QQ(row.get(size + n + j, 0), denominator)
            for j in range(m)
            if row.get(size + n + j, 0)
        }
    )
    # Witness denominators divide one common D<=K. Every replay numerator
    # over D is bounded by (N+1)*K*max(||A||1,||B||1), hence (N+1)*K**2.
    if s * _primitive_ring(r, a) + t * _primitive_ring(r, b) != common:
        raise RuntimeError("fraction-free Bézout reconstruction failed")
    s = s.quo_ground(QQ(a.scale.numerator, a.scale.denominator))
    t = t.quo_ground(QQ(b.scale.numerator, b.scale.denominator))
    return _value(s, axis), _value(t, axis), _value(common, axis)


def _short_reduction(
    a: _Primitive,
    b: _Primitive,
    left: RationalPolynomial,
    right: RationalPolynomial,
    ledger: _Ledger,
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial] | None:
    """Recognize one support-proved reduction to a linear or constant remainder.

    R=bLead*A-aLead*x**delta*B has only its two possible low coefficients.
    If R=cR*Rprim and inner witnesses have shared denominator D<=Kinner,
    composing gives bLead*v/(cR*D) and (cR*u-aLead*x**delta*v)/(cR*D).
    This prices the actual shared denominator rather than a fictitious square.
    Original source content scales are applied only after this composition.
    """

    swap = a.degree < b.degree
    high, low = (b, a) if swap else (a, b)
    delta = high.degree - low.degree
    if any(e > 1 for e, _ in high.terms[1:]) or any(
        e + delta > 1 for e, _ in low.terms[1:]
    ):
        return None
    leading_a, leading_b = high.terms[0][1], low.terms[0][1]
    coefficients = {e: ledger.multiply(leading_b, c) for e, c in high.terms[1:]}
    for e, c in low.terms[1:]:
        product = ledger.multiply(leading_a, c)
        current = coefficients.get(e + delta, 0)
        ledger.charge(1, max(abs(current).bit_length(), abs(product).bit_length()) + 1)
        coefficients[e + delta] = current - product
    terms = tuple(sorted(((e, c) for e, c in coefficients.items() if c), reverse=True))
    if not terms:
        return _simple(left, right, "left" if swap else "right", ledger)
    content = 0
    for _, coefficient in terms:
        ledger.charge(1, max(content.bit_length(), abs(coefficient).bit_length()))
        content = gcd(content, coefficient)
    ledger.charge(3 * len(terms) + 1, max(abs(c).bit_length() for _, c in terms) + 2)
    primitive_terms = tuple((e, c // content) for e, c in terms)
    remainder = _Primitive(
        primitive_terms, Fraction(content), sum(abs(c) for _, c in primitive_terms)
    )
    h = (
        0
        if remainder.degree == 0
        else (low.norm - 1).bit_length()
        + low.degree * (remainder.norm - 1).bit_length()
    )
    ledger.charge(1, max(content.bit_length(), abs(leading_a).bit_length()) + 1)
    outer_factor = max(content + abs(leading_a), abs(leading_b), content)
    scale = max(max(s.numerator, s.denominator) for s in (1 / a.scale, 1 / b.scale))
    result_bits = h + outer_factor.bit_length() + scale.bit_length()
    if result_bits > _CANONICAL_LIMIT.bit_length() + 2:
        return None
    outer_bound = ledger.multiply(1 << h, outer_factor)
    bound = ledger.multiply(outer_bound, scale)
    if bound >= _CANONICAL_LIMIT:
        return None
    size = a.degree + b.degree
    # Small witnesses do not imply small replay inputs. Include the primitive
    # source norms independently, including for A=B+1 with large coefficients.
    replay_bits = (
        h
        + outer_factor.bit_length()
        + max(a.norm.bit_length(), b.norm.bit_length())
        + (size + 1).bit_length()
    )
    scalar_bits = 2 * max(result_bits, replay_bits, h + content.bit_length()) + 8
    operations, cells = 128 * (size + 1), 64 * (size + 1)
    digits = sum(
        _digits_from_bits(abs(t.coefficient.num).bit_length())
        + _digits_from_bits(t.coefficient.den.bit_length())
        for p in (left, right)
        for t in p.polynomial.terms
    ) + 2 * (size + 3) * _digits_from_bits(bound.bit_length())
    if (
        scalar_bits > MAX_BEZOUT_PRIVATE_BITS
        or ledger.work + operations * max(1, (scalar_bits + 63) // 64) ** 2
        > MAX_BEZOUT_WORK
        or ledger.coefficient_bits + cells * scalar_bits
        > MAX_BEZOUT_COEFFICIENT_STORAGE_BITS
        or digits > MAX_BEZOUT_RESULT_DIGITS
    ):
        return None
    _reserve(
        ledger,
        scalar_bits=scalar_bits,
        operations=operations,
        cells=cells,
        digits=digits,
    )
    return _execute_short_reduction(
        a, b, remainder, swap, delta, leading_a, leading_b, left.variables
    )


def _execute_short_reduction(
    a: _Primitive,
    b: _Primitive,
    remainder: _Primitive,
    swap: bool,
    delta: int,
    leading_a: int,
    leading_b: int,
    axis: tuple[str, ...],
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]:
    from sympy import QQ, Symbol
    from sympy.polys.rings import ring

    r = ring((Symbol(axis[0]),), QQ)[0]
    low = a if swap else b
    if remainder.degree == 0:
        scalar = remainder.scale * remainder.terms[0][1]
        u, v, common = (
            r.zero,
            r.ground_new(QQ(scalar.denominator, scalar.numerator)),
            r.one,
        )
    else:
        u, v, common = _linear_polynomials(
            _Primitive(low.terms, Fraction(1), low.norm), remainder, axis, r
        )
    s = v * leading_b
    t = u - r.term_new((delta,), QQ(leading_a)) * v
    if swap:
        s, t = t, s
    if s * _primitive_ring(r, a) + t * _primitive_ring(r, b) != common:
        raise RuntimeError("short-reduction Bézout reconstruction failed")
    s = s.quo_ground(QQ(a.scale.numerator, a.scale.denominator))
    t = t.quo_ground(QQ(b.scale.numerator, b.scale.denominator))
    request_checkpoint("after maintained short Bézout reduction")
    return _value(s, axis), _value(t, axis), _value(common, axis)
