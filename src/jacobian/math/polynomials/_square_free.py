"""Owner-local bounded plans and execution for canonical square-free values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb, factorial, gcd, prod
from typing import Any, Literal, NoReturn

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._models import (
    PolynomialSquareFreeDecompositionResult,
    PolynomialSquareFreeFactor,
)
from jacobian.math.polynomials._square_free_bounds import (
    Envelope,
    IntegerSource,
    canonical_components,
    canonical_height,
    clear_source,
    clearing_envelope,
    decomposition_envelope,
    divisor_bits,
    refuse,
    source_storage,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


@dataclass(frozen=True)
class Axis:
    index: int
    integer: IntegerSource
    degree: int
    kind: Literal["monomial", "binomial", "general"]
    terms: dict[int, Fraction]


@dataclass(frozen=True)
class Plan:
    source: RationalPolynomial
    terms: dict[tuple[int, ...], Fraction]
    kind: Literal["constant", "compact", "squarefree", "power", "separable", "general"]
    dilation: int = 1
    multiplicity: int = 1
    axes: tuple[Axis, ...] = ()
    integer: IntegerSource | None = None
    normalized: dict[tuple[int, ...], Fraction] | None = None


def _domain(message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("polynomial",), code="polynomial.square_free_source", message=message
    )


def _source_terms(source: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    if not isinstance(source, RationalPolynomial) or not isinstance(
        getattr(source, "polynomial", None), SparseRationalPolynomial
    ):
        _domain("square-free source must be a canonical RationalPolynomial")
    if (
        not isinstance(getattr(source, "variables", None), tuple)
        or getattr(source, "domain", None) != "QQ"
    ):
        _domain("square-free source must retain its canonical QQ ring")
    if len(source.variables) > MAX_POLYNOMIAL_VARIABLES or any(
        type(v) is not str for v in source.variables
    ):
        _domain("square-free source variables must be canonical named axes")
    terms = getattr(source.polynomial, "terms", None)
    if not isinstance(terms, tuple):
        _domain("square-free terms must be a canonical tuple")
    if len(terms) > MAX_POLYNOMIAL_TERMS:
        refuse(
            "source_terms",
            "square-free source exceeds the 4096-term canonical envelope",
        )
    previous = None
    values = {}
    for term in terms:
        if not isinstance(term, RationalPolynomialTerm) or not isinstance(
            term.exponents, tuple
        ):
            _domain("square-free terms must retain canonical exponent tuples")
        if len(term.exponents) != len(source.variables) or any(
            type(e) is not int or not 0 <= e <= MAX_POLYNOMIAL_EXPONENT
            for e in term.exponents
        ):
            _domain("square-free exponents must match the canonical source axes")
        if previous is not None and previous <= term.exponents:
            _domain(
                "square-free terms must be distinct and in descending exponent order"
            )
        previous = term.exponents
        q = getattr(term, "coefficient", None)
        if (
            not isinstance(q, CanonicalRational)
            or type(getattr(q, "num", None)) is not int
            or type(getattr(q, "den", None)) is not int
            or q.num == 0
            or q.den <= 0
        ):
            _domain("square-free coefficients must be nonzero reduced exact rationals")
        if (
            max(abs(q.num).bit_length(), q.den.bit_length())
            > 4 * MAX_CANONICAL_RATIONAL_DIGITS
        ):
            _domain("square-free coefficient exceeds its canonical scalar carrier")
        try:
            require_bounded_rational(
                q,
                max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="square-free coefficient",
            )
        except ValueError:
            _domain("square-free coefficient exceeds its canonical scalar carrier")
        if gcd(abs(q.num), q.den) != 1:
            _domain("square-free coefficients must be reduced exact rationals")
        values[term.exponents] = q.as_fraction()
    # Validate domain and ordered variable names without serializing scalars or
    # trusting model_copy to have replayed the source's structural invariants.
    try:
        RationalPolynomial(
            domain=source.domain,
            variables=source.variables,
            polynomial=source.polynomial,
        )
    except ValueError:
        _domain("square-free source ring or axes are not canonical")
    return values


def _height(values: Any) -> int:
    return max(
        (
            max(abs(q.numerator).bit_length(), q.denominator.bit_length())
            for q in values
        ),
        default=1,
    )


def _normalize(
    terms: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    leading = terms[max(terms)]
    raw_bits = _height(terms.values()) + _height((leading,)) + 1
    Envelope(
        work=32 * len(terms) * (len(next(iter(terms))) + 1),
        bits=raw_bits,
        storage=8 * len(terms) * raw_bits,
    ).checked()
    result = {powers: value / leading for powers, value in terms.items()}
    canonical_components(result.values())
    return result


def _obviously_square_free(
    terms: dict[tuple[int, ...], Fraction], degrees: tuple[int, ...]
) -> bool:
    # A repeated nonconstant factor forces degree >=2 in some axis.
    if max(degrees) <= 1:
        return True
    # A nonconstant monomial plus a nonzero constant is coprime to at
    # least one of its monomial partial derivatives in characteristic zero.
    if len(terms) == 2 and (0,) * len(degrees) in terms:
        return True
    # If one partial derivative is a monomial and the source is divisible
    # by none of that monomial's variables, their gcd is one. A repeated
    # factor would divide every partial derivative, so the source is square-free.
    minimum = tuple(min(powers[i] for powers in terms) for i in range(len(degrees)))
    for axis in range(len(degrees)):
        active = [powers for powers in terms if powers[axis]]
        if len(active) == 1 and all(
            exponent - (index == axis) == 0 or minimum[index] == 0
            for index, exponent in enumerate(active[0])
        ):
            return True
    return False


def _recognition_envelope(
    terms: dict[tuple[int, ...], Fraction], degrees: tuple[int, ...]
) -> Envelope:
    n, axes = max(degrees), len(degrees)
    raw = 2 * _height(terms.values()) + n.bit_length() + 3
    bits = (n + axes + 2) * raw + factorial(n).bit_length() + 2
    return Envelope(
        work=512 * len(terms) * (axes + 1) * (n + 1),
        bits=bits,
        storage=max(
            8 * len(terms) * raw,
            source_storage(terms)
            + 16 * (axes + 1) * raw
            + 8 * bits
            + 8 * (n + 1) * factorial(n).bit_length(),
        ),
    ).checked()


def _with_recognition(envelope: Envelope, recognition: Envelope) -> None:
    Envelope(
        envelope.work + recognition.work,
        max(envelope.bits, recognition.bits),
        max(envelope.storage, recognition.storage),
    ).checked()


def _affine_power(
    terms: dict[tuple[int, ...], Fraction],
) -> tuple[dict[tuple[int, ...], Fraction], int] | None:
    """Prove an affine power by complete exact multinomial coefficients.

    None is a proved mismatch, never an aborted or resource-limited probe.
    Thus a quadratic mismatch proves square-freeness: a repeated nonconstant
    factor of a degree-two polynomial would have to be an affine square.
    """
    leading_powers = max(terms)
    active = [i for i, e in enumerate(leading_powers) if e]
    if len(active) != 1:
        return None
    pivot = active[0]
    multiplicity = leading_powers[pivot]
    if max(map(sum, terms)) != multiplicity:
        return None
    leading = terms[leading_powers]
    width = len(leading_powers)
    base = [0] * width
    base[pivot] = multiplicity - 1
    zero = (0,) * width
    linear = {tuple(1 if i == pivot else 0 for i in range(width)): Fraction(1)}
    divisor = multiplicity * leading
    constant = terms.get(tuple(base), Fraction()) / divisor
    if constant:
        linear[zero] = constant
    coefficients = {pivot: Fraction(1)}
    for axis in range(width):
        if axis == pivot:
            continue
        cross_powers = list(base)
        cross_powers[axis] += 1
        coefficient = terms.get(tuple(cross_powers), Fraction()) / divisor
        if coefficient:
            coefficients[axis] = coefficient
            linear[tuple(1 if i == axis else 0 for i in range(width))] = coefficient
    slots = len(coefficients) + bool(constant)
    if comb(multiplicity + slots - 1, slots - 1) != len(terms):
        return None
    factorials = [factorial(i) for i in range(multiplicity + 1)]
    for powers, value in terms.items():
        residual = multiplicity - sum(powers)
        if residual < 0 or (residual and not constant):
            return None
        denominator = factorials[residual]
        expected = leading
        for axis, exponent in enumerate(powers):
            if exponent and axis not in coefficients:
                return None
            denominator *= factorials[exponent]
            if exponent:
                expected *= coefficients[axis] ** exponent
        if residual:
            expected *= constant**residual
        expected *= factorials[multiplicity] // denominator
        if expected != value:
            return None
    canonical_components(linear.values())
    return linear, multiplicity


def _axes(
    terms: dict[tuple[int, ...], Fraction], degrees: tuple[int, ...]
) -> tuple[Axis, ...] | None:
    projections = tuple(
        tuple(sorted({powers[i] for powers in terms})) for i in range(len(degrees))
    )
    if prod(map(len, projections)) != len(terms):
        return None
    leading_powers = tuple(row[-1] for row in projections)
    if leading_powers not in terms:
        return None
    leading = terms[leading_powers]
    normalized: list[dict[int, Fraction]] = []
    slices: list[dict[int, Fraction]] = []
    raw_bits = 2 * _height(terms.values()) + 1
    Envelope(
        work=16 * len(terms) * (len(degrees) + 1),
        bits=(len(degrees) + 1) * raw_bits,
        storage=8 * len(terms) * raw_bits,
    ).checked()
    for axis, support in enumerate(projections):
        raw = {}
        for exponent in support:
            slice_powers = list(leading_powers)
            slice_powers[axis] = exponent
            if tuple(slice_powers) not in terms:
                return None
            raw[exponent] = terms[tuple(slice_powers)]
        slices.append(raw)
        normalized.append({e: q / leading for e, q in raw.items()})
    for powers in product(*projections):
        if powers not in terms:
            return None
        coefficient = leading
        for axis, exponent in enumerate(powers):
            coefficient *= normalized[axis][exponent]
        if coefficient != terms[powers]:
            return None
    clearings = [
        clearing_envelope({(e,): q for e, q in raw.items()})
        for degree, raw in zip(degrees, slices, strict=True)
        if degree
    ]
    Envelope(
        work=sum(p.work for p in clearings),
        bits=max(p.bits for p in clearings),
        storage=source_storage(terms) + sum(p.storage for p in clearings),
    ).checked()
    axes = []
    for index, (degree, raw, monic) in enumerate(
        zip(degrees, slices, normalized, strict=True)
    ):
        if degree == 0:
            continue
        kind: Literal["monomial", "binomial", "general"] = (
            "monomial"
            if len(raw) == 1
            else "binomial"
            if len(raw) == 2 and 0 in raw
            else "general"
        )
        integer = clear_source({(e,): q for e, q in raw.items()})
        axes.append(Axis(index, integer, degree, kind, monic))
    return tuple(axes)


def _admit_axes(
    axes: tuple[Axis, ...],
    terms: dict[tuple[int, ...], Fraction],
    recognition: Envelope,
) -> None:
    source_size = len(terms)
    support = prod(
        1 if a.kind == "monomial" else 2 if a.kind == "binomial" else a.degree + 1
        for a in axes
    )
    if support > MAX_POLYNOMIAL_TERMS:
        refuse(
            "grouped_support",
            f"proved grouped square-free support {support} exceeds 4096",
        )
    height = sum(
        1
        if a.kind == "monomial"
        else _height(a.terms.values())
        if a.kind == "binomial"
        else divisor_bits((a.degree,), a.integer.height)
        for a in axes
    )
    canonical_height(height)
    profiles = [
        decomposition_envelope((a.degree,), a.integer.height)
        for a in axes
        if a.kind == "general"
    ]
    envelope = Envelope(
        work=sum(p.work for p in profiles)
        + sum(a.integer.envelope.work for a in axes)
        + 128 * (source_size + support + 63) * (len(axes) + 1),
        bits=max(
            [
                height,
                _height(terms.values()),
                *[a.integer.envelope.bits for a in axes],
                *[p.bits for p in profiles],
            ]
        ),
        storage=source_storage(terms)
        + sum(a.integer.envelope.storage for a in axes)
        + max([0, *[p.storage for p in profiles]])
        + 8 * (support + 63) * max(1, height),
    ).checked()

    _with_recognition(envelope, recognition)


def admit(source: RationalPolynomial) -> Plan:
    terms = _source_terms(source)
    degrees = tuple(
        max((e[i] for e in terms), default=0) for i in range(len(source.variables))
    )
    if not terms or not any(degrees):
        height = _height(terms.values())
        Envelope(
            work=64, bits=height, storage=source_storage(terms) + 16 * height
        ).checked()
        return Plan(source, terms, "constant")
    if len(terms) == 1 and max(degrees) > 64:
        refuse(
            "multiplicity_budget",
            "monomial factor multiplicity exceeds the square-free limit of 64",
        )
    if len(source.variables) == 1 and len(terms) == 2 and (0,) in terms:
        # a*x^n+b has no common root with a*n*x^(n-1) when a*b != 0.
        # The inherited degree-one backend/lift remains valid at any admitted
        # source exponent. Admit the actual monic ratio before expansion so
        # a previously emitted normalized factor remains a valid source.
        normalized = _normalize(terms)
        height = _height(terms.values())
        Envelope(work=1024, bits=3 * height + 4, storage=32 * height).checked()
        return Plan(
            source, terms, "compact", dilation=degrees[0], normalized=normalized
        )
    if max(degrees) > 64:
        refuse(
            "general_degree",
            "general square-free source exponent exceeds the 64-degree envelope",
        )
    if _obviously_square_free(terms, degrees):
        return Plan(source, terms, "squarefree", normalized=_normalize(terms))
    recognition = _recognition_envelope(terms, degrees)
    power = _affine_power(terms)
    if power is not None:
        normalized, multiplicity = power
        return Plan(
            source, terms, "power", multiplicity=multiplicity, normalized=normalized
        )
    if max(map(sum, terms)) == 2:
        normalized = _normalize(terms)
        _with_recognition(
            Envelope(
                work=32 * len(terms) * (len(degrees) + 1),
                bits=2 * _height(terms.values()) + 1,
                storage=8 * len(terms) * (2 * _height(terms.values()) + 1),
            ),
            recognition,
        )
        return Plan(source, terms, "squarefree", normalized=normalized)
    axes = _axes(terms, degrees)
    if axes is not None:
        _admit_axes(axes, terms, recognition)
        return Plan(source, terms, "separable", axes=axes)
    box = prod(d + 1 for d in degrees)
    if box > MAX_POLYNOMIAL_TERMS:
        refuse(
            "grouped_support", f"proved grouped square-free support {box} exceeds 4096"
        )
    integer = clear_source(terms)
    profile = decomposition_envelope(degrees, integer.height)
    output_height = divisor_bits(degrees, integer.height)
    reconstruction_bits = 2 * (
        _height((integer.scale,)) + 2 * output_height + box.bit_length() + 2
    )
    envelope = Envelope(
        profile.work + integer.envelope.work + 8 * len(terms),
        max(profile.bits, integer.envelope.bits, reconstruction_bits),
        max(
            integer.envelope.storage,
            profile.storage
            + source_storage(terms)
            + 16 * box * reconstruction_bits
            + 8 * (box + 63) * output_height,
        ),
    ).checked()
    _with_recognition(envelope, recognition)
    return Plan(source, terms, "general", integer=integer)


def _polynomial(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    exponents=e, coefficient=CanonicalRational.from_fraction(q)
                )
                for e, q in sorted(terms.items(), reverse=True)
                if q
            )
        ),
    )


def _univariate_factors(
    axis: Axis, variable: str
) -> list[tuple[dict[int, Fraction], int]]:
    if axis.kind == "monomial":
        return [({1: Fraction(1)}, axis.degree)]
    if axis.kind == "binomial":
        return [(axis.terms, 1)]
    from sympy import QQ, ZZ, Poly, Rational, Symbol

    from jacobian.math.polynomials._square_free_kernel import grouped_factors
    from jacobian.math.polynomials._sympy import _monic_decomposition

    generator = Symbol(variable)
    integer = Poly.from_dict(axis.integer.terms, generator, domain=ZZ)
    monic = integer.monic()
    raw = grouped_factors(integer)
    coefficient, factors, _ = _monic_decomposition(
        monic,
        (Rational(1, int(integer.LC())), raw),
        label="controlled univariate square-free decomposition",
    )
    if coefficient != 1:
        raise RuntimeError("monic axis square-free decomposition changed its scalar")
    return [
        (
            {
                int(e[0]): Fraction(int(q.p), int(q.q))
                for e, q in factor.set_domain(QQ).terms()
            },
            m,
        )
        for factor, m in factors
    ]


def compute(plan: Plan) -> PolynomialSquareFreeDecompositionResult:
    source, terms = plan.source, plan.terms
    leading = terms[max(terms)] if terms else Fraction()
    factors: list[PolynomialSquareFreeFactor] = []
    if plan.kind in {"squarefree", "power"}:
        if plan.normalized is None:
            raise RuntimeError(
                "square-free normalization plan is missing its admitted value"
            )
        factors.append(
            PolynomialSquareFreeFactor(
                factor=_polynomial(source.variables, plan.normalized),
                multiplicity=plan.multiplicity,
            )
        )
    elif plan.kind == "separable":
        groups: dict[int, dict[tuple[int, ...], Fraction]] = {}
        zero = (0,) * len(source.variables)
        for axis in plan.axes:
            for factor, multiplicity in _univariate_factors(
                axis, source.variables[axis.index]
            ):
                previous = groups.get(multiplicity, {zero: Fraction(1)})
                combined = {}
                for (powers, q), (exponent, c) in product(
                    previous.items(), factor.items()
                ):
                    target = list(powers)
                    target[axis.index] = exponent
                    combined[tuple(target)] = q * c
                groups[multiplicity] = combined
        factors = [
            PolynomialSquareFreeFactor(
                factor=_polynomial(source.variables, group), multiplicity=m
            )
            for m, group in sorted(groups.items())
        ]
    elif plan.kind == "general":
        factors = _general_factors(plan)
    elif plan.kind != "constant":
        raise RuntimeError("compact square-free plan must use the degree-one backend")
    return PolynomialSquareFreeDecompositionResult._from_kernel(
        polynomial=source,
        coefficient=CanonicalRational.from_fraction(leading),
        factors=tuple(factors),
        reconstructed=source,
    )


def _general_factors(plan: Plan) -> list[PolynomialSquareFreeFactor]:
    from sympy import QQ, ZZ, Poly, Rational, Symbol

    from jacobian.math.polynomials._square_free_kernel import grouped_factors
    from jacobian.math.polynomials._sympy import _monic_decomposition

    if plan.integer is None:
        raise RuntimeError("square-free generic plan is missing its integer source")
    active = tuple(
        i for i in range(len(plan.source.variables)) if any(e[i] for e in plan.terms)
    )
    generators = tuple(Symbol(plan.source.variables[i]) for i in active)
    integer = Poly.from_dict(
        {tuple(e[i] for i in active): c for e, c in plan.integer.terms.items()},
        *generators,
        domain=ZZ,
    )
    scale = plan.integer.scale
    rational_source = integer.set_domain(QQ) * Rational(
        scale.numerator, scale.denominator
    )
    _, factors, _ = _monic_decomposition(
        rational_source,
        (Rational(scale.numerator, scale.denominator), grouped_factors(integer)),
        label="controlled square-free decomposition",
    )
    records = []
    for factor, multiplicity in factors:
        values = {}
        for powers, q in factor.set_domain(QQ).terms():
            full = [0] * len(plan.source.variables)
            for index, exponent in zip(active, powers, strict=True):
                full[index] = int(exponent)
            values[tuple(full)] = Fraction(int(q.p), int(q.q))
        records.append(
            PolynomialSquareFreeFactor(
                factor=_polynomial(plan.source.variables, values),
                multiplicity=multiplicity,
            )
        )
    return records
