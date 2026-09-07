"""Exact common weak-interlacing admission and SymPy kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from typing import TYPE_CHECKING

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RationalIsolatingInterval,
    RealAlgebraicValue,
    _compare_admitted_real_algebraic,
)
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    MAX_COMMON_INTERLACING_FACTOR_DEGREE,
    MAX_COMMON_INTERLACING_FAMILY_SIZE,
    MAX_COMMON_INTERLACING_INPUT_DIGITS,
    MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    MAX_COMMON_INTERLACING_SOURCE_TERMS,
    MAX_COMMON_INTERLACING_TOTAL_DEGREE,
    MAX_COMMON_INTERLACING_TOTAL_TERMS,
    CommonInterlacingDoesNotExist,
    CommonInterlacingExists,
    CommonInterlacingGap,
    CommonInterlacingOutcome,
    CommonInterlacingProfile,
    EmptyGapObstruction,
    LabelledRationalPolynomial,
    NonRealRootObstruction,
    PolynomialRealRoot,
    PolynomialRootReference,
    SourceRootProfile,
)

if TYPE_CHECKING:
    from sympy import Poly
    from sympy.core.numbers import Rational as SympyRational

# These budgets cover separately the mandatory exact phases.  Source height is
# measured after clearing rational denominators and taking primitive content.
# The factor and root-separation bounds below are then properties of that one
# primitive ZZ[x] source, rather than guesses based only on wire size.
MAX_COMMON_INTERLACING_PRIMITIVE_HEIGHT_DIGITS = 256
MAX_COMMON_INTERLACING_CLEARING_DENOMINATOR_DIGITS = (
    MAX_COMMON_INTERLACING_SOURCE_TERMS * MAX_COMMON_INTERLACING_INPUT_DIGITS
)
MAX_COMMON_INTERLACING_CLEARING_INTERMEDIATE_DIGITS = (
    MAX_COMMON_INTERLACING_CLEARING_DENOMINATOR_DIGITS
    + MAX_COMMON_INTERLACING_INPUT_DIGITS
)
MAX_COMMON_INTERLACING_FACTORIZATION_WORK = 10_000_000
MAX_COMMON_INTERLACING_ISOLATION_WORK = 100_000_000
MAX_COMMON_INTERLACING_TOTAL_FACTORS = 128
MAX_COMMON_INTERLACING_FACTOR_ROOT_CHECKS = 2_048
MAX_COMMON_INTERLACING_COMPARISONS = 512


@dataclass(frozen=True, slots=True)
class _FactorPlan:
    polynomial: Poly
    multiplicity: int
    canonical_coefficients: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SourcePlan:
    factors: tuple[_FactorPlan, ...]
    squarefree_product: Poly
    squarefree_height_digits: int
    isolation_endpoint_digits: int


@dataclass(frozen=True, slots=True)
class _PrimitiveSourcePlan:
    coefficients: tuple[int, ...]
    degree: int
    height_digits: int
    term_count: int


@dataclass(frozen=True, slots=True)
class _CommonInterlacingPlan:
    family: tuple[LabelledRationalPolynomial, ...]
    common_degree: int
    sources: tuple[_SourcePlan, ...]


def _reject(
    location: tuple[str | int, ...],
    reason: str,
    message: str,
) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"polynomial.common_interlacing_{reason}",
        message=message,
    )


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(value).lstrip("-"))


def _coefficient_digits(value: CanonicalRational) -> int:
    return max(
        len(format_canonical_integer(abs(value.num))),
        len(format_canonical_integer(value.den)),
    )


def _dense_primitive_coefficients(
    source: LabelledRationalPolynomial,
) -> tuple[tuple[int, ...], int, int, int]:
    terms = source.polynomial.polynomial.terms
    degree = terms[0].exponents[0] if terms else 0
    dense = [Fraction(0)] * (degree + 1)
    for term in terms:
        dense[degree - term.exponents[0]] = term.coefficient.as_fraction()

    common_denominator = 1
    for coefficient in dense:
        common_denominator = lcm(common_denominator, coefficient.denominator)
    integer_coefficients: list[int] = [
        int(coefficient.numerator)
        * (common_denominator // int(coefficient.denominator))
        for coefficient in dense
    ]
    denominator_digits = _integer_digits(common_denominator)
    intermediate_digits = max(
        _integer_digits(coefficient) for coefficient in integer_coefficients
    )
    content = 0
    for integer_coefficient in integer_coefficients:
        content = gcd(content, abs(integer_coefficient))
    primitive = tuple(
        integer_coefficient // content for integer_coefficient in integer_coefficients
    )
    # Monicity makes this unreachable, but keep the internal ZZ carrier canonical.
    if primitive[0] < 0:
        primitive = tuple(-coefficient for coefficient in primitive)
    height_digits = max(_integer_digits(coefficient) for coefficient in primitive)
    return (
        primitive,
        height_digits,
        denominator_digits,
        intermediate_digits,
    )


def _canonical_factor(factor: Poly) -> tuple[int, ...]:
    coefficients = [int(coefficient) for coefficient in factor.all_coeffs()]
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    coefficients = [coefficient // content for coefficient in coefficients]
    if coefficients[0] < 0:
        coefficients = [-coefficient for coefficient in coefficients]
    return tuple(coefficients)


def _factor_digit_bound(degree: int, height_digits: int) -> int:
    """Landau--Mignotte decimal bound for every primitive factor coefficient."""

    multiplier_digits = _integer_digits((1 << degree) * (degree + 1))
    return height_digits + multiplier_digits


def _isolation_endpoint_digit_bound(degree: int, height_digits: int) -> int:
    """Conservative Mignotte separation/grid bound for rational endpoints.

    For a square-free primitive degree-n integer polynomial, Mignotte's root
    separation inequality and Cauchy's root bound admit an isolating rational
    grid with this many decimal digits in each numerator and denominator.  The
    factor two reserves both sides of each isolating cell and deliberately
    overestimates the logarithmic degree terms.
    """

    degree_digits = _integer_digits(max(degree, 1))
    return 2 * degree * (height_digits + degree_digits + 2) + height_digits + 8


def _preflight_source(
    source: LabelledRationalPolynomial,
    *,
    source_index: int,
    common_variables: tuple[str, ...],
) -> _PrimitiveSourcePlan:
    polynomial = source.polynomial
    location = ("family", source_index, "polynomial")
    if polynomial.variables != common_variables:
        _reject(
            (*location, "variables"),
            "source_ring",
            "common interlacing sources must use the same named variable",
        )
    terms = polynomial.polynomial.terms
    if len(terms) > MAX_COMMON_INTERLACING_SOURCE_TERMS:
        _reject(
            (*location, "polynomial", "terms"),
            "term_count",
            "a common interlacing source exceeds the "
            f"{MAX_COMMON_INTERLACING_SOURCE_TERMS}-term bound",
        )
    if any(
        _coefficient_digits(term.coefficient) > MAX_COMMON_INTERLACING_INPUT_DIGITS
        for term in terms
    ):
        _reject(
            (*location, "polynomial", "terms"),
            "coefficient_digits",
            "a common interlacing source coefficient exceeds the "
            f"{MAX_COMMON_INTERLACING_INPUT_DIGITS}-digit input bound",
        )
    degree = terms[0].exponents[0] if terms else 0
    if degree < 1:
        _reject(
            location,
            "positive_degree",
            "common interlacing sources must have positive degree",
        )
    if degree > MAX_COMMON_INTERLACING_SOURCE_DEGREE:
        _reject(
            location,
            "source_degree",
            "a common interlacing source exceeds the "
            f"degree-{MAX_COMMON_INTERLACING_SOURCE_DEGREE} bound",
        )
    if terms[0].coefficient.as_fraction() != 1:
        _reject(
            location,
            "monic",
            "common interlacing sources must be monic over QQ",
        )
    (
        primitive,
        height_digits,
        denominator_digits,
        intermediate_digits,
    ) = _dense_primitive_coefficients(source)
    if denominator_digits > MAX_COMMON_INTERLACING_CLEARING_DENOMINATOR_DIGITS:
        _reject(
            location,
            "clearing_denominator",
            "the common denominator for a rational source exceeds the "
            f"{MAX_COMMON_INTERLACING_CLEARING_DENOMINATOR_DIGITS}-digit intermediate bound",
        )
    if intermediate_digits > MAX_COMMON_INTERLACING_CLEARING_INTERMEDIATE_DIGITS:
        _reject(
            location,
            "clearing_intermediate",
            "denominator clearing exceeds the "
            f"{MAX_COMMON_INTERLACING_CLEARING_INTERMEDIATE_DIGITS}-digit intermediate bound",
        )
    if height_digits > MAX_COMMON_INTERLACING_PRIMITIVE_HEIGHT_DIGITS:
        _reject(
            location,
            "primitive_height",
            "a primitive integer source exceeds the "
            f"{MAX_COMMON_INTERLACING_PRIMITIVE_HEIGHT_DIGITS}-digit height bound",
        )
    factor_bound = _factor_digit_bound(degree, height_digits)
    if factor_bound > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS:
        _reject(
            location,
            "factor_height",
            "the Landau--Mignotte factor-height bound exceeds the "
            f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit canonical algebraic-value bound",
        )
    return _PrimitiveSourcePlan(
        coefficients=primitive,
        degree=degree,
        height_digits=height_digits,
        term_count=len(terms),
    )


def _require_aggregate_source_bounds(
    primitive_sources: tuple[_PrimitiveSourcePlan, ...],
) -> None:
    total_degree = sum(source.degree for source in primitive_sources)
    if total_degree > MAX_COMMON_INTERLACING_TOTAL_DEGREE:
        _reject(
            ("family",),
            "total_degree",
            "common interlacing sources exceed the "
            f"{MAX_COMMON_INTERLACING_TOTAL_DEGREE} total-degree bound",
        )
    total_terms = sum(source.term_count for source in primitive_sources)
    if total_terms > MAX_COMMON_INTERLACING_TOTAL_TERMS:
        _reject(
            ("family",),
            "total_terms",
            "common interlacing sources exceed the "
            f"{MAX_COMMON_INTERLACING_TOTAL_TERMS}-term family bound",
        )
    factorization_work = sum(
        source.degree**4 * source.height_digits for source in primitive_sources
    )
    if factorization_work > MAX_COMMON_INTERLACING_FACTORIZATION_WORK:
        _reject(
            ("family",),
            "factorization_work",
            "exact factorization exceeds the work bound "
            f"(sum degree^4*height={factorization_work} > "
            f"{MAX_COMMON_INTERLACING_FACTORIZATION_WORK})",
        )


def _preflight_common_interlacing_sources(
    family: tuple[LabelledRationalPolynomial, ...],
) -> tuple[_PrimitiveSourcePlan, ...]:
    """Admit source degree, height, and aggregate work before backend launch."""

    common_variables = family[0].polynomial.variables
    primitive_sources = tuple(
        _preflight_source(
            source,
            source_index=source_index,
            common_variables=common_variables,
        )
        for source_index, source in enumerate(family)
    )
    common_degree = primitive_sources[0].degree
    for source_index, source in enumerate(primitive_sources[1:], start=1):
        if source.degree != common_degree:
            _reject(
                ("family", source_index, "polynomial"),
                "common_degree",
                "common interlacing sources must have the same positive degree",
            )
    _require_aggregate_source_bounds(primitive_sources)
    return primitive_sources


def _factor_source(
    primitive: _PrimitiveSourcePlan,
    *,
    source_index: int,
) -> _SourcePlan:
    import sympy

    variable = sympy.Symbol("x")
    polynomial = sympy.Poly.from_list(
        primitive.coefficients,
        gens=variable,
        domain=sympy.ZZ,
    )
    _unit, factorization = polynomial.factor_list()
    factors: list[_FactorPlan] = []
    squarefree_product = sympy.Poly(1, variable, domain=sympy.ZZ)
    for factor, multiplicity in factorization:
        canonical_coefficients = _canonical_factor(factor)
        if any(
            abs(coefficient) >= 10**MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
            for coefficient in canonical_coefficients
        ):
            _reject(
                ("family", source_index, "polynomial"),
                "factor_height",
                "an irreducible factor exceeds the "
                f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit RealAlgebraicValue bound",
            )
        factors.append(
            _FactorPlan(
                polynomial=factor,
                multiplicity=int(multiplicity),
                canonical_coefficients=canonical_coefficients,
            )
        )
        squarefree_product *= factor

    squarefree_degree = squarefree_product.degree()
    squarefree_height_digits = max(
        _integer_digits(int(coefficient))
        for coefficient in squarefree_product.all_coeffs()
    )
    endpoint_digits = _isolation_endpoint_digit_bound(
        squarefree_degree,
        squarefree_height_digits,
    )
    if endpoint_digits > CanonicalLimits().max_integer_digits:
        _reject(
            ("family", source_index, "polynomial"),
            "isolation_endpoint_digits",
            "exact rational isolating endpoints exceed the canonical "
            f"{CanonicalLimits().max_integer_digits}-digit scalar bound",
        )
    return _SourcePlan(
        factors=tuple(factors),
        squarefree_product=squarefree_product,
        squarefree_height_digits=squarefree_height_digits,
        isolation_endpoint_digits=endpoint_digits,
    )


def _require_factored_plan_bounds(sources: tuple[_SourcePlan, ...]) -> None:
    total_factors = sum(len(source.factors) for source in sources)
    isolation_work = sum(
        source.squarefree_product.degree() ** 5 * source.squarefree_height_digits
        for source in sources
    )
    factor_root_checks = sum(
        source.squarefree_product.degree() * len(source.factors) for source in sources
    )
    if total_factors > MAX_COMMON_INTERLACING_TOTAL_FACTORS:
        _reject(
            ("family",),
            "factor_count",
            "common interlacing sources exceed the "
            f"{MAX_COMMON_INTERLACING_TOTAL_FACTORS}-factor bound",
        )
    if isolation_work > MAX_COMMON_INTERLACING_ISOLATION_WORK:
        _reject(
            ("family",),
            "isolation_work",
            "exact real-root isolation exceeds the work bound "
            f"(sum squarefree-degree^5*height={isolation_work} > "
            f"{MAX_COMMON_INTERLACING_ISOLATION_WORK})",
        )
    if factor_root_checks > MAX_COMMON_INTERLACING_FACTOR_ROOT_CHECKS:
        _reject(
            ("family",),
            "factor_root_checks",
            "root-to-factor attribution exceeds the "
            f"{MAX_COMMON_INTERLACING_FACTOR_ROOT_CHECKS}-check bound",
        )


def _admit_common_interlacing(
    family: tuple[LabelledRationalPolynomial, ...],
    *,
    primitive_sources: tuple[_PrimitiveSourcePlan, ...] | None = None,
) -> _CommonInterlacingPlan:
    if not 2 <= len(family) <= MAX_COMMON_INTERLACING_FAMILY_SIZE:
        _reject(
            ("family",),
            "family_size",
            "common interlacing requires between 2 and "
            f"{MAX_COMMON_INTERLACING_FAMILY_SIZE} family members",
        )
    labels = tuple(source.label for source in family)
    if len(set(labels)) != len(labels):
        _reject(
            ("family",),
            "duplicate_label",
            "common interlacing family labels must be unique",
        )

    common_variables = family[0].polynomial.variables
    if len(common_variables) != 1:
        _reject(
            ("family", 0, "polynomial", "variables"),
            "variable_count",
            "common interlacing sources must be univariate",
        )
    if primitive_sources is None:
        primitive_sources = _preflight_common_interlacing_sources(family)
    common_degree = primitive_sources[0].degree

    # This is the operation's only factorization layer.  The exact factor plan
    # is retained for isolation, multiplicity expansion, root attribution, and
    # canonical result construction.
    sources = tuple(
        _factor_source(
            source,
            source_index=source_index,
        )
        for source_index, source in enumerate(primitive_sources)
    )
    _require_factored_plan_bounds(sources)
    comparison_count = (2 * len(family) - 1) * max(common_degree - 1, 0)
    if comparison_count > MAX_COMMON_INTERLACING_COMPARISONS:
        _reject(
            ("family",),
            "comparison_count",
            "common interlacing requires more than the "
            f"{MAX_COMMON_INTERLACING_COMPARISONS} exact endpoint-comparison bound",
        )
    return _CommonInterlacingPlan(
        family=family,
        common_degree=common_degree,
        sources=sources,
    )


def _canonical_rational(value: SympyRational) -> CanonicalRational:
    import sympy

    rational = sympy.Rational(value)
    return CanonicalRational(
        num=int(rational.p),
        den=int(rational.q),
    )


def _root_profile(source_index: int, plan: _SourcePlan) -> SourceRootProfile:
    factor_root_indices = [0] * len(plan.factors)
    roots: list[PolynomialRealRoot] = []
    for lower, upper in plan.squarefree_product.intervals(sqf=True):
        matches = [
            factor_index
            for factor_index, factor in enumerate(plan.factors)
            if strict_root_count(factor.polynomial, lower, upper) == 1
        ]
        if len(matches) != 1:  # pragma: no cover - exact backend invariant
            raise RuntimeError(
                "exact factor isolation did not identify one source root"
            )
        factor_index = matches[0]
        factor = plan.factors[factor_index]
        if factor.polynomial.degree() > MAX_COMMON_INTERLACING_FACTOR_DEGREE:
            _reject(
                ("family", source_index, "polynomial"),
                "factor_degree",
                "an irreducible factor with real roots exceeds the degree-"
                f"{MAX_COMMON_INTERLACING_FACTOR_DEGREE} result bound",
            )
        real_root_index = factor_root_indices[factor_index]
        factor_root_indices[factor_index] += 1
        value = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=factor.canonical_coefficients,
            real_root_index=real_root_index,
        )
        roots.append(
            PolynomialRealRoot(
                value=value,
                multiplicity=factor.multiplicity,
                isolating_interval=RationalIsolatingInterval(
                    lower=_canonical_rational(lower),
                    upper=_canonical_rational(upper),
                    interval_type="SINGLETON" if lower == upper else "OPEN",
                ),
            )
        )
    return SourceRootProfile(source_index=source_index, roots=tuple(roots))


def _expanded_root_references(
    profile: SourceRootProfile,
) -> tuple[PolynomialRootReference, ...]:
    return tuple(
        PolynomialRootReference(
            source_index=profile.source_index,
            distinct_root_index=root_index,
        )
        for root_index, root in enumerate(profile.roots)
        for _ in range(root.multiplicity)
    )


def _referenced_value(
    profiles: tuple[SourceRootProfile, ...],
    reference: PolynomialRootReference,
) -> RealAlgebraicValue:
    return profiles[reference.source_index].roots[reference.distinct_root_index].value


def _common_interlacing_outcome(
    profiles: tuple[SourceRootProfile, ...],
    common_degree: int,
) -> CommonInterlacingOutcome:
    comparison_cache: dict[
        tuple[tuple[tuple[int, ...], int], tuple[tuple[int, ...], int]], str
    ] = {}

    def root_order(
        left: PolynomialRootReference,
        right: PolynomialRootReference,
    ) -> str:
        left_root = profiles[left.source_index].roots[left.distinct_root_index]
        right_root = profiles[right.source_index].roots[right.distinct_root_index]
        left_key = (tuple(left_root.value.polynomial), left_root.value.real_root_index)
        right_key = (
            tuple(right_root.value.polynomial),
            right_root.value.real_root_index,
        )
        if left_key == right_key:
            return "EQ"
        cached = comparison_cache.get((left_key, right_key))
        if cached is not None:
            return cached
        if left_key[0] == right_key[0]:
            order = "LT" if left_key[1] < right_key[1] else "GT"
        else:
            order = _compare_admitted_real_algebraic(
                left_root.value,
                right_root.value,
                left_root.isolating_interval,
                right_root.isolating_interval,
            )
        comparison_cache[(left_key, right_key)] = order
        comparison_cache[(right_key, left_key)] = {
            "LT": "GT",
            "EQ": "EQ",
            "GT": "LT",
        }[order]
        return order

    expanded = tuple(_expanded_root_references(profile) for profile in profiles)
    for source_index, roots in enumerate(expanded):
        if len(roots) != common_degree:
            return CommonInterlacingDoesNotExist(
                obstruction=NonRealRootObstruction(
                    source_index=source_index,
                    real_root_multiplicity=len(roots),
                    nonreal_root_multiplicity=common_degree - len(roots),
                )
            )

    gaps: list[CommonInterlacingGap] = []
    for gap_index in range(common_degree - 1):
        # Starting from source zero and updating only for strict inequalities
        # gives the required lowest-source tie attribution deterministically.
        maximum_lower = expanded[0][gap_index]
        minimum_upper = expanded[0][gap_index + 1]
        for source_roots in expanded[1:]:
            lower_candidate = source_roots[gap_index]
            if root_order(maximum_lower, lower_candidate) == "LT":
                maximum_lower = lower_candidate
            upper_candidate = source_roots[gap_index + 1]
            if root_order(minimum_upper, upper_candidate) == "GT":
                minimum_upper = upper_candidate

        endpoint_order = root_order(maximum_lower, minimum_upper)
        if endpoint_order == "GT":
            return CommonInterlacingDoesNotExist(
                obstruction=EmptyGapObstruction(
                    gap_index=gap_index,
                    maximum_lower=maximum_lower,
                    minimum_upper=minimum_upper,
                )
            )
        gaps.append(
            CommonInterlacingGap(
                gap_index=gap_index,
                lower=maximum_lower,
                upper=minimum_upper,
            )
        )
    return CommonInterlacingExists(gaps=tuple(gaps))


def _common_interlacing_profile_in_process(
    family: tuple[LabelledRationalPolynomial, ...],
) -> CommonInterlacingProfile:
    plan = _admit_common_interlacing(family)
    root_profiles = tuple(
        _root_profile(source_index, source)
        for source_index, source in enumerate(plan.sources)
    )
    outcome = _common_interlacing_outcome(root_profiles, plan.common_degree)
    return CommonInterlacingProfile._from_kernel(
        family=plan.family,
        root_profiles=root_profiles,
        outcome=outcome,
    )


def common_interlacing_profile(
    family: tuple[LabelledRationalPolynomial, ...],
) -> CommonInterlacingProfile:
    """Return the complete exact common weak-interlacing profile of ``family``."""

    from jacobian.math.polynomials.real_algebra._common_interlacing_process import (
        run_common_interlacing_profile,
    )

    return run_common_interlacing_profile(family)


__all__ = ["common_interlacing_profile"]
