"""Exact evaluation of additive-group actions on arbitrary polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import (
    MAX_DERIVATION_COEFFICIENT_DIGITS,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._stable_operations import (
    _admit_and_verify_ga_action,
)
from jacobian.math.polynomials.derivations.orbits._models import (
    MAX_GA_ORBIT_COEFFICIENT_DIGITS,
    MAX_GA_ORBIT_EXPANSIONS,
    MAX_GA_ORBIT_INTERMEDIATE_BYTES,
    MAX_GA_ORBIT_OUTPUT_BYTES,
    MAX_GA_ORBIT_SOURCE_DEGREE,
    MAX_GA_ORBIT_SOURCE_TERMS,
    MAX_GA_ORBIT_WORK,
    GaPolynomialOrbitRequest,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

_Monomial = tuple[int, ...]
_Image = tuple[tuple[_Monomial, Fraction], ...]


@dataclass(frozen=True)
class _OrbitPlan:
    request: GaPolynomialOrbitRequest
    images: tuple[_Image, ...]
    candidate_count: int
    work_bound: int
    intermediate_bytes_bound: int
    coefficient_digits_bound: int


@dataclass(frozen=True)
class _TermPlan:
    expansion_count: int
    pair_products: int
    denominator_digits: int
    numerator_digits: int
    maximum_partial_support: int


def _reject_resource(code: str, message: str, location: tuple[str | int, ...]) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"polynomial_ga_orbit.{code}",
        message=message,
    )


def _canonical_request(
    action: PolynomialGaAction, polynomial: RationalPolynomial
) -> GaPolynomialOrbitRequest:
    if (
        type(action) is not PolynomialGaAction
        or type(polynomial) is not RationalPolynomial
    ):
        raise OperationDomainValidationError(
            location=("action", "polynomial"),
            code="polynomial_ga_orbit.noncanonical_input",
            message="orbit evaluation requires canonical action and polynomial values",
        )
    try:
        request = GaPolynomialOrbitRequest.model_validate(
            {"action": action.model_dump(), "polynomial": polynomial.model_dump()},
            strict=True,
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("action", "polynomial"),
            code="polynomial_ga_orbit.request_shape",
            message="the action and polynomial do not form a valid orbit request",
        ) from exc
    return request


def _admit_source(source: RationalPolynomial) -> None:
    try:
        require_polynomial_budget(
            source,
            maximum_terms=MAX_GA_ORBIT_SOURCE_TERMS,
            maximum_exponent=MAX_GA_ORBIT_SOURCE_DEGREE,
            maximum_coefficient_digits=MAX_DERIVATION_COEFFICIENT_DIGITS,
            label="orbit source polynomial",
        )
    except ValueError as exc:
        _reject_resource(
            "source_polynomial_budget",
            str(exc),
            ("polynomial",),
        )
    for index, term in enumerate(source.polynomial.terms):
        if sum(term.exponents) > MAX_GA_ORBIT_SOURCE_DEGREE:
            _reject_resource(
                "source_total_degree",
                f"orbit source total degree exceeds {MAX_GA_ORBIT_SOURCE_DEGREE}",
                ("polynomial", "polynomial", "terms", index),
            )


def _image_maps(action: PolynomialGaAction) -> tuple[_Image, ...]:
    return tuple(
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in image.polynomial.terms
        )
        for image in action.generator_images
    )


def _ceil_log10(value: int) -> int:
    if value <= 1:
        return 0
    return len(str(value - 1))


def _preflight_term(
    action: PolynomialGaAction,
    term: RationalPolynomialTerm,
    term_index: int,
    image_counts: tuple[int, ...],
    image_num_digits: tuple[int, ...],
    image_denominator_digits: tuple[int, ...],
    image_axis_maxima: tuple[tuple[int, ...], ...],
) -> _TermPlan:
    """Admit one source monomial's expansion without materializing products."""
    expansion_count = 1
    pair_products = 0
    denominator_digits = (
        len(str(term.coefficient.den)) if term.coefficient.den > 1 else 0
    )
    numerator_digits = len(str(abs(term.coefficient.num)))
    maximum_partial_support = 1
    for generator_index, exponent in enumerate(term.exponents):
        image_count = image_counts[generator_index]
        if exponent and image_count == 0:
            return _TermPlan(0, pair_products, 0, 0, maximum_partial_support)
        for _ in range(exponent):
            if expansion_count > MAX_GA_ORBIT_EXPANSIONS // image_count:
                _reject_resource(
                    "expansion_budget",
                    "polynomial orbit exceeds the admitted monomial expansion count",
                    ("polynomial", "polynomial", "terms", term_index),
                )
            expansion_count *= image_count
            pair_products += expansion_count
            maximum_partial_support = max(maximum_partial_support, expansion_count)
            if pair_products > MAX_GA_ORBIT_WORK:
                _reject_resource(
                    "work_budget",
                    "polynomial orbit exceeds its repeated-product work bound",
                    ("polynomial", "polynomial", "terms", term_index),
                )
        denominator_digits += exponent * image_denominator_digits[generator_index]
        numerator_digits += exponent * image_num_digits[generator_index]

    for target_axis in range(len(action.source_variables) + 1):
        output_exponent = sum(
            exponent * image_axis_maxima[index][target_axis]
            for index, exponent in enumerate(term.exponents)
        )
        if output_exponent > MAX_POLYNOMIAL_EXPONENT:
            _reject_resource(
                "output_exponent_budget",
                "polynomial orbit would exceed the exact polynomial exponent bound",
                ("polynomial", "polynomial", "terms", term_index),
            )
    return _TermPlan(
        expansion_count,
        pair_products,
        denominator_digits,
        numerator_digits,
        maximum_partial_support,
    )


def _preflight(request: GaPolynomialOrbitRequest) -> _OrbitPlan:
    action = request.action
    source = request.polynomial
    _admit_source(source)
    request_checkpoint("before verifying additive-group coaction laws")
    _admit_and_verify_ga_action(action)
    request_checkpoint("after verifying additive-group coaction laws")
    images = _image_maps(action)
    image_counts = tuple(len(image) for image in images)
    image_max_num_digits = tuple(
        max(
            (len(str(abs(coefficient.numerator))) for _, coefficient in image),
            default=1,
        )
        for image in images
    )
    image_denominator_digits = tuple(
        sum(
            len(str(denominator))
            for denominator in {coefficient.denominator for _, coefficient in image}
            if denominator > 1
        )
        for image in images
    )
    image_axis_maxima = tuple(
        tuple(
            max((monomial[axis] for monomial, _ in image), default=0)
            for axis in range(len(action.source_variables) + 1)
        )
        for image in images
    )

    candidate_count = 0
    pair_products = 0
    maximum_partial_support = 1
    # Distinct source denominators can combine whenever substituted supports
    # overlap, even when their bounding-box maxima differ.  A global set is a
    # conservative collision group and avoids charging repeated denominators.
    source_denominators: set[int] = set()
    maximum_contribution_numerator_digits = 1
    target_axis_count = len(action.source_variables) + 1
    for term_index, term in enumerate(source.polynomial.terms):
        term_plan = _preflight_term(
            action,
            term,
            term_index,
            image_counts,
            image_max_num_digits,
            image_denominator_digits,
            image_axis_maxima,
        )
        if candidate_count + term_plan.expansion_count > MAX_GA_ORBIT_EXPANSIONS:
            _reject_resource(
                "expansion_budget",
                "polynomial orbit exceeds the admitted monomial expansion count",
                ("polynomial", "polynomial", "terms", term_index),
            )
        candidate_count += term_plan.expansion_count
        pair_products += term_plan.pair_products
        maximum_partial_support = max(
            maximum_partial_support, term_plan.maximum_partial_support
        )
        source_denominators.add(term.coefficient.den)
        for exponent, image in zip(term.exponents, images, strict=True):
            if exponent:
                source_denominators.update(
                    coefficient.denominator for _, coefficient in image
                )
        maximum_contribution_numerator_digits = max(
            maximum_contribution_numerator_digits, term_plan.numerator_digits
        )

    if pair_products * (target_axis_count + 1) > MAX_GA_ORBIT_WORK:
        _reject_resource(
            "work_budget",
            "polynomial orbit exceeds its coefficient and exponent work bound",
            ("polynomial",),
        )
    sort_work = candidate_count * max(1, candidate_count.bit_length())
    work_bound = pair_products * (target_axis_count + 1) + sort_work + candidate_count
    if work_bound > MAX_GA_ORBIT_WORK:
        _reject_resource(
            "work_budget",
            "polynomial orbit exceeds its admitted exact work bound",
            ("polynomial",),
        )

    maximum_collision_denominator_digits = sum(
        len(str(denominator)) for denominator in source_denominators if denominator > 1
    )
    coefficient_digits_bound = (
        maximum_contribution_numerator_digits
        + maximum_collision_denominator_digits
        + _ceil_log10(candidate_count)
    )
    if (
        maximum_collision_denominator_digits > MAX_GA_ORBIT_COEFFICIENT_DIGITS
        or coefficient_digits_bound > MAX_GA_ORBIT_COEFFICIENT_DIGITS
    ):
        _reject_resource(
            "coefficient_growth",
            "polynomial orbit coefficient growth exceeds the 128-digit bound",
            ("action", "generator_images"),
        )

    # A canonical JSON term needs at most 400 bytes: two 128-digit rational
    # components, eight five-digit exponents, JSON keys/punctuation, and sign.
    output_bytes_bound = 4096 + candidate_count * 400
    intermediate_bytes_bound = (
        candidate_count + maximum_partial_support
    ) * 128 + pair_products * 16
    if output_bytes_bound > MAX_GA_ORBIT_OUTPUT_BYTES:
        _reject_resource(
            "output_bytes",
            "polynomial orbit exceeds its serialized output bound",
            ("polynomial",),
        )
    if intermediate_bytes_bound > MAX_GA_ORBIT_INTERMEDIATE_BYTES:
        _reject_resource(
            "intermediate_bytes",
            "polynomial orbit exceeds its intermediate allocation bound",
            ("action", "polynomial"),
        )

    return _OrbitPlan(
        request=request,
        images=images,
        candidate_count=candidate_count,
        work_bound=work_bound,
        intermediate_bytes_bound=intermediate_bytes_bound,
        coefficient_digits_bound=coefficient_digits_bound,
    )


def _expand(plan: _OrbitPlan, ledger: OperationWorkLedger) -> RationalPolynomial:
    action = plan.request.action
    source = plan.request.polynomial
    target_variables = (*action.source_variables, action.parameter)
    accumulated: dict[_Monomial, Fraction] = {}
    for term in source.polynomial.terms:
        request_checkpoint("during additive-group polynomial orbit expansion")
        partial: dict[_Monomial, Fraction] = {
            tuple(0 for _ in target_variables): term.coefficient.as_fraction()
        }
        annihilated = False
        for generator_index, exponent in enumerate(term.exponents):
            image = plan.images[generator_index]
            if exponent and not image:
                annihilated = True
                break
            for _ in range(exponent):
                multiplied: dict[_Monomial, Fraction] = {}
                for left_exponents, left_coefficient in partial.items():
                    for right_exponents, right_coefficient in image:
                        ledger.charge(len(target_variables) + 1)
                        exponent_sum = tuple(
                            left + right
                            for left, right in zip(
                                left_exponents, right_exponents, strict=True
                            )
                        )
                        multiplied[exponent_sum] = (
                            multiplied.get(exponent_sum, Fraction(0))
                            + left_coefficient * right_coefficient
                        )
                partial = {
                    exponents: coefficient
                    for exponents, coefficient in multiplied.items()
                    if coefficient
                }
                if not partial:
                    annihilated = True
                    break
            if annihilated:
                break
        for exponents, coefficient in partial.items():
            ledger.charge()
            accumulated[exponents] = (
                accumulated.get(exponents, Fraction(0)) + coefficient
            )

    ledger.charge(len(accumulated) * max(1, len(accumulated).bit_length()))
    request_checkpoint("before additive-group orbit result construction")
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(coefficient),
            exponents=exponents,
        )
        for exponents, coefficient in sorted(accumulated.items(), reverse=True)
        if coefficient
    )
    return RationalPolynomial(
        variables=target_variables,
        polynomial=SparseRationalPolynomial(terms=terms),
    )


def ga_polynomial_orbit(
    action: PolynomialGaAction, polynomial: RationalPolynomial
) -> RationalPolynomial:
    """Compute the coaction image of a source polynomial in the parameter ring."""
    request_checkpoint("before additive-group polynomial orbit request admission")
    request = _canonical_request(action, polynomial)
    plan = _preflight(request)
    request_checkpoint("after additive-group polynomial orbit admission")
    ledger = OperationWorkLedger(plan.work_bound)
    return _expand(plan, ledger)


def compute_ga_polynomial_orbit(
    request: GaPolynomialOrbitRequest,
) -> RationalPolynomial:
    """Catalog entry point sharing the native operation's admission boundary."""
    return ga_polynomial_orbit(request.action, request.polynomial)
