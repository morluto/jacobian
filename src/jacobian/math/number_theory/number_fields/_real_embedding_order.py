"""Exact element order under one recognized real number-field embedding."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, lcm
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.complex import (
    algebraic_root_separation_denominator_bound,
    complex_isolator_component_digit_bound,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RationalIsolatingInterval,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS,
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    NumberFieldRealValueEnclosure,
    RealNumberFieldEmbeddingRecord,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
    SimpleNumberFieldRealEmbeddingOrder,
    SimpleNumberFieldRealOrder,
)

# The selected-image minimal polynomial has degree at most the field degree.
# The dynamic resultant bound below must fit both the existing exact real-root
# carrier and this root-isolation envelope before SymPy constructs it.
MAX_REAL_EMBEDDING_ORDER_REFINEMENT_BITS = 32_768
MAX_REAL_EMBEDDING_ORDER_RESULTANT_STORAGE_BITS = 262_144


class NumberFieldRealEmbeddingOrderError(ValueError):
    """A proved owner-local rejection for selected-real-embedding order."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RealEmbeddingDifferenceAdmission:
    coordinates: tuple[Fraction, ...]
    minimal_polynomial_coefficient_bound: int
    root_refinement_bits: int
    predicted_resultant_storage_bits: int
    predicted_isolator_component_digits: int


@dataclass(frozen=True, slots=True)
class RecognizedRealEmbeddingContext:
    """One request-scoped recognized record and its private SymPy field."""

    record: RealNumberFieldEmbeddingRecord
    polynomial: Any
    algebraic_field: Any

    @property
    def presentation(self) -> SimpleNumberFieldPresentation:
        return self.record.embedding.presentation

    def to_backend(self, element: SimpleNumberFieldElement) -> Any:
        if element.presentation != self.presentation:
            raise NumberFieldRealEmbeddingOrderError(
                "element_field_mismatch",
                "field element does not belong to the recognized real embedding",
            )
        coefficients = [
            self.algebraic_field.dom.convert(coefficient.as_fraction())
            for coefficient in reversed(element.coefficients_ascending)
        ]
        return self.algebraic_field.new(coefficients)

    def from_backend(self, element: Any) -> SimpleNumberFieldElement:
        descending = list(element.to_list())
        degree = self.presentation.degree
        if len(descending) > degree:
            raise RuntimeError("SymPy returned an unreduced number-field element")
        descending = [self.algebraic_field.dom.zero] * (
            degree - len(descending)
        ) + descending
        ascending = tuple(
            Fraction(int(value.numerator), int(value.denominator))
            for value in reversed(descending)
        )
        _require_element_coordinates_fit(ascending, reason="kernel_coordinate_bound")
        return _element_from_fractions(self.presentation, ascending)


def _decimal_digits_from_bits(bits: int) -> int:
    return (max(bits, 1) * 30_103) // 100_000 + 1


def _canonical_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value.numerator, value.denominator)


def _element_from_fractions(
    presentation: SimpleNumberFieldPresentation,
    coordinates: tuple[Fraction, ...],
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            _canonical_rational(coordinate) for coordinate in coordinates
        ),
    )


def _rational_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(value.numerator).lstrip("-")),
        len(format_canonical_integer(value.denominator)),
    )


def _require_element_coordinates_fit(
    coordinates: tuple[Fraction, ...],
    *,
    reason: str,
) -> None:
    if any(
        _rational_component_digits(coordinate)
        > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS
        for coordinate in coordinates
    ):
        raise NumberFieldRealEmbeddingOrderError(
            reason,
            "exact reduced field-element coordinates exceed the "
            f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit bound",
        )


def _cleared_integer_polynomial(
    coordinates: tuple[Fraction, ...],
) -> tuple[int, tuple[int, ...]]:
    denominator = 1
    for coordinate in coordinates:
        denominator = lcm(denominator, coordinate.denominator)
    integers = tuple(
        coordinate.numerator * (denominator // coordinate.denominator)
        for coordinate in coordinates
    )
    while len(integers) > 1 and integers[-1] == 0:
        integers = integers[:-1]
    return denominator, integers


def _minimal_polynomial_height_bound(
    presentation: SimpleNumberFieldPresentation,
    coordinates: tuple[Fraction, ...],
) -> int:
    """Bound the primitive minimal polynomial of ``sum(c_i*alpha^i)``.

    Clear denominators as ``H(alpha)/D`` and eliminate ``alpha`` through
    ``Res_x(f(x), D*y-H(x))``.  The Sylvester determinant has size ``n+k``.
    Each entry has at most two terms in ``y``; a Leibniz expansion therefore
    bounds every resultant coefficient by ``s! 2^s C^s``.  The
    Landau--Mignotte factor bound then bounds every primitive factor, including
    the selected image's minimal polynomial, by ``2^n(n+1)`` times that height.
    """

    denominator, integer_coordinates = _cleared_integer_polynomial(coordinates)
    element_degree = len(integer_coordinates) - 1
    field_degree = presentation.degree
    coefficients = tuple(
        parse_canonical_integer(value)
        for value in presentation.coefficients_descending
    )
    entry_height = max(
        denominator,
        *(abs(value) for value in integer_coordinates),
        *(abs(value) for value in coefficients),
        1,
    )
    sylvester_size = field_degree + element_degree
    resultant_height = (
        factorial(sylvester_size)
        * 2**sylvester_size
        * entry_height**sylvester_size
    )
    return int((field_degree + 1) * 2**field_degree * resultant_height)


def _minimal_polynomial_height_bound_from_cleared_envelope(
    presentation: SimpleNumberFieldPresentation,
    *,
    cleared_denominator_bound: int,
    cleared_coordinate_bound: int,
    element_degree: int,
) -> int:
    field_degree = presentation.degree
    coefficients = tuple(
        parse_canonical_integer(value)
        for value in presentation.coefficients_descending
    )
    entry_height = max(
        cleared_denominator_bound,
        cleared_coordinate_bound,
        *(abs(value) for value in coefficients),
        1,
    )
    sylvester_size = field_degree + element_degree
    resultant_height = (
        factorial(sylvester_size)
        * 2**sylvester_size
        * entry_height**sylvester_size
    )
    return int((field_degree + 1) * 2**field_degree * resultant_height)


def _admit_image_polynomial_bound(
    presentation: SimpleNumberFieldPresentation,
    *,
    coordinates: tuple[Fraction, ...],
    coefficient_bound: int,
) -> RealEmbeddingDifferenceAdmission:
    coefficient_digits = _decimal_digits_from_bits(coefficient_bound.bit_length())
    if coefficient_digits > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS:
        raise NumberFieldRealEmbeddingOrderError(
            "image_minimal_polynomial_bound",
            "the selected image's elimination bound exceeds the "
            f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS:,}-digit exact real-algebraic "
            "coefficient envelope",
        )

    worst_coefficients = tuple(
        format_canonical_integer(coefficient_bound)
        for _ in range(presentation.degree + 1)
    )
    separation_denominator = algebraic_root_separation_denominator_bound(
        worst_coefficients
    )
    refinement_bits = separation_denominator.bit_length() + 8
    isolator_digits = complex_isolator_component_digit_bound(worst_coefficients)
    resultant_storage_bits = (
        presentation.degree + 1
    ) * coefficient_bound.bit_length()
    if resultant_storage_bits > MAX_REAL_EMBEDDING_ORDER_RESULTANT_STORAGE_BITS:
        raise NumberFieldRealEmbeddingOrderError(
            "image_resultant_storage_bound",
            "the selected image's exact elimination polynomial exceeds the "
            f"{MAX_REAL_EMBEDDING_ORDER_RESULTANT_STORAGE_BITS:,}-bit storage bound",
        )
    if refinement_bits > MAX_REAL_EMBEDDING_ORDER_REFINEMENT_BITS:
        raise NumberFieldRealEmbeddingOrderError(
            "image_root_refinement_bound",
            "selected-image real-root isolation exceeds the "
            f"{MAX_REAL_EMBEDDING_ORDER_REFINEMENT_BITS:,}-bit refinement bound",
        )
    if isolator_digits > MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS:
        raise NumberFieldRealEmbeddingOrderError(
            "image_isolator_component_bound",
            "selected-image isolation exceeds the "
            f"{MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS:,}-digit rational "
            "component bound",
        )
    return RealEmbeddingDifferenceAdmission(
        coordinates=coordinates,
        minimal_polynomial_coefficient_bound=coefficient_bound,
        root_refinement_bits=refinement_bits,
        predicted_resultant_storage_bits=resultant_storage_bits,
        predicted_isolator_component_digits=isolator_digits,
    )


def admit_real_embedding_difference(
    presentation: SimpleNumberFieldPresentation,
    coordinates: tuple[Fraction, ...],
) -> RealEmbeddingDifferenceAdmission:
    """Preflight exact coordinates, elimination, isolation, and result shape."""

    _require_element_coordinates_fit(coordinates, reason="difference_coordinate_bound")
    if all(coordinate == 0 for coordinate in coordinates):
        return RealEmbeddingDifferenceAdmission(
            coordinates=coordinates,
            minimal_polynomial_coefficient_bound=1,
            root_refinement_bits=1,
            predicted_resultant_storage_bits=1,
            predicted_isolator_component_digits=1,
        )
    if all(coordinate == 0 for coordinate in coordinates[1:]):
        component_digits = _rational_component_digits(coordinates[0])
        return RealEmbeddingDifferenceAdmission(
            coordinates=coordinates,
            minimal_polynomial_coefficient_bound=max(
                abs(coordinates[0].numerator), coordinates[0].denominator
            ),
            root_refinement_bits=1,
            predicted_resultant_storage_bits=max(
                abs(coordinates[0].numerator).bit_length(),
                coordinates[0].denominator.bit_length(),
            ),
            predicted_isolator_component_digits=component_digits,
        )

    return _admit_image_polynomial_bound(
        presentation,
        coordinates=coordinates,
        coefficient_bound=_minimal_polynomial_height_bound(
            presentation, coordinates
        ),
    )


def admit_real_embedding_difference_envelope(
    presentation: SimpleNumberFieldPresentation,
    *,
    coordinate_numerator_bound: int,
    coordinate_denominator_bound: int,
) -> RealEmbeddingDifferenceAdmission:
    """Admit every reduced difference within rational component bounds."""

    if coordinate_numerator_bound < 1 or coordinate_denominator_bound < 1:
        raise ValueError("coordinate component bounds must be positive")
    if (
        len(format_canonical_integer(coordinate_numerator_bound))
        > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS
        or len(format_canonical_integer(coordinate_denominator_bound))
        > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS
    ):
        raise NumberFieldRealEmbeddingOrderError(
            "difference_coordinate_bound",
            "exact reduced field-element coordinates exceed the "
            f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit bound",
        )
    degree = presentation.degree
    cleared_denominator_bound = coordinate_denominator_bound**degree
    cleared_coordinate_bound = (
        coordinate_numerator_bound * coordinate_denominator_bound ** (degree - 1)
    )
    coefficient_bound = _minimal_polynomial_height_bound_from_cleared_envelope(
        presentation,
        cleared_denominator_bound=cleared_denominator_bound,
        cleared_coordinate_bound=cleared_coordinate_bound,
        element_degree=degree - 1,
    )
    return _admit_image_polynomial_bound(
        presentation,
        coordinates=(),
        coefficient_bound=coefficient_bound,
    )


def _admit_binding_comparison(
    left: SimpleNumberFieldRealEmbeddingBinding,
    right: SimpleNumberFieldRealEmbeddingBinding,
) -> RealEmbeddingDifferenceAdmission:
    if left.embedding_record != right.embedding_record:
        raise NumberFieldRealEmbeddingOrderError(
            "embedding_record_mismatch",
            "comparison requires one exact selected real embedding record",
        )
    coordinates = tuple(
        left_value.as_fraction() - right_value.as_fraction()
        for left_value, right_value in zip(
            left.element.coefficients_ascending,
            right.element.coefficients_ascending,
            strict=True,
        )
    )
    return admit_real_embedding_difference(left.element.presentation, coordinates)


def recognize_real_embedding_record(
    record: RealNumberFieldEmbeddingRecord,
) -> RecognizedRealEmbeddingContext:
    """Recognize one caller-authored record through the complete producer path."""

    # Keep recognition owner-local without making the embedding producer
    # depend on this consumer's private arithmetic kernel.
    from jacobian.math.number_theory.number_fields.operations import (
        NumberFieldEmbeddingAdmissionError,
        embeddings,
    )

    try:
        profile = embeddings(record.embedding.presentation)
    except NumberFieldEmbeddingAdmissionError as exc:
        raise NumberFieldRealEmbeddingOrderError(
            f"embedding_{exc.reason}", str(exc)
        ) from exc
    recognized = next(
        (
            candidate
            for candidate in profile.records
            if isinstance(candidate, RealNumberFieldEmbeddingRecord)
            and candidate == record
        ),
        None,
    )
    if recognized is None:
        raise NumberFieldRealEmbeddingOrderError(
            "embedding_record_not_recognized",
            "real embedding record does not match the complete exact embedding producer",
        )

    import sympy

    alpha = sympy.Symbol("alpha")
    polynomial = sympy.Poly.from_list(
        [
            parse_canonical_integer(value)
            for value in recognized.embedding.presentation.coefficients_descending
        ],
        gens=alpha,
        domain=sympy.QQ,
    )
    algebraic_field = sympy.QQ.alg_field_from_poly(
        polynomial,
        alias="alpha",
        root_index=recognized.embedding.root.real_root_index,
    )
    return RecognizedRealEmbeddingContext(
        record=recognized,
        polynomial=polynomial,
        algebraic_field=algebraic_field,
    )


def recognize_real_embedding_binding(
    binding: SimpleNumberFieldRealEmbeddingBinding,
) -> RecognizedRealEmbeddingContext:
    return recognize_real_embedding_record(binding.embedding_record)


def _normalize_minimal_polynomial(polynomial: Any) -> Any:
    _denominator, integral = polynomial.clear_denoms(convert=True)
    _content, primitive = integral.primitive()
    return -primitive if primitive.LC() < 0 else primitive


def isolate_backend_real_value(
    context: RecognizedRealEmbeddingContext,
    value: Any,
    admission: RealEmbeddingDifferenceAdmission,
) -> tuple[SimpleNumberFieldRealOrder, RationalIsolatingInterval]:
    """Return exact sign and isolation evidence for one admitted backend value."""

    if value == context.algebraic_field.zero:
        zero = _canonical_rational(Fraction(0))
        return (
            "EQ",
            RationalIsolatingInterval(
                lower=zero,
                upper=zero,
                interval_type="SINGLETON",
            ),
        )

    descending = list(value.to_list())
    if len(descending) == 1:
        rational = Fraction(
            int(descending[0].numerator), int(descending[0].denominator)
        )
        canonical = _canonical_rational(rational)
        return (
            "LT" if rational < 0 else "GT",
            RationalIsolatingInterval(
                lower=canonical,
                upper=canonical,
                interval_type="SINGLETON",
            ),
        )

    import sympy

    image = context.algebraic_field.to_sympy(value)
    variable = sympy.Symbol("image")
    minimal_polynomial = _normalize_minimal_polynomial(
        sympy.minpoly(image, variable, polys=True)
    )
    if minimal_polynomial.degree() > context.presentation.degree:
        raise RuntimeError("SymPy returned an over-degree selected-image polynomial")
    actual_height = max(
        abs(int(coefficient)) for coefficient in minimal_polynomial.all_coeffs()
    )
    if actual_height > admission.minimal_polynomial_coefficient_bound:
        raise RuntimeError("selected-image polynomial exceeded its admitted height bound")

    real_roots = minimal_polynomial.real_roots(radicals=False)
    matches = tuple(
        index
        for index, root in enumerate(real_roots)
        if minimal_polynomial.same_root(root, image)
    )
    if len(matches) != 1:
        raise RuntimeError("exact selected-image isolation did not identify one real root")

    constant = abs(int(minimal_polynomial.TC()))
    if constant == 0:
        raise RuntimeError("a nonzero field element received a zero-root minimal polynomial")
    height = max(abs(int(coefficient)) for coefficient in minimal_polynomial.all_coeffs())
    epsilon = sympy.Rational(constant, 2 * (constant + height))
    intervals = minimal_polynomial.intervals(eps=epsilon)
    (lower, upper), _multiplicity = intervals[matches[0]]
    lower_fraction = Fraction(int(lower.p), int(lower.q))
    upper_fraction = Fraction(int(upper.p), int(upper.q))
    if upper_fraction < 0:
        order: SimpleNumberFieldRealOrder = "LT"
    elif lower_fraction > 0:
        order = "GT"
    else:
        raise RuntimeError("selected-image isolation did not establish a strict sign")
    if any(
        _rational_component_digits(endpoint)
        > MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS
        for endpoint in (lower_fraction, upper_fraction)
    ):
        raise RuntimeError("selected-image isolator exceeded its admitted component bound")
    return (
        order,
        RationalIsolatingInterval(
            lower=_canonical_rational(lower_fraction),
            upper=_canonical_rational(upper_fraction),
            interval_type="SINGLETON" if lower_fraction == upper_fraction else "OPEN",
        ),
    )


def compare_real_embedding_elements(
    left: SimpleNumberFieldRealEmbeddingBinding,
    right: SimpleNumberFieldRealEmbeddingBinding,
) -> SimpleNumberFieldRealEmbeddingOrder:
    """Compare two exact field elements under one selected real embedding."""

    admission = _admit_binding_comparison(left, right)
    context = recognize_real_embedding_binding(left)
    difference_element = _element_from_fractions(
        context.presentation,
        admission.coordinates,
    )
    difference = SimpleNumberFieldRealEmbeddingBinding(
        element=difference_element,
        embedding_record=context.record,
    )
    backend_difference = context.to_backend(difference_element)
    order, interval = isolate_backend_real_value(
        context,
        backend_difference,
        admission,
    )
    return SimpleNumberFieldRealEmbeddingOrder._from_kernel(
        left=left,
        right=right,
        difference=difference,
        order=order,
        difference_enclosure=NumberFieldRealValueEnclosure(
            lower=interval.lower,
            upper=interval.upper,
            interval_type=(
                "SINGLETON" if interval.lower == interval.upper else "CLOSED"
            ),
        ),
    )


__all__ = [
    "MAX_REAL_EMBEDDING_ORDER_REFINEMENT_BITS",
    "MAX_REAL_EMBEDDING_ORDER_RESULTANT_STORAGE_BITS",
    "NumberFieldRealEmbeddingOrderError",
    "RealEmbeddingDifferenceAdmission",
    "RecognizedRealEmbeddingContext",
    "admit_real_embedding_difference",
    "admit_real_embedding_difference_envelope",
    "compare_real_embedding_elements",
    "isolate_backend_real_value",
    "recognize_real_embedding_binding",
    "recognize_real_embedding_record",
]
