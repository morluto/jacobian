"""Number field operations backed by SymPy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
    RationalComplexIsolatingRectangle,
    _isolate_complex_algebraic,
    _public_to_backend_root_indices_from_count,
    _root_evidence_parameters,
    algebraic_real_part_separation_denominator_bound,
    algebraic_root_separation_denominator_bound,
    complex_isolator_component_digit_bound,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
    _sympy_polynomial_from_coefficients,
)
from jacobian.math.number_theory.number_fields.values import (
    ComplexNumberFieldEmbedding,
    ComplexNumberFieldEmbeddingRecord,
    NumberFieldConjugatePair,
    NumberFieldEmbeddingProfile,
    NumberFieldSignature,
    RealNumberFieldEmbedding,
    RealNumberFieldEmbeddingRecord,
    SimpleNumberFieldPresentation,
)

MAX_NUMBER_FIELD_EMBEDDING_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS = 2_097_152
MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS = 32_768


class NumberFieldEmbeddingAdmissionError(ValueError):
    """A proved owner-local resource rejection for embedding enumeration."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class NumberFieldEmbeddingAdmission:
    degree: int
    real_embedding_count: int
    complex_conjugate_pair_count: int
    coefficient_height_bits: int
    root_separation_denominator_bits: int
    real_part_separation_denominator_bits: int
    root_isolation_bits: int
    real_part_resultant_coefficient_bits: int
    real_part_resultant_storage_bits: int
    isolator_component_digits: int
    polynomial_discriminant_digits: int
    predicted_result_bytes: int
    root_refinement_calls: int
    root_refinement_precision_bits: int
    root_refinement_bit_work: int
    real_isolating_intervals: tuple[RationalIsolatingInterval, ...]
    complex_isolating_rectangles: tuple[RationalComplexIsolatingRectangle, ...]


def _decimal_digits_from_bits(bits: int) -> int:
    return (max(bits, 1) * 30_103) // 100_000 + 1


def _backend_fraction(value: Any) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def _dyadic_floor(value: Fraction, denominator: int) -> int:
    return (value.numerator * denominator) // value.denominator


def _dyadic_ceiling(value: Fraction, denominator: int) -> int:
    return -((-value.numerator * denominator) // value.denominator)


def _normalize_real_isolator(
    lower: Any,
    upper: Any,
    *,
    grid_denominator: int,
) -> RationalIsolatingInterval:
    lower_fraction = _backend_fraction(lower)
    upper_fraction = _backend_fraction(upper)
    if lower_fraction == upper_fraction:
        endpoint = CanonicalRational.from_fraction(lower_fraction)
        return RationalIsolatingInterval(
            lower=endpoint,
            upper=endpoint,
            interval_type="SINGLETON",
        )
    lower_cell = _dyadic_floor(lower_fraction, grid_denominator)
    upper_cell = _dyadic_ceiling(upper_fraction, grid_denominator)
    return RationalIsolatingInterval(
        lower=CanonicalRational.from_fraction(
            Fraction(lower_cell - 1, grid_denominator)
        ),
        upper=CanonicalRational.from_fraction(
            Fraction(upper_cell + 1, grid_denominator)
        ),
        interval_type="OPEN",
    )


def _normalize_complex_isolator(
    lower: Any,
    upper: Any,
    *,
    grid_denominator: int,
) -> RationalComplexIsolatingRectangle:
    lower_real, lower_imaginary = lower.as_real_imag()
    upper_real, upper_imaginary = upper.as_real_imag()
    real_lower_cell = _dyadic_floor(_backend_fraction(lower_real), grid_denominator)
    real_upper_cell = _dyadic_ceiling(_backend_fraction(upper_real), grid_denominator)
    imaginary_lower_cell = _dyadic_floor(
        _backend_fraction(lower_imaginary), grid_denominator
    )
    imaginary_upper_cell = _dyadic_ceiling(
        _backend_fraction(upper_imaginary), grid_denominator
    )
    return RationalComplexIsolatingRectangle(
        real_lower=CanonicalRational.from_fraction(
            Fraction(real_lower_cell - 1, grid_denominator)
        ),
        real_upper=CanonicalRational.from_fraction(
            Fraction(real_upper_cell + 1, grid_denominator)
        ),
        imaginary_lower=CanonicalRational.from_fraction(
            Fraction(imaginary_lower_cell - 1, grid_denominator)
        ),
        imaginary_upper=CanonicalRational.from_fraction(
            Fraction(imaginary_upper_cell + 1, grid_denominator)
        ),
    )


def _admit_number_field_embeddings(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldEmbeddingAdmission:
    """Preflight root work, intermediates, exact output, and serialization."""

    coefficients = tuple(
        parse_canonical_integer(coefficient)
        for coefficient in field.coefficients_descending
    )
    degree = field.degree
    height = max(abs(coefficient) for coefficient in coefficients)
    height_bits = max(height.bit_length(), 1)
    separation_denominator = algebraic_root_separation_denominator_bound(
        field.coefficients_descending
    )
    separation_bits = separation_denominator.bit_length()
    # Evidence uses a dyadic grid more than 2^4 finer than Mignotte separation
    # and indexed-root matching another 2^4 finer than that grid.  A backend
    # isolator is first refined to one grid cell; outward normalization spans
    # at most four cells, whose diagonal remains below the separation bound.
    isolation_bits = separation_bits + 8
    isolator_digits = complex_isolator_component_digit_bound(
        field.coefficients_descending
    )
    if degree == 1:
        # The exact singleton root -b/a retains a reduced divisor of the
        # source coefficients rather than a dyadic denominator.
        isolator_digits = max(
            isolator_digits,
            *(
                len(coefficient.lstrip("-"))
                for coefficient in field.coefficients_descending
            ),
        )

    if isolator_digits > 4_096:
        raise NumberFieldEmbeddingAdmissionError(
            "isolation_intermediate_bound",
            "the Mignotte-derived root-isolation envelope exceeds the "
            "4,096-digit rational component bound",
        )
    if isolation_bits > MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:
        raise NumberFieldEmbeddingAdmissionError(
            "root_isolation_precision_bound",
            "exact root isolation exceeds the "
            f"{MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:,}-bit refinement bound",
        )

    # Hadamard on the (2n-1)-square Sylvester matrix gives
    # |disc(f)| <= |Res(f,f')|
    # <= n^n (n+1)^((2n-1)/2) H^(2n-1).
    # Replacing the half power by (n+1)^(2n-1) is an integer upper bound.
    discriminant_bound = (
        1
        if degree == 1
        else degree**degree
        * (degree + 1) ** (2 * degree - 1)
        * height ** (2 * degree - 1)
    )
    discriminant_digits = _decimal_digits_from_bits(discriminant_bound.bit_length())

    # In f(u + i*v), every coefficient of Re(f) and Im(f) has magnitude at
    # most C = (n+1) 2^n H.  A Leibniz expansion of the at-most-2n Sylvester
    # determinant therefore bounds every resultant coefficient by
    # (2n)! (n+1)^(2n-1) C^(2n).  Its u-degree is at most 2n^2.  The final
    # Landau-Mignotte factor covers coefficient growth when taking the
    # primitive square-free part used for real-coordinate separation.
    sylvester_size = 2 * degree
    resultant_degree = 2 * degree * degree
    expanded_coefficient_bound = (degree + 1) * 2**degree * height
    expanded_resultant_coefficient_bound = (
        factorial(sylvester_size)
        * (degree + 1) ** max(sylvester_size - 1, 0)
        * expanded_coefficient_bound**sylvester_size
    )
    resultant_coefficient_bound = (
        (resultant_degree + 1)
        * 2**resultant_degree
        * expanded_resultant_coefficient_bound
    )
    resultant_coefficient_bits = max(resultant_coefficient_bound.bit_length(), 1)
    resultant_storage_bits = (resultant_degree + 1) * resultant_coefficient_bits
    if resultant_storage_bits > MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS:
        raise NumberFieldEmbeddingAdmissionError(
            "pair_ordering_resultant_bound",
            "the exact real-coordinate resultant exceeds the "
            f"{MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS:,}-bit "
            "intermediate storage bound",
        )

    # Every record repeats one field presentation and one indexed polynomial;
    # complex evidence is the larger case at four bounded rationals.  The
    # fixed allowance covers all field names, enum literals, tuple delimiters,
    # signature/pair records, and canonical JSON punctuation at degree <= 8.
    source_bytes = len(field.model_dump_json().encode("utf-8"))
    polynomial_bytes = (
        sum(len(coefficient) for coefficient in field.coefficients_descending)
        + 3 * (degree + 1)
        + 64
    )
    rational_bytes = 2 * isolator_digits + 32
    embedding_bytes = source_bytes + polynomial_bytes + 256
    record_bytes = embedding_bytes + 4 * rational_bytes + 512
    predicted_result_bytes = (
        source_bytes + degree * record_bytes + discriminant_digits + 4_096
    )
    if predicted_result_bytes > MAX_NUMBER_FIELD_EMBEDDING_RESULT_BYTES:
        raise NumberFieldEmbeddingAdmissionError(
            "result_byte_bound",
            "the complete retained-source embedding profile exceeds the "
            f"{MAX_NUMBER_FIELD_EMBEDDING_RESULT_BYTES:,}-byte result bound",
        )

    # The raw carrier has already bounded degree and coefficient digits, and
    # the formulas above have now admitted exact real-root isolation plus the
    # largest possible elimination resultant.  It is therefore safe to ask
    # the maintained backend for the exact signature before deciding whether
    # pair ordering needs that resultant at all.
    import sympy

    polynomial = _sympy_polynomial_from_coefficients(field.coefficients_descending)
    real_count = int(polynomial.count_roots(-sympy.oo, sympy.oo))
    pair_count = (degree - real_count) // 2
    real_part_separation_denominator = (
        algebraic_real_part_separation_denominator_bound(field.coefficients_descending)
        if pair_count > 1
        else 1
    )
    real_part_separation_bits = real_part_separation_denominator.bit_length()
    refinement_precision_bits = max(isolation_bits, real_part_separation_bits + 4)
    if refinement_precision_bits > MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:
        raise NumberFieldEmbeddingAdmissionError(
            "pair_ordering_precision_bound",
            "exact conjugate-pair ordering exceeds the "
            f"{MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:,}-bit refinement bound",
        )

    grid_denominator, _matching_error = _root_evidence_parameters(
        field.coefficients_descending
    )
    backend_real_intervals, backend_complex_rectangles = polynomial.intervals(
        all=True,
        eps=sympy.Rational(1, grid_denominator),
    )
    real_isolating_intervals = tuple(
        _normalize_real_isolator(
            lower,
            upper,
            grid_denominator=grid_denominator,
        )
        for (lower, upper), _multiplicity in backend_real_intervals
    )
    complex_isolating_rectangles = tuple(
        _normalize_complex_isolator(
            lower,
            upper,
            grid_denominator=grid_denominator,
        )
        for (lower, upper), _multiplicity in backend_complex_rectangles
    )

    # The kernel makes two exact rational approximations per conjugate pair to
    # establish sign/order and one more to select its deterministic admitted
    # rectangle.  Real evidence is reused directly from the signature/root-
    # isolation pass.  This is a precision-and-call bound on the actual root
    # refinement plan, rather than an unrelated combinatorial proxy.
    root_refinement_calls = 3 * pair_count
    root_refinement_bit_work = root_refinement_calls * refinement_precision_bits
    return NumberFieldEmbeddingAdmission(
        degree=degree,
        real_embedding_count=real_count,
        complex_conjugate_pair_count=pair_count,
        coefficient_height_bits=height_bits,
        root_separation_denominator_bits=separation_bits,
        real_part_separation_denominator_bits=real_part_separation_bits,
        root_isolation_bits=isolation_bits,
        real_part_resultant_coefficient_bits=resultant_coefficient_bits,
        real_part_resultant_storage_bits=resultant_storage_bits,
        isolator_component_digits=isolator_digits,
        polynomial_discriminant_digits=discriminant_digits,
        predicted_result_bytes=predicted_result_bytes,
        root_refinement_calls=root_refinement_calls,
        root_refinement_precision_bits=refinement_precision_bits,
        root_refinement_bit_work=root_refinement_bit_work,
        real_isolating_intervals=real_isolating_intervals,
        complex_isolating_rectangles=complex_isolating_rectangles,
    )


def embeddings(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldEmbeddingProfile:
    """Return every exact embedding of one bounded presented field.

    Root identity uses increasing real roots followed by conjugate pairs sorted
    by the positive representative's exact ``(Re, Im)`` coordinates, with the
    negative root first in each pair.  An exact real-coordinate elimination and
    Mignotte separation map this public order to SymPy's private root indexes.
    Only indexed canonical roots and rational isolation evidence cross the
    boundary.
    """

    admission = _admit_number_field_embeddings(field)

    polynomial = _sympy_polynomial_from_coefficients(field.coefficients_descending)
    real_count = admission.real_embedding_count
    pair_count = admission.complex_conjugate_pair_count
    backend_root_indices = _public_to_backend_root_indices_from_count(
        field.coefficients_descending,
        real_count,
    )

    records: list[
        RealNumberFieldEmbeddingRecord | ComplexNumberFieldEmbeddingRecord
    ] = []
    for root_index in range(real_count):
        real_root = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=field.coefficients_descending,
            real_root_index=root_index,
        )
        real_embedding = RealNumberFieldEmbedding(presentation=field, root=real_root)
        records.append(
            RealNumberFieldEmbeddingRecord._from_kernel(
                embedding=real_embedding,
                isolating_interval=admission.real_isolating_intervals[root_index],
            )
        )

    conjugate_pairs: list[NumberFieldConjugatePair] = []
    for pair_offset in range(pair_count):
        negative_index = real_count + 2 * pair_offset
        positive_index = negative_index + 1
        negative_root = ComplexAlgebraicValue._from_admitted_polynomial(
            polynomial=field.coefficients_descending,
            root_index=negative_index,
        )
        positive_root = ComplexAlgebraicValue._from_admitted_polynomial(
            polynomial=field.coefficients_descending,
            root_index=positive_index,
        )
        negative_embedding = ComplexNumberFieldEmbedding(
            presentation=field, root=negative_root
        )
        positive_embedding = ComplexNumberFieldEmbedding(
            presentation=field, root=positive_root
        )
        negative_rectangle = _isolate_complex_algebraic(
            negative_root,
            backend_root_index=backend_root_indices[negative_index],
            candidates=admission.complex_isolating_rectangles,
        )
        positive_rectangle = negative_rectangle.conjugate()
        records.extend(
            (
                ComplexNumberFieldEmbeddingRecord._from_kernel(
                    embedding=negative_embedding,
                    isolating_rectangle=negative_rectangle,
                    half_plane="NEGATIVE_IMAGINARY",
                ),
                ComplexNumberFieldEmbeddingRecord._from_kernel(
                    embedding=positive_embedding,
                    isolating_rectangle=positive_rectangle,
                    half_plane="POSITIVE_IMAGINARY",
                ),
            )
        )
        conjugate_pairs.append(
            NumberFieldConjugatePair(
                negative_embedding_index=negative_index,
                positive_embedding_index=positive_index,
            )
        )

    result = NumberFieldEmbeddingProfile._from_kernel(
        field=field,
        records=tuple(records),
        signature=NumberFieldSignature(
            real_embedding_count=real_count,
            complex_conjugate_pair_count=pair_count,
        ),
        complex_conjugate_pairs=tuple(conjugate_pairs),
        defining_polynomial_discriminant=format_canonical_integer(
            int(polynomial.discriminant())
        ),
    )
    return result


__all__ = [
    "discriminant",
    "embeddings",
    "ring_of_integers",
]


def _integral_basis(
    coefficients_descending: Sequence[str], variable: str
) -> tuple[Any, Any]:
    import sympy
    from sympy.polys.numberfields import round_two

    x = sympy.Symbol(variable)
    polynomial = sum(
        sympy.Rational(parse_canonical_integer(coefficient))
        * x ** (len(coefficients_descending) - 1 - index)
        for index, coefficient in enumerate(coefficients_descending)
    )
    return cast(tuple[Any, Any], round_two(sympy.Poly(polynomial, x)))


def discriminant(coefficients_descending: Sequence[str], variable: str) -> str:
    _ring_of_integers, field_discriminant = _integral_basis(
        coefficients_descending, variable
    )
    return str(field_discriminant)


def ring_of_integers(
    coefficients_descending: Sequence[str], variable: str
) -> list[str]:
    """Return the exact integral basis expressed in the defining power basis."""
    ring, _field_discriminant = _integral_basis(coefficients_descending, variable)
    return [str(element.as_expr()) for element in ring.basis_element_pullbacks()]
