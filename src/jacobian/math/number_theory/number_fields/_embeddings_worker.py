"""One-shot SymPy kernel for exact number-field embedding profiles."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.complex import (
    RationalComplexIsolatingRectangle,
    algebraic_real_part_separation_denominator_bound,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
)
from jacobian.math.number_theory.number_fields._embedding_limits import (
    MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS,
)
from jacobian.math.number_theory.number_fields._embedding_protocol import (
    NumberFieldEmbeddingWorkerComplete,
    NumberFieldEmbeddingWorkerInvalid,
    NumberFieldEmbeddingWorkerRejected,
    NumberFieldEmbeddingWorkerRequest,
    NumberFieldEmbeddingWorkerResponse,
)


@dataclass(frozen=True, slots=True)
class _RationalRectangle:
    real_lower: Fraction
    real_upper: Fraction
    imaginary_lower: Fraction
    imaginary_upper: Fraction

    def conjugate(self) -> _RationalRectangle:
        return _RationalRectangle(
            real_lower=self.real_lower,
            real_upper=self.real_upper,
            imaginary_lower=-self.imaginary_upper,
            imaginary_upper=-self.imaginary_lower,
        )


def _backend_fraction(value: Any) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def _dyadic_floor(value: Fraction, denominator: int) -> int:
    return (value.numerator * denominator) // value.denominator


def _dyadic_ceiling(value: Fraction, denominator: int) -> int:
    return -((-value.numerator * denominator) // value.denominator)


def _normalize_real_isolator(
    lower: Fraction,
    upper: Fraction,
    *,
    grid_denominator: int,
) -> RationalIsolatingInterval:
    if lower == upper:
        endpoint = CanonicalRational.from_fraction(lower)
        return RationalIsolatingInterval(
            lower=endpoint,
            upper=endpoint,
            interval_type="SINGLETON",
        )
    lower_cell = _dyadic_floor(lower, grid_denominator)
    upper_cell = _dyadic_ceiling(upper, grid_denominator)
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
    rectangle: _RationalRectangle,
    *,
    grid_denominator: int,
) -> RationalComplexIsolatingRectangle:
    real_lower_cell = _dyadic_floor(rectangle.real_lower, grid_denominator)
    real_upper_cell = _dyadic_ceiling(rectangle.real_upper, grid_denominator)
    imaginary_lower_cell = _dyadic_floor(rectangle.imaginary_lower, grid_denominator)
    imaginary_upper_cell = _dyadic_ceiling(rectangle.imaginary_upper, grid_denominator)
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


def _rational_rectangle(lower: Any, upper: Any) -> _RationalRectangle:
    lower_real, lower_imaginary = lower.as_real_imag()
    upper_real, upper_imaginary = upper.as_real_imag()
    return _RationalRectangle(
        real_lower=_backend_fraction(lower_real),
        real_upper=_backend_fraction(upper_real),
        imaginary_lower=_backend_fraction(lower_imaginary),
        imaginary_upper=_backend_fraction(upper_imaginary),
    )


def _ordered_negative_representatives(
    rectangles: tuple[_RationalRectangle, ...],
) -> tuple[_RationalRectangle, ...]:
    negative = tuple(
        rectangle for rectangle in rectangles if rectangle.imaginary_upper < 0
    )
    positive = {
        rectangle: rectangle
        for rectangle in rectangles
        if rectangle.imaginary_lower > 0
    }
    if len(negative) * 2 != len(rectangles) or len(positive) != len(negative):
        raise RuntimeError("complex root isolators do not lie in open half-planes")

    pairs: list[tuple[_RationalRectangle, _RationalRectangle]] = []
    for negative_rectangle in negative:
        positive_rectangle = positive.pop(negative_rectangle.conjugate(), None)
        if positive_rectangle is None:
            raise RuntimeError("complex root isolators are not exact conjugate pairs")
        pairs.append((negative_rectangle, positive_rectangle))
    if positive:
        raise RuntimeError("complex root isolators have unmatched conjugates")

    def compare(
        left: tuple[_RationalRectangle, _RationalRectangle],
        right: tuple[_RationalRectangle, _RationalRectangle],
    ) -> int:
        if left is right:
            return 0
        left_positive = left[1]
        right_positive = right[1]
        if left_positive.real_upper < right_positive.real_lower:
            return -1
        if right_positive.real_upper < left_positive.real_lower:
            return 1
        if left_positive.imaginary_upper < right_positive.imaginary_lower:
            return -1
        if right_positive.imaginary_upper < left_positive.imaginary_lower:
            return 1
        raise RuntimeError("refined complex root isolators do not establish order")

    pairs.sort(key=cmp_to_key(compare))
    return tuple(negative_rectangle for negative_rectangle, _positive in pairs)


def compute_embeddings_worker_response(
    request: NumberFieldEmbeddingWorkerRequest,
) -> NumberFieldEmbeddingWorkerResponse:
    """Recognize and compute one profile with exactly one all-root isolation."""

    import sympy

    field = request.field
    variable = sympy.Symbol("x")
    polynomial = sympy.Poly.from_list(
        [
            parse_canonical_integer(coefficient)
            for coefficient in field.coefficients_descending
        ],
        gens=variable,
        domain=sympy.ZZ,
    )
    if polynomial.is_irreducible is not True:
        return NumberFieldEmbeddingWorkerInvalid(
            kind="invalid",
            reason="not_irreducible",
        )

    real_count = int(polynomial.count_roots(-sympy.oo, sympy.oo))
    pair_count = (field.degree - real_count) // 2
    # The exact Sturm signature count determines whether pair ordering needs
    # the elimination polynomial.  It does not construct complex isolators;
    # the call below remains the sole all-root isolation pass.
    real_part_separation = (
        algebraic_real_part_separation_denominator_bound(field.coefficients_descending)
        if pair_count > 1
        else 1
    )
    refinement_bits = max(
        request.root_isolation_bits,
        real_part_separation.bit_length() + 4,
    )
    if refinement_bits > MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:
        return NumberFieldEmbeddingWorkerRejected(
            kind="rejected",
            reason="pair_ordering_precision_bound",
        )

    fine_grid_denominator = 1 << refinement_bits
    backend_real_intervals, backend_complex_rectangles = polynomial.intervals(
        all=True,
        eps=sympy.Rational(1, fine_grid_denominator),
    )
    if any(multiplicity != 1 for _interval, multiplicity in backend_real_intervals):
        raise RuntimeError("irreducible polynomial returned repeated real roots")
    if any(
        multiplicity != 1 for _rectangle, multiplicity in backend_complex_rectangles
    ):
        raise RuntimeError("irreducible polynomial returned repeated complex roots")

    evidence_grid_denominator = 1 << request.evidence_grid_bits
    real_intervals = tuple(
        _normalize_real_isolator(
            _backend_fraction(lower),
            _backend_fraction(upper),
            grid_denominator=evidence_grid_denominator,
        )
        for (lower, upper), _multiplicity in backend_real_intervals
    )
    raw_complex_rectangles = tuple(
        _rational_rectangle(lower, upper)
        for (lower, upper), _multiplicity in backend_complex_rectangles
    )
    ordered_negative = _ordered_negative_representatives(raw_complex_rectangles)
    negative_rectangles = tuple(
        _normalize_complex_isolator(
            rectangle,
            grid_denominator=evidence_grid_denominator,
        )
        for rectangle in ordered_negative
    )
    if len(real_intervals) + 2 * len(negative_rectangles) != field.degree:
        raise RuntimeError("root isolation did not return a complete field signature")
    if len(real_intervals) != real_count or len(negative_rectangles) != pair_count:
        raise RuntimeError("root isolation disagrees with the exact field signature")

    return NumberFieldEmbeddingWorkerComplete(
        kind="complete",
        real_intervals=real_intervals,
        negative_complex_rectangles=negative_rectangles,
        defining_polynomial_discriminant=format_canonical_integer(
            int(polynomial.discriminant())
        ),
    )


def main() -> int:
    request = NumberFieldEmbeddingWorkerRequest.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    response = compute_embeddings_worker_response(request)
    sys.stdout.write(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
