"""Exact affine-tuple translation contracts and kernel."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.affine_forms.values import MAX_AFFINE_COMPONENT_DIGITS
from jacobian.math.number_theory.prime_affine_forms._interval import (
    IntervalEndpointInteger,
    require_bounded_affine_endpoints,
)
from jacobian.math.number_theory.prime_affine_forms._kernel import translated_tuple
from jacobian.math.number_theory.prime_affine_forms._models import (
    _digits,
    _run_admission,
    _validation_error,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
    MAX_AFFINE_AGGREGATE_DIGITS,
    PrimeAffineTuple,
)


class PrimeAffineTranslationRequest(StrictModel):
    """Translate every source form by the variable substitution n -> n+shift."""

    source: PrimeAffineTuple
    shift: IntervalEndpointInteger


def _admit_translation(request: PrimeAffineTranslationRequest) -> int:
    require_bounded_affine_endpoints(request.source, request.shift, label="translation")
    shift = parse_canonical_integer(request.shift)
    aggregate_digits = 0
    for form in request.source.forms:
        translated_constant = form.evaluate(shift)
        if _digits(translated_constant) > MAX_AFFINE_COMPONENT_DIGITS:
            raise _validation_error(
                "translated constant exceeds the canonical affine component bound"
            )
        aggregate_digits += _digits(form.coefficient) + _digits(translated_constant)
    if aggregate_digits > MAX_AFFINE_AGGREGATE_DIGITS:
        raise _validation_error(
            "translated affine tuple exceeds the aggregate coefficient-digit "
            f"bound {MAX_AFFINE_AGGREGATE_DIGITS}"
        )
    return shift


class PrimeAffineTranslationResult(StrictModel):
    source: PrimeAffineTuple
    shift: IntervalEndpointInteger
    translated: PrimeAffineTuple

    @classmethod
    def _from_kernel(
        cls, request: PrimeAffineTranslationRequest, *, translated: PrimeAffineTuple
    ) -> PrimeAffineTranslationResult:
        """Build after the admitted translation kernel established the tuple."""

        return cls(source=request.source, shift=request.shift, translated=translated)


def compute_translation(
    request: PrimeAffineTranslationRequest,
) -> PrimeAffineTranslationResult:
    """Apply one admitted translation and retain its exact canonical tuple."""

    shift = _run_admission(lambda: _admit_translation(request))
    return PrimeAffineTranslationResult._from_kernel(
        request,
        translated=translated_tuple(request.source, shift),
    )


__all__ = [
    "PrimeAffineTranslationRequest",
    "PrimeAffineTranslationResult",
    "compute_translation",
]
