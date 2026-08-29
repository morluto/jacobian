"""Public operation declarations for prime-affine local arithmetic."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.prime_affine_forms._admissibility import (
    PrimeTupleAdmissibilityRequest,
    PrimeTupleAdmissibilityResult,
)
from jacobian.math.number_theory.prime_affine_forms._interval import (
    PrimeAffineIntervalCountRequest,
    PrimeAffineIntervalEnumerateRequest,
    PrimePatternIntervalCountResult,
    PrimePatternIntervalEnumerateResult,
)
from jacobian.math.number_theory.prime_affine_forms._local_factors import (
    FinitePrimeTupleFactorProduct,
    PrimeTupleLocalFactorRequest,
    PrimeTupleLocalFactorResult,
    PrimeTupleLocalFactorsRequest,
)
from jacobian.math.number_theory.prime_affine_forms._residue_wheel import (
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleIntervalResidueProfileResult,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
    PrimeTupleWheelMembershipResult,
)
from jacobian.math.number_theory.prime_affine_forms._translation import (
    PrimeAffineTranslationRequest,
    PrimeAffineTranslationResult,
    parse_translation_shift,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    enumerate_residue_wheel as native_enumerate_residue_wheel,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    interval_count as native_interval_count,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    interval_enumerate as native_interval_enumerate,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    interval_residue_profile as native_interval_residue_profile,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    local_admissibility as native_local_admissibility,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    local_factor as native_local_factor,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    local_factors as native_local_factors,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    residue_wheel as native_residue_wheel,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    translate_tuple as native_translate_tuple,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    wheel_membership as native_wheel_membership,
)


def compute_local_admissibility(
    request: PrimeTupleAdmissibilityRequest,
) -> PrimeTupleAdmissibilityResult:
    return native_local_admissibility(request.source)


def compute_local_factor(
    request: PrimeTupleLocalFactorRequest,
) -> PrimeTupleLocalFactorResult:
    return native_local_factor(request.source, request.prime)


def compute_local_factors(
    request: PrimeTupleLocalFactorsRequest,
) -> FinitePrimeTupleFactorProduct:
    return native_local_factors(request.source, request.primes)


def compute_translation(
    request: PrimeAffineTranslationRequest,
) -> PrimeAffineTranslationResult:
    return native_translate_tuple(
        request.source, parse_translation_shift(request.source, request.shift)
    )


def compute_residue_wheel(
    request: PrimeTupleResidueWheelRequest,
) -> PrimeTupleResidueWheel:
    return native_residue_wheel(request.source, request.primes)


def compute_residue_wheel_enumeration(
    request: PrimeTupleResidueWheelEnumerationRequest,
) -> PrimeTupleResidueWheelEnumeration:
    return native_enumerate_residue_wheel(request.wheel)


def compute_wheel_membership(
    request: PrimeTupleWheelMembershipRequest,
) -> PrimeTupleWheelMembershipResult:
    return native_wheel_membership(
        request.wheel, parse_canonical_integer(request.value)
    )


def compute_interval_residue_profile(
    request: PrimeTupleIntervalResidueProfileRequest,
) -> PrimeTupleIntervalResidueProfileResult:
    return native_interval_residue_profile(
        request.wheel,
        parse_canonical_integer(request.lower),
        parse_canonical_integer(request.upper),
    )


def compute_interval_count(
    request: PrimeAffineIntervalCountRequest,
) -> PrimePatternIntervalCountResult:
    return native_interval_count(
        request.source,
        request.lower,
        request.upper,
    )


def compute_interval_enumerate(
    request: PrimeAffineIntervalEnumerateRequest,
) -> PrimePatternIntervalEnumerateResult:
    return native_interval_enumerate(
        request.source,
        request.lower,
        request.upper,
    )


def _operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...],
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_TWIN_PRIME_SOURCE = {
    "forms": [
        {"form_id": "n", "coefficient": "1", "constant": "0"},
        {"form_id": "n_plus_2", "coefficient": "1", "constant": "2"},
    ]
}

_TWIN_PRIME_WHEEL = {
    "source": _TWIN_PRIME_SOURCE,
    "primes": [2, 3],
    "local_rows": [
        {
            "prime": 2,
            "bad_residues": [
                {"residue": 0, "form_ids": ["n", "n_plus_2"]},
            ],
            "bad_count": 1,
            "valid_count": 1,
        },
        {
            "prime": 3,
            "bad_residues": [
                {"residue": 0, "form_ids": ["n"]},
                {"residue": 1, "form_ids": ["n_plus_2"]},
            ],
            "bad_count": 2,
            "valid_count": 1,
        },
    ],
    "modulus": "6",
    "valid_count": "1",
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _operation(
        "number_theory.prime_affine_forms.local_factor.compute",
        "Compute one prime-affine local factor",
        "For a finite tuple of primitive affine forms and one bounded modulus "
        "p, return the complete residue partition and the exact Hardy-Littlewood "
        "density term (1-nu_p/p)/(1-1/p)^k.",
        PrimeTupleLocalFactorRequest,
        PrimeTupleLocalFactorResult,
        compute_local_factor,
        "number-theory",
        "local-obstruction",
        "hardy-littlewood",
        "exact",
        examples=(
            example(
                "twin_prime_local_factor_mod_3",
                "Compute the complete modulo-3 local profile of n and n+2; "
                "the forms must be distinct, primitive, and nonconstant.",
                {"source": _TWIN_PRIME_SOURCE, "prime": 3},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.local_factors.compute",
        "Compute a finite prime-affine factor product",
        "Compute compact exact local-factor rows for a finite canonical prime "
        "set and their exact finite product. The result makes no infinite "
        "singular-series or asymptotic claim.",
        PrimeTupleLocalFactorsRequest,
        FinitePrimeTupleFactorProduct,
        compute_local_factors,
        "number-theory",
        "prime-tuple",
        "finite-product",
        "exact",
        examples=(
            example(
                "twin_prime_factors_2_3",
                "Compute the finite local-factor product for p=2 and p=3; "
                "the prime tuple must be strictly increasing and duplicate-free.",
                {"source": _TWIN_PRIME_SOURCE, "primes": [2, 3]},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.local_admissibility.compute",
        "Decide local admissibility of primitive affine forms",
        "Check every prime through the form count k, then use primitivity to "
        "prove that each form excludes at most one residue and every p>k has "
        "a permitted class. This does not assert simultaneous primality.",
        PrimeTupleAdmissibilityRequest,
        PrimeTupleAdmissibilityResult,
        compute_local_admissibility,
        "number-theory",
        "prime-tuple",
        "admissibility",
        "exact",
        examples=(
            example(
                "twin_prime_admissibility",
                "Decide local admissibility of n and n+2; every supplied form "
                "must be primitive and have a nonzero coefficient.",
                {"source": _TWIN_PRIME_SOURCE},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.residue_wheel.compute",
        "Construct a compact exact prime-affine residue wheel",
        "Return a source-bound product of valid local residue sets, its CRT "
        "modulus, and exact valid count without expanding every combined residue.",
        PrimeTupleResidueWheelRequest,
        PrimeTupleResidueWheel,
        compute_residue_wheel,
        "number-theory",
        "prime-tuple",
        "crt",
        "exact",
        examples=(
            example(
                "twin_prime_wheel_6",
                "Construct the compact modulo-6 wheel of n and n+2 from primes "
                "2 and 3; primes must be distinct and strictly increasing.",
                {"source": _TWIN_PRIME_SOURCE, "primes": [2, 3]},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.residue_wheel.enumerate.compute",
        "Enumerate every residue of a compact prime-affine wheel",
        "Materialize every permitted CRT residue and its aligned prime components "
        "from a supplied compact wheel under separate residue, work, and output "
        "bounds.",
        PrimeTupleResidueWheelEnumerationRequest,
        PrimeTupleResidueWheelEnumeration,
        compute_residue_wheel_enumeration,
        "number-theory",
        "prime-tuple",
        "crt",
        "enumeration",
        "exact",
        examples=(
            example(
                "enumerate_twin_prime_wheel_6",
                "Enumerate the sole permitted residue of the compact modulo-6 "
                "twin-prime wheel; the supplied wheel must be complete and "
                "source-bound.",
                {"wheel": _TWIN_PRIME_WHEEL},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.wheel_membership.compute",
        "Check membership in a prime-affine residue wheel",
        "Reduce one exact integer through a source-bound residue wheel and "
        "return its CRT components plus the first excluded prime and vanishing "
        "forms when it is not locally permitted.",
        PrimeTupleWheelMembershipRequest,
        PrimeTupleWheelMembershipResult,
        compute_wheel_membership,
        "number-theory",
        "prime-tuple",
        "crt",
        "membership",
        "exact",
        examples=(
            example(
                "twin_prime_wheel_member",
                "Check that 5 is permitted by the exact modulo-6 twin-prime "
                "wheel; the wheel must be a complete source-bound result.",
                {"wheel": _TWIN_PRIME_WHEEL, "value": "5"},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.interval_count.compute",
        "Count exact prime-affine matches on an interval",
        "Count every integer n in a bounded closed interval for which all source "
        "forms have ordinary positive-prime values. Every positive candidate "
        "is admitted below SymPy's deterministic 2^64 primality boundary.",
        PrimeAffineIntervalCountRequest,
        PrimePatternIntervalCountResult,
        compute_interval_count,
        "number-theory",
        "prime-tuple",
        "interval",
        "exact",
        examples=(
            example(
                "twin_primes_through_10_count",
                "Count twin-prime starts from 0 through 10 exactly; lower and "
                "upper must form a nonempty canonical integer interval.",
                {"source": _TWIN_PRIME_SOURCE, "lower": "0", "upper": "10"},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.interval_enumerate.compute",
        "Enumerate exact prime-affine matches on an interval",
        "Return every integer n and aligned affine-value tuple for which all "
        "forms are ordinary positive primes on a bounded closed interval. The "
        "complete match output has a stricter bound than count-only execution.",
        PrimeAffineIntervalEnumerateRequest,
        PrimePatternIntervalEnumerateResult,
        compute_interval_enumerate,
        "number-theory",
        "prime-tuple",
        "interval",
        "enumeration",
        "exact",
        examples=(
            example(
                "twin_primes_through_10",
                "Enumerate twin-prime starts from 0 through 10 exactly; lower "
                "and upper must form a nonempty canonical integer interval.",
                {"source": _TWIN_PRIME_SOURCE, "lower": "0", "upper": "10"},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.interval_residue_profile.compute",
        "Enumerate interval survivors of a prime-affine wheel",
        "Return every integer in a bounded closed interval whose residue is "
        "permitted by the supplied exact wheel. Wheel survival is only local "
        "divisibility data and does not assert primality.",
        PrimeTupleIntervalResidueProfileRequest,
        PrimeTupleIntervalResidueProfileResult,
        compute_interval_residue_profile,
        "number-theory",
        "prime-tuple",
        "crt",
        "interval",
        "exact",
        examples=(
            example(
                "twin_prime_wheel_survivors",
                "Enumerate modulo-6 twin-prime wheel survivors from 0 through "
                "12; the supplied wheel must be complete and source-bound.",
                {"wheel": _TWIN_PRIME_WHEEL, "lower": "0", "upper": "12"},
            ),
        ),
    ),
    _operation(
        "number_theory.prime_affine_forms.translation.compute",
        "Translate a primitive affine-form tuple",
        "Apply n -> n+c exactly to every labelled primitive affine form, "
        "preserving form IDs and producing a canonical tuple whose local factors "
        "are transported by residue translation.",
        PrimeAffineTranslationRequest,
        PrimeAffineTranslationResult,
        compute_translation,
        "number-theory",
        "prime-tuple",
        "translation",
        "exact",
        examples=(
            example(
                "translate_twin_prime_tuple",
                "Translate n and n+2 by one to obtain n+1 and n+3; the resulting "
                "constants must remain within the canonical component bound.",
                {"source": _TWIN_PRIME_SOURCE, "shift": "1"},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
