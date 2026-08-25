"""Public operation declarations for prime-affine local arithmetic."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.prime_affine_forms._models import (
    FinitePrimeTupleFactorProduct,
    PrimeAffineIntervalCountRequest,
    PrimeAffineIntervalEnumerateRequest,
    PrimeAffineTranslationRequest,
    PrimeAffineTranslationResult,
    PrimePatternIntervalCountResult,
    PrimePatternIntervalEnumerateResult,
    PrimeTupleAdmissibilityRequest,
    PrimeTupleAdmissibilityResult,
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleIntervalResidueProfileResult,
    PrimeTupleLocalFactorRequest,
    PrimeTupleLocalFactorResult,
    PrimeTupleLocalFactorsRequest,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
    PrimeTupleWheelMembershipResult,
)
from jacobian.math.prime_affine_forms._operations import (
    compute_interval_count,
    compute_interval_enumerate,
    compute_interval_residue_profile,
    compute_local_admissibility,
    compute_local_factor,
    compute_local_factors,
    compute_residue_wheel,
    compute_residue_wheel_enumeration,
    compute_translation,
    compute_wheel_membership,
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
