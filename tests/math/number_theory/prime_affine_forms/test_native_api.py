"""Native public surface of prime-affine tuple operations."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.prime_affine_forms import (
    PrimeAffineTuple,
    PrimitiveIntegerAffineForm,
    enumerate_residue_wheel,
    interval_count,
    interval_enumerate,
    interval_residue_profile,
    local_admissibility,
    local_factor,
    local_factors,
    residue_wheel,
    translate_tuple,
    wheel_membership,
)
from jacobian.math.number_theory.prime_affine_forms._admissibility import (
    PrimeTupleAdmissibilityRequest,
)
from jacobian.math.number_theory.prime_affine_forms._interval import (
    PrimeAffineIntervalCountRequest,
    PrimeAffineIntervalEnumerateRequest,
)
from jacobian.math.number_theory.prime_affine_forms._local_factors import (
    PrimeTupleLocalFactorRequest,
    PrimeTupleLocalFactorsRequest,
)
from jacobian.math.number_theory.prime_affine_forms._residue_wheel import (
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
)
from jacobian.math.number_theory.prime_affine_forms._tools import (
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
from jacobian.math.number_theory.prime_affine_forms._translation import (
    PrimeAffineTranslationRequest,
)


def _form(form_id: str, coefficient: int, constant: int) -> PrimitiveIntegerAffineForm:
    return PrimitiveIntegerAffineForm(
        form_id=form_id,
        coefficient=str(coefficient),
        constant=str(constant),
    )


TWIN_PRIMES = PrimeAffineTuple(forms=(_form("n", 1, 0), _form("n_plus_2", 1, 2)))


def test_native_surface_composes_without_private_imports() -> None:
    factor = local_factor(TWIN_PRIMES, 3)
    assert factor.factor.as_integer_ratio() == (3, 4)
    assert not factor.locally_obstructed

    product = local_factors(TWIN_PRIMES, (2, 3))
    assert product.product.as_integer_ratio() == (3, 2)
    assert product.first_obstructing_prime is None

    admissibility = local_admissibility(TWIN_PRIMES)
    assert admissibility.status == "LOCALLY_ADMISSIBLE"
    assert admissibility.cutoff == 2
    assert admissibility.checked_primes == (2,)

    wheel = residue_wheel(TWIN_PRIMES, (2, 3))
    assert wheel.modulus == "6"
    assert wheel.valid_count == "1"

    enumeration = enumerate_residue_wheel(wheel)
    assert tuple((row.residue, row.components) for row in enumeration.residues) == (
        ("5", (1, 2)),
    )

    member = wheel_membership(wheel, 5)
    excluded = wheel_membership(wheel, 1)
    assert member.is_permitted and member.canonical_residue == "5"
    assert not excluded.is_permitted
    assert (excluded.first_excluded_prime, excluded.vanishing_form_ids) == (
        3,
        ("n_plus_2",),
    )

    count = interval_count(TWIN_PRIMES, 0, 12)
    matches = interval_enumerate(TWIN_PRIMES, 0, 12)
    assert count.match_count == 3
    assert (count.first_match, count.last_match) == ("3", "11")
    assert tuple(
        (match.parameter, match.prime_values) for match in matches.matches
    ) == (("3", ("3", "5")), ("5", ("5", "7")), ("11", ("11", "13")))

    identity_wheel = residue_wheel(PrimeAffineTuple(forms=(_form("n", 1, 0),)), (2,))
    profile = interval_residue_profile(identity_wheel, 24, 26)
    assert profile.survivors == ("25",)

    translated = translate_tuple(TWIN_PRIMES, 1)
    assert tuple(form.constant for form in translated.translated.forms) == ("1", "3")


def test_native_results_equal_the_wire_request_path() -> None:
    wheel = residue_wheel(TWIN_PRIMES, (2, 3))

    expected = (
        local_factor(TWIN_PRIMES, 3),
        compute_local_factor(PrimeTupleLocalFactorRequest(source=TWIN_PRIMES, prime=3)),
    )
    assert expected[0] == expected[1]

    assert local_factors(TWIN_PRIMES, (2, 3)) == compute_local_factors(
        PrimeTupleLocalFactorsRequest(source=TWIN_PRIMES, primes=(2, 3))
    )
    assert local_admissibility(TWIN_PRIMES) == compute_local_admissibility(
        PrimeTupleAdmissibilityRequest(source=TWIN_PRIMES)
    )
    assert wheel == compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=TWIN_PRIMES, primes=(2, 3))
    )
    assert enumerate_residue_wheel(wheel) == compute_residue_wheel_enumeration(
        PrimeTupleResidueWheelEnumerationRequest(wheel=wheel)
    )
    assert wheel_membership(wheel, -7) == compute_wheel_membership(
        PrimeTupleWheelMembershipRequest(wheel=wheel, value="-7")
    )
    assert interval_count(TWIN_PRIMES, 4, 8) == compute_interval_count(
        PrimeAffineIntervalCountRequest(source=TWIN_PRIMES, lower="4", upper="8")
    )
    assert interval_enumerate(TWIN_PRIMES, 4, 8) == compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(source=TWIN_PRIMES, lower="4", upper="8")
    )
    assert interval_residue_profile(wheel, 0, 6) == compute_interval_residue_profile(
        PrimeTupleIntervalResidueProfileRequest(wheel=wheel, lower="0", upper="6")
    )
    assert translate_tuple(TWIN_PRIMES, -5) == compute_translation(
        PrimeAffineTranslationRequest(source=TWIN_PRIMES, shift="-5")
    )


def test_native_calls_reject_out_of_envelope_requests() -> None:
    with pytest.raises(OperationDomainValidationError):
        local_factor(TWIN_PRIMES, 8_209)
    with pytest.raises(OperationDomainValidationError):
        local_factor(TWIN_PRIMES, 15)
    with pytest.raises(OperationDomainValidationError):
        local_factors(TWIN_PRIMES, (3, 2))
    with pytest.raises(OperationDomainValidationError):
        residue_wheel(TWIN_PRIMES, (2, 2))
    with pytest.raises(OperationDomainValidationError):
        interval_count(PrimeAffineTuple(forms=(_form("n", 1, 0),)), 0, 100_000)
    with pytest.raises(OperationDomainValidationError):
        interval_enumerate(PrimeAffineTuple(forms=(_form("n", 1, 0),)), 2**64, 2**64)
    with pytest.raises(OperationDomainValidationError):
        translate_tuple(
            PrimeAffineTuple(forms=(_form("large", 1, int("9" * 256)),)),
            10**65 + 1,
        )
    large_wheel = residue_wheel(PrimeAffineTuple(forms=(_form("n", 1, 0),)), (8_209,))
    with pytest.raises(OperationDomainValidationError):
        enumerate_residue_wheel(large_wheel)
