from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from math import prod

import pytest
from pydantic import ValidationError
from sympy import primerange

import jacobian.math.number_theory.prime_affine_forms._interval as interval_contracts
import jacobian.math.number_theory.prime_affine_forms._translation as translation_contracts
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.affine_forms import IntegerAffineForm
from jacobian.math.number_theory.prime_affine_forms import (
    PrimeAffineTuple,
    PrimitiveIntegerAffineForm,
)
from jacobian.math.number_theory.prime_affine_forms._admissibility import (
    PrimeTupleAdmissibilityRequest,
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
from jacobian.math.number_theory.prime_affine_forms._models import (
    PrimeTupleLocalSummary,
)
from jacobian.math.number_theory.prime_affine_forms._residue_wheel import (
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
)
from jacobian.math.number_theory.prime_affine_forms._tools import (
    TOOLS,
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
    PrimeAffineTranslationResult,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
    MAX_AFFINE_AGGREGATE_DIGITS,
)


def _form(form_id: str, coefficient: int, constant: int) -> PrimitiveIntegerAffineForm:
    return PrimitiveIntegerAffineForm(
        form_id=form_id,
        coefficient=str(coefficient),
        constant=str(constant),
    )


def _tuple(*forms: PrimitiveIntegerAffineForm) -> PrimeAffineTuple:
    return PrimeAffineTuple(forms=forms)


TWIN_PRIMES = _tuple(_form("n_plus_2", 1, 2), _form("n", 1, 0))


def test_affine_tuple_is_canonical_and_closed() -> None:
    constant_form = IntegerAffineForm(
        form_id="constant",
        coefficient="0",
        constant="-4",
    )
    assert constant_form.evaluate(999) == -4
    with pytest.raises(ValidationError):
        IntegerAffineForm(form_id="zero", coefficient="0", constant="0")

    assert tuple(form.form_id for form in TWIN_PRIMES.forms) == ("n", "n_plus_2")
    assert TWIN_PRIMES.forms[1].evaluate(7) == 9

    with pytest.raises(ValidationError):
        _form("constant", 0, 1)
    with pytest.raises(ValidationError):
        _form("nonprimitive", 2, 2)
    with pytest.raises(ValidationError):
        _tuple(_form("same", 1, 0), _form("same", 1, 2))
    with pytest.raises(ValidationError):
        _tuple(_form("first", 1, 0), _form("second", 1, 0))

    with pytest.raises(ValidationError) as exc_info:
        _form("coded_zero", 0, 1)
    assert (
        exc_info.value.errors()[0]["type"] == "prime_affine_form.coefficient_required"
    )
    with pytest.raises(ValidationError) as exc_info:
        _form("coded_nonprimitive", 2, 2)
    assert exc_info.value.errors()[0]["type"] == "prime_affine_form.primitive_required"

    _form("f" * 32, 1, int("9" * 256))
    with pytest.raises(ValidationError):
        _form("f" * 33, 1, 0)
    with pytest.raises(ValidationError):
        _form("too_many_digits", 1, int("9" * 257))

    _tuple(*(_form(f"f{index:03d}", 1, index) for index in range(512)))
    with pytest.raises(ValidationError):
        _tuple(*(_form(f"f{index:03d}", 1, index) for index in range(513)))


def test_local_factor_returns_complete_residue_partition() -> None:
    result = compute_local_factor(
        PrimeTupleLocalFactorRequest(source=TWIN_PRIMES, prime=3)
    )

    assert tuple(
        (row.residue, row.vanishing_form_ids) for row in result.residue_rows
    ) == ((0, ("n",)), (1, ("n_plus_2",)), (2, ()))
    assert (result.bad_count, result.valid_count, result.locally_obstructed) == (
        2,
        1,
        False,
    )
    assert result.factor.as_integer_ratio() == (3, 4)


def test_local_factor_handles_divisible_coefficients_and_overlaps() -> None:
    no_root = compute_local_factor(
        PrimeTupleLocalFactorRequest(
            source=_tuple(_form("two_n_plus_1", 2, 1)),
            prime=2,
        )
    )
    assert no_root.bad_count == 0
    assert no_root.factor.as_integer_ratio() == (2, 1)

    overlap = compute_local_factor(
        PrimeTupleLocalFactorRequest(source=TWIN_PRIMES, prime=2)
    )
    assert overlap.residue_rows[0].vanishing_form_ids == ("n", "n_plus_2")
    assert overlap.bad_count == 1
    assert overlap.factor.as_integer_ratio() == (2, 1)


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_local_factor_agrees_with_direct_residue_oracle(prime: int) -> None:
    source = _tuple(
        _form("two_n_plus_1", 2, 1),
        _form("three_n_plus_2", 3, 2),
        _form("five_n_minus_1", 5, -1),
    )
    result = compute_local_factor(
        PrimeTupleLocalFactorRequest(source=source, prime=prime)
    )

    expected_rows = tuple(
        (
            residue,
            tuple(
                form.form_id
                for form in source.forms
                if form.evaluate(residue) % prime == 0
            ),
        )
        for residue in range(prime)
    )
    bad_count = sum(bool(form_ids) for _, form_ids in expected_rows)
    expected_factor = (
        Fraction(prime - bad_count, prime)
        / Fraction(prime - 1, prime) ** source.form_count
    )

    assert (
        tuple((row.residue, row.vanishing_form_ids) for row in result.residue_rows)
        == expected_rows
    )
    assert result.factor.as_fraction() == expected_factor


def test_local_factor_result_round_trips_structurally() -> None:
    result = compute_local_factor(
        PrimeTupleLocalFactorRequest(source=TWIN_PRIMES, prime=3)
    )
    payload = result.model_dump(mode="json")

    wrong_factor = deepcopy(payload)
    wrong_factor["factor"] = {"num": "1", "den": "1"}
    assert PrimeTupleLocalFactorResult.model_validate(wrong_factor) != result

    wrong_source = deepcopy(payload)
    wrong_source["source"]["forms"][1]["constant"] = "4"
    assert PrimeTupleLocalFactorResult.model_validate(wrong_source) != result

    wrong_row = deepcopy(payload)
    wrong_row["residue_rows"][1]["vanishing_form_ids"] = []
    assert PrimeTupleLocalFactorResult.model_validate(wrong_row) != result
    assert PrimeTupleLocalFactorResult.model_validate(payload) == result


def test_finite_local_factor_product_and_obstruction() -> None:
    result = compute_local_factors(
        PrimeTupleLocalFactorsRequest(source=TWIN_PRIMES, primes=(2, 3))
    )
    assert result.product.as_integer_ratio() == (3, 2)
    assert result.first_obstructing_prime is None

    obstructed = _tuple(
        _form("n", 1, 0),
        _form("n_plus_2", 1, 2),
        _form("n_plus_4", 1, 4),
    )
    zero = compute_local_factors(
        PrimeTupleLocalFactorsRequest(source=obstructed, primes=(2, 3, 5))
    )
    assert zero.product.as_integer_ratio() == (0, 1)
    assert zero.first_obstructing_prime == 3

    payload = zero.model_dump(mode="json")
    payload["first_obstructing_prime"] = 2
    forged = FinitePrimeTupleFactorProduct.model_validate(payload)
    assert forged != zero


def test_local_admissibility_uses_complete_finite_cutoff() -> None:
    admissible = compute_local_admissibility(
        PrimeTupleAdmissibilityRequest(source=TWIN_PRIMES)
    )
    assert admissible.status == "LOCALLY_ADMISSIBLE"
    assert admissible.cutoff == 2
    assert admissible.checked_primes == (2,)
    assert admissible.large_prime_lower_bound == 3
    assert admissible.maximum_large_prime_bad_residues == 2

    consecutive = _tuple(_form("n", 1, 0), _form("n_plus_1", 1, 1))
    obstructed = compute_local_admissibility(
        PrimeTupleAdmissibilityRequest(source=consecutive)
    )
    assert obstructed.status == "LOCALLY_OBSTRUCTED"
    assert obstructed.least_obstructing_prime == 2
    assert obstructed.local_rows[0].valid_count == 0

    admissible_triple = _tuple(
        _form("n", 1, 0),
        _form("n_plus_2", 1, 2),
        _form("n_plus_6", 1, 6),
    )
    triple_result = compute_local_admissibility(
        PrimeTupleAdmissibilityRequest(source=admissible_triple)
    )
    assert triple_result.status == "LOCALLY_ADMISSIBLE"

    planted_obstruction = _tuple(
        _form("n", 1, 0),
        _form("n_plus_2", 1, 2),
        _form("n_plus_4", 1, 4),
    )
    planted_result = compute_local_admissibility(
        PrimeTupleAdmissibilityRequest(source=planted_obstruction)
    )
    assert planted_result.status == "LOCALLY_OBSTRUCTED"
    assert planted_result.least_obstructing_prime == 3

    huge_coefficient = _tuple(_form("huge_coefficient", int("9" * 256), 1))
    huge_result = compute_local_admissibility(
        PrimeTupleAdmissibilityRequest(source=huge_coefficient)
    )
    assert huge_result.status == "LOCALLY_ADMISSIBLE"
    assert huge_result.cutoff == 1
    assert huge_result.checked_primes == ()


def test_local_admissibility_supports_the_343_form_demand_scale() -> None:
    small_primorial = prod(int(prime) for prime in primerange(2, 344))
    source = _tuple(
        *(_form(f"h{index:03d}", 1, index * small_primorial) for index in range(343))
    )

    result = compute_local_admissibility(PrimeTupleAdmissibilityRequest(source=source))

    assert result.status == "LOCALLY_ADMISSIBLE"
    assert result.cutoff == 343
    assert result.checked_primes == tuple(int(p) for p in primerange(2, 344))
    assert all(row.bad_count == 1 for row in result.local_rows)


def test_translation_preserves_local_factors_and_form_ids() -> None:
    translated = compute_translation(
        PrimeAffineTranslationRequest(source=TWIN_PRIMES, shift="1")
    )
    assert tuple(form.form_id for form in translated.translated.forms) == (
        "n",
        "n_plus_2",
    )
    assert tuple(form.constant for form in translated.translated.forms) == ("1", "3")

    before = compute_local_factor(
        PrimeTupleLocalFactorRequest(source=TWIN_PRIMES, prime=5)
    )
    after = compute_local_factor(
        PrimeTupleLocalFactorRequest(source=translated.translated, prime=5)
    )
    assert (before.bad_count, before.valid_count, before.factor) == (
        after.bad_count,
        after.valid_count,
        after.factor,
    )

    payload = translated.model_dump(mode="json")
    payload["translated"]["forms"][0]["constant"] = "2"
    forged = PrimeAffineTranslationResult.model_validate(payload)
    assert forged != translated


def test_translation_rejects_translated_tuple_exceeding_aggregate_digit_bound() -> None:
    source = _tuple(
        *(_form(f"f{index:03d}", 10**127 + index, 1) for index in range(512))
    )
    with pytest.raises(OperationDomainValidationError):
        compute_translation(
            PrimeAffineTranslationRequest(source=source, shift=str(10**63))
        )


def test_translation_admits_translated_tuple_at_the_aggregate_digit_bound() -> None:
    source = _tuple(
        *(_form(f"f{index:03d}", 10**127 + index, 1) for index in range(512))
    )
    result = compute_translation(
        PrimeAffineTranslationRequest(source=source, shift="1")
    )
    assert result.translated.form_count == 512
    assert result.translated.forms[0].form_id == "f000"
    assert all(len(form.constant) == 128 for form in result.translated.forms)


def test_translation_admits_a_65_digit_cancelling_shift() -> None:
    """Translation is bounded by its canonical result, not shift syntax length."""

    shift = 10**64
    source = _tuple(_form("cancelled", 1, -shift))

    result = compute_translation(
        PrimeAffineTranslationRequest(source=source, shift=str(shift))
    )

    assert result.translated.forms == (_form("cancelled", 1, 0),)


def test_interval_preflights_oversized_endpoints_before_integer_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parse(_: str) -> int:
        raise AssertionError("endpoint reached integer parsing")

    monkeypatch.setattr(interval_contracts, "parse_canonical_integer", fail_parse)

    with pytest.raises(
        OperationDomainValidationError, match="source-sensitive pre-parse"
    ):
        compute_interval_count(
            PrimeAffineIntervalCountRequest(
                source=TWIN_PRIMES,
                lower="9" * 258,
                upper="9" * 258,
            )
        )


def test_residue_profile_preflights_oversized_endpoints_before_integer_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=TWIN_PRIMES, primes=(2, 3))
    )

    def fail_parse(_: str) -> int:
        raise AssertionError("endpoint reached integer parsing")

    monkeypatch.setattr(interval_contracts, "parse_canonical_integer", fail_parse)

    with pytest.raises(
        OperationDomainValidationError, match="source-sensitive pre-parse"
    ):
        compute_interval_residue_profile(
            PrimeTupleIntervalResidueProfileRequest(
                wheel=wheel,
                lower="9" * 258,
                upper="9" * 258,
            )
        )


def test_translation_preflights_oversized_shift_before_integer_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parse(_: str) -> int:
        raise AssertionError("shift reached integer parsing")

    monkeypatch.setattr(translation_contracts, "parse_canonical_integer", fail_parse)

    with pytest.raises(
        OperationDomainValidationError, match="source-sensitive pre-parse"
    ):
        compute_translation(
            PrimeAffineTranslationRequest(source=TWIN_PRIMES, shift="9" * 258)
        )


def test_residue_wheel_and_membership_compose_without_reconstruction() -> None:
    wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=TWIN_PRIMES, primes=(2, 3))
    )
    assert wheel.modulus == "6"
    assert wheel.valid_count == "1"
    enumeration = compute_residue_wheel_enumeration(
        PrimeTupleResidueWheelEnumerationRequest(wheel=wheel)
    )
    assert tuple((row.residue, row.components) for row in enumeration.residues) == (
        ("5", (1, 2)),
    )
    assert tuple(
        residue
        for residue in range(6)
        if all(
            form.evaluate(residue) % prime != 0
            for prime in (2, 3)
            for form in TWIN_PRIMES.forms
        )
    ) == (5,)

    serialized = wheel.model_dump(mode="json")
    request = PrimeTupleWheelMembershipRequest.model_validate(
        {"wheel": serialized, "value": "5"}
    )
    member = compute_wheel_membership(request)
    assert member.is_permitted
    assert member.canonical_residue == "5"

    excluded = compute_wheel_membership(
        PrimeTupleWheelMembershipRequest(wheel=wheel, value="1")
    )
    assert not excluded.is_permitted
    assert excluded.first_excluded_prime == 3
    assert excluded.vanishing_form_ids == ("n_plus_2",)

    first_excluded = compute_wheel_membership(
        PrimeTupleWheelMembershipRequest(wheel=wheel, value="0")
    )
    assert first_excluded.first_excluded_prime == 2
    assert first_excluded.vanishing_form_ids == ("n", "n_plus_2")


def test_empty_prime_wheel_is_the_modulus_one_identity() -> None:
    wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=TWIN_PRIMES, primes=())
    )
    assert wheel.modulus == "1"
    assert wheel.valid_count == "1"
    enumeration = compute_residue_wheel_enumeration(
        PrimeTupleResidueWheelEnumerationRequest(wheel=wheel)
    )
    assert tuple((row.residue, row.components) for row in enumeration.residues) == (
        ("0", ()),
    )
    result = compute_wheel_membership(
        PrimeTupleWheelMembershipRequest(wheel=wheel, value="-123")
    )
    assert result.is_permitted
    assert result.canonical_residue == "0"
    assert result.first_excluded_prime is None


def test_wheel_result_rejects_component_and_source_mutations() -> None:
    wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=TWIN_PRIMES, primes=(2, 3))
    )
    payload = wheel.model_dump(mode="json")

    wrong_source = deepcopy(payload)
    wrong_source["source"]["forms"][1]["constant"] = "4"
    forged_wheel = PrimeTupleResidueWheel.model_validate(wrong_source)
    assert forged_wheel != wheel

    enumeration = compute_residue_wheel_enumeration(
        PrimeTupleResidueWheelEnumerationRequest(wheel=wheel)
    )
    wrong_component = enumeration.model_dump(mode="json")
    wrong_component["residues"][0]["components"] = [0, 2]
    forged_enumeration = PrimeTupleResidueWheelEnumeration.model_validate(
        wrong_component
    )
    assert forged_enumeration != enumeration

    oversized_scalar = deepcopy(payload)
    oversized_scalar["modulus"] = "9" * 4_097
    with pytest.raises(ValidationError):
        PrimeTupleWheelMembershipRequest.model_validate(
            {"wheel": oversized_scalar, "value": "1"}
        )


def test_wheel_consumers_reject_a_structurally_valid_forged_wheel() -> None:
    wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=TWIN_PRIMES, primes=(2, 3))
    )
    forged = wheel.model_dump(mode="json")
    forged["modulus"] = "0"

    with pytest.raises(OperationDomainValidationError, match="wheel modulus"):
        compute_residue_wheel_enumeration(
            PrimeTupleResidueWheelEnumerationRequest.model_validate({"wheel": forged})
        )
    with pytest.raises(OperationDomainValidationError, match="wheel modulus"):
        compute_wheel_membership(
            PrimeTupleWheelMembershipRequest.model_validate(
                {"wheel": forged, "value": "5"}
            )
        )
    with pytest.raises(OperationDomainValidationError, match="wheel modulus"):
        compute_interval_residue_profile(
            PrimeTupleIntervalResidueProfileRequest.model_validate(
                {"wheel": forged, "lower": "0", "upper": "12"}
            )
        )


def test_interval_count_and_enumeration_are_exact_and_aligned() -> None:
    count = compute_interval_count(
        PrimeAffineIntervalCountRequest(
            source=TWIN_PRIMES,
            lower="0",
            upper="12",
        )
    )
    enumeration = compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(
            source=TWIN_PRIMES,
            lower="0",
            upper="12",
        )
    )

    assert count.match_count == 3
    assert (count.first_match, count.last_match) == ("3", "11")
    assert tuple(
        (match.parameter, match.prime_values) for match in enumeration.matches
    ) == (("3", ("3", "5")), ("5", ("5", "7")), ("11", ("11", "13")))
    assert count.match_count == len(enumeration.matches)

    missing_endpoints = count.model_dump(mode="json")
    missing_endpoints["first_match"] = None
    missing_endpoints["last_match"] = None
    with pytest.raises(ValidationError, match="if and only if match_count is positive"):
        PrimePatternIntervalCountResult.model_validate(missing_endpoints)

    impossible_count = count.model_dump(mode="json")
    impossible_count["lower"] = "3"
    impossible_count["upper"] = "3"
    impossible_count["interval_size"] = 1
    impossible_count["affine_values_examined"] = 2
    impossible_count["match_count"] = 2
    impossible_count["first_match"] = "3"
    impossible_count["last_match"] = "3"
    with pytest.raises(ValidationError, match="cannot exceed interval_size"):
        PrimePatternIntervalCountResult.model_validate(impossible_count)

    payload = enumeration.model_dump(mode="json")
    payload["matches"][0]["prime_values"][1] = "7"
    forged = PrimePatternIntervalEnumerateResult.model_validate(payload)
    assert forged != enumeration

    payload = enumeration.model_dump(mode="json")
    payload["matches"][0]["parameter"] = "-1"
    with pytest.raises(ValidationError, match="must lie in the interval"):
        PrimePatternIntervalEnumerateResult.model_validate(payload)


def test_wheel_survival_is_not_mislabelled_as_primality() -> None:
    identity_form = _tuple(_form("n", 1, 0))
    wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=identity_form, primes=(2,))
    )
    profile = compute_interval_residue_profile(
        PrimeTupleIntervalResidueProfileRequest(
            wheel=wheel,
            lower="2",
            upper="3",
        )
    )
    primes = compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(
            source=identity_form,
            lower="2",
            upper="3",
        )
    )

    assert profile.survivors == ("3",)
    assert tuple(match.parameter for match in primes.matches) == ("2", "3")

    composite_survivor = compute_interval_residue_profile(
        PrimeTupleIntervalResidueProfileRequest(
            wheel=wheel,
            lower="25",
            upper="25",
        )
    )
    composite_matches = compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(
            source=identity_form,
            lower="25",
            upper="25",
        )
    )
    assert composite_survivor.survivors == ("25",)
    assert composite_matches.matches == ()


def test_request_boundaries_reject_before_expansion() -> None:
    identity_form = _tuple(_form("n", 1, 0))

    compute_local_factor(
        PrimeTupleLocalFactorRequest(source=identity_form, prime=8_191)
    )
    with pytest.raises(OperationDomainValidationError):
        compute_local_factor(
            PrimeTupleLocalFactorRequest(source=identity_form, prime=8_209)
        )
    with pytest.raises(OperationDomainValidationError):
        compute_local_factor(
            PrimeTupleLocalFactorRequest(source=identity_form, prime=15)
        )
    with pytest.raises(ValidationError):
        PrimeTupleLocalSummary(
            prime=15,
            bad_residues=(),
            bad_count=0,
            valid_count=15,
        )
    with pytest.raises(OperationDomainValidationError):
        compute_local_factors(
            PrimeTupleLocalFactorsRequest(source=identity_form, primes=(3, 2))
        )
    with pytest.raises(OperationDomainValidationError):
        compute_local_factors(
            PrimeTupleLocalFactorsRequest(source=identity_form, primes=(2, 2))
        )

    compute_interval_count(
        PrimeAffineIntervalCountRequest(source=identity_form, lower="0", upper="99999")
    )
    with pytest.raises(OperationDomainValidationError):
        compute_interval_count(
            PrimeAffineIntervalCountRequest(
                source=identity_form, lower="0", upper="100000"
            )
        )

    # A 65-digit endpoint is harmless when its affine values cancel to the
    # admitted deterministic primality range.  The two existing interval scans
    # are still charged by the 2 * interval_size * form_count envelope.
    shifted_identity = _tuple(_form("shifted_n", 1, -(10**64)))
    accepted_cancellation = PrimeAffineIntervalCountRequest(
        source=shifted_identity,
        lower=str(10**64),
        upper=str(10**64),
    )
    cancelled_count = compute_interval_count(accepted_cancellation)
    assert (cancelled_count.match_count, cancelled_count.affine_values_examined) == (
        0,
        1,
    )
    cancelled_enumeration = compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(
            source=shifted_identity,
            lower=str(10**64),
            upper=str(10**64),
        )
    )
    assert cancelled_enumeration.matches == ()

    compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(
            source=identity_form, lower="0", upper="32767"
        )
    )
    with pytest.raises(OperationDomainValidationError):
        compute_interval_enumerate(
            PrimeAffineIntervalEnumerateRequest(
                source=identity_form, lower="0", upper="32768"
            )
        )

    cancellation_source = _tuple(_form("shifted_n", 1, -(10**63)))
    with pytest.raises(OperationDomainValidationError):
        compute_interval_enumerate(
            PrimeAffineIntervalEnumerateRequest(
                source=cancellation_source,
                lower=str(10**63),
                upper=str(10**63 + 20_000),
            )
        )

    compute_interval_count(
        PrimeAffineIntervalCountRequest(
            source=identity_form, lower=str(2**64 - 1), upper=str(2**64 - 1)
        )
    )
    with pytest.raises(OperationDomainValidationError):
        compute_interval_count(
            PrimeAffineIntervalCountRequest(
                source=identity_form, lower=str(2**64), upper=str(2**64)
            )
        )

    large_machine_prime = 4_294_967_291
    compute_local_factors(
        PrimeTupleLocalFactorsRequest(
            source=identity_form, primes=(large_machine_prime,)
        )
    )
    machine_prime_wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(
            source=identity_form,
            primes=(large_machine_prime,),
        )
    )
    assert machine_prime_wheel.modulus == str(large_machine_prime)
    assert machine_prime_wheel.valid_count == str(large_machine_prime - 1)

    large_compact_wheel = compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=identity_form, primes=(8_209,))
    )
    assert large_compact_wheel.valid_count == "8208"
    with pytest.raises(OperationDomainValidationError):
        compute_residue_wheel_enumeration(
            PrimeTupleResidueWheelEnumerationRequest(wheel=large_compact_wheel)
        )

    with pytest.raises(OperationDomainValidationError):
        compute_translation(
            PrimeAffineTranslationRequest(
                source=_tuple(_form("large", 1, int("9" * 256))), shift="1"
            )
        )


def test_prime_sets_and_interval_relations_are_schema_visible() -> None:
    factor_schema = PrimeTupleLocalFactorsRequest.model_json_schema()
    assert "strictly increasing" in factor_schema["properties"]["primes"]["description"]
    assert factor_schema["properties"]["primes"]["items"]["maximum"] == ((1 << 53) - 1)
    source_schema = factor_schema["$defs"]["PrimeAffineTuple"]
    assert (
        str(MAX_AFFINE_AGGREGATE_DIGITS)
        in source_schema["properties"]["forms"]["description"]
    )
    form_schema = factor_schema["$defs"]["PrimitiveIntegerAffineForm"]
    assert form_schema["properties"]["coefficient"]["maxLength"] == 257
    assert (
        "at most 256 digits" in form_schema["properties"]["coefficient"]["description"]
    )
    interval_schema = PrimeAffineIntervalCountRequest.model_json_schema()
    assert "at least lower" in interval_schema["properties"]["upper"]["description"]


def test_every_prime_affine_tool_has_an_executable_example() -> None:
    expected_ids = {
        "number_theory.prime_affine_forms.interval_count.compute",
        "number_theory.prime_affine_forms.interval_enumerate.compute",
        "number_theory.prime_affine_forms.interval_residue_profile.compute",
        "number_theory.prime_affine_forms.local_admissibility.compute",
        "number_theory.prime_affine_forms.local_factor.compute",
        "number_theory.prime_affine_forms.local_factors.compute",
        "number_theory.prime_affine_forms.residue_wheel.compute",
        "number_theory.prime_affine_forms.residue_wheel.enumerate.compute",
        "number_theory.prime_affine_forms.translation.compute",
        "number_theory.prime_affine_forms.wheel_membership.compute",
    }
    assert {tool.operation_id for tool in TOOLS} == expected_ids
    assert all(tool.examples for tool in TOOLS)

    for tool in TOOLS:
        for operation_example in tool.examples:
            request = tool.request_type.model_validate(operation_example.input)
            result = tool.run(request)
            tool.result_type.model_validate(result.model_dump(mode="json"))
