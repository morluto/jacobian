"""Exact square-free affine-form local factors and finite Euler products."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from math import isqrt

import pytest
from pydantic import ValidationError
from sympy import primerange

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.squarefree_affine_forms import (
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
    euler_product,
    infinite_product_enclosure,
    interval_count,
    local_admissibility,
    local_factor,
    verify_squarefree_affine_family,
)
from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    LocalAdmissibilityRequest,
    LocalAdmissibilityResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._euler_product import (
    SquarefreeEulerProductRequest,
    SquarefreeEulerProductResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._infinite_product import (
    SquarefreeInfiniteProductRequest,
)
from jacobian.math.number_theory.squarefree_affine_forms._interval_count import (
    IntervalCountRequest,
    IntervalCountResult,
    _interval_sieve_work,
)
from jacobian.math.number_theory.squarefree_affine_forms._kernel import (
    _NoSolutions,
    _SolutionCoset,
    form_solution_profile,
)
from jacobian.math.number_theory.squarefree_affine_forms._local_factor import (
    SquarefreeLocalFactorRequest,
    SquarefreeLocalFactorResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._models import (
    MAX_EULER_PRIMES,
    MAX_INTERVAL_SIEVE_RESIDUES,
    MAX_INTERVAL_SIEVE_VISITS,
    MAX_LOCAL_FACTOR_PRIME,
    _decimal_digits,
    admit_admissibility_cutoff,
    admit_interval_sieve_residues,
)
from jacobian.math.number_theory.squarefree_affine_forms._tools import (
    TOOLS,
    compute_euler_product,
    compute_infinite_product,
    compute_interval_count,
    compute_local_admissibility,
    compute_local_factor,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    MAX_SQUAREFREE_FORMS,
)


def _form(form_id: str, coefficient: int, constant: int) -> SquarefreeAffineForm:
    return SquarefreeAffineForm(
        form_id=form_id, coefficient=coefficient, constant=constant
    )


def _family(*forms: SquarefreeAffineForm) -> SquarefreeAffineFamily:
    return SquarefreeAffineFamily(forms=forms)


SINGLE_N = _family(_form("n", 1, 0))
CONSECUTIVE_PAIR = _family(_form("n", 1, 0), _form("n_plus_1", 1, 1))


def _brute_force_ledger(
    family: SquarefreeAffineFamily, prime: int
) -> dict[int, tuple[str, ...]]:
    modulus = prime * prime
    ledger: dict[int, tuple[str, ...]] = {}
    for residue in range(modulus):
        form_ids = tuple(
            sorted(
                form.form_id
                for form in family.forms
                if (form.coefficient * residue + form.constant) % modulus == 0
            )
        )
        if form_ids:
            ledger[residue] = form_ids
    return ledger


def test_known_answer_single_form_mod_four() -> None:
    result = compute_local_factor(
        SquarefreeLocalFactorRequest(source=SINGLE_N, prime=2)
    )
    assert result.modulus == 4
    assert [(row.residue, row.form_ids) for row in result.bad_residues] == [(0, ("n",))]
    assert (result.bad_count, result.valid_count) == (1, 3)
    assert result.local_factor.as_integer_ratio() == (3, 4)
    assert not result.has_local_obstruction
    assert result.overlap_count == 0
    assert [
        (row.form_id, row.bad_count, row.root, row.stride) for row in result.form_rows
    ] == [("n", 1, 0, 4)]


def test_known_answer_consecutive_pair_mod_four() -> None:
    result = compute_local_factor(
        SquarefreeLocalFactorRequest(source=CONSECUTIVE_PAIR, prime=2)
    )
    assert [(row.residue, row.form_ids) for row in result.bad_residues] == [
        (0, ("n",)),
        (3, ("n_plus_1",)),
    ]
    assert (result.bad_count, result.valid_count) == (2, 2)
    assert result.local_factor.as_integer_ratio() == (1, 2)


def test_known_answer_base_ten_dead_end_family() -> None:
    family = _family(
        _form("n", 1, 0),
        _form("d1", 10, 1),
        _form("d3", 10, 3),
    )
    result = compute_local_factor(SquarefreeLocalFactorRequest(source=family, prime=3))
    expected = _brute_force_ledger(family, 3)
    assert {(row.residue, row.form_ids) for row in result.bad_residues} == set(
        expected.items()
    )
    assert result.bad_count == len(expected) == 3
    assert result.local_factor.as_integer_ratio() == (2, 3)

    product = compute_euler_product(
        SquarefreeEulerProductRequest(source=family, primes=(3,))
    )
    assert product.product.as_integer_ratio() == (2, 3)
    assert not product.has_local_obstruction
    assert product.first_obstructing_prime is None


def test_solution_profiles_are_tagged_and_nonnullable() -> None:
    empty = form_solution_profile(_form("empty", 6, 5), 3)
    assert isinstance(empty, _NoSolutions)

    coset = form_solution_profile(_form("progression", 6, 3), 3)
    assert isinstance(coset, _SolutionCoset)
    assert (coset.count, coset.root, coset.stride) == (3, 1, 3)
    with pytest.raises(ValueError):
        _SolutionCoset(0, 0, 1)
    with pytest.raises(ValueError):
        _SolutionCoset(1, 3, 3)


def test_constant_forms_are_handled_explicitly() -> None:
    full = compute_local_factor(
        SquarefreeLocalFactorRequest(source=_family(_form("c", 0, 4)), prime=2)
    )
    assert full.covers_all_residues
    assert full.bad_residues == ()
    assert (full.bad_count, full.valid_count) == (4, 0)
    assert full.local_factor.as_integer_ratio() == (0, 1)
    assert full.has_local_obstruction

    empty = compute_local_factor(
        SquarefreeLocalFactorRequest(source=_family(_form("c", 0, 5)), prime=2)
    )
    assert not empty.covers_all_residues
    assert (empty.bad_count, empty.valid_count) == (0, 4)
    assert empty.local_factor.as_integer_ratio() == (1, 1)

    # p divides the coefficient but p^2 does not divide the constant.
    no_root = compute_local_factor(
        SquarefreeLocalFactorRequest(source=_family(_form("f", 6, 5)), prime=3)
    )
    assert (no_root.bad_count, no_root.valid_count) == (0, 9)

    # p divides coefficient and constant: exactly p bad residues in one coset.
    progression = compute_local_factor(
        SquarefreeLocalFactorRequest(source=_family(_form("f", 6, 3)), prime=3)
    )
    row = progression.form_rows[0]
    assert (row.bad_count, row.stride) == (3, 3)
    assert row.root is not None
    assert [item.residue for item in progression.bad_residues] == [
        row.root,
        row.root + 3,
        row.root + 6,
    ]
    assert progression.local_factor.as_integer_ratio() == (2, 3)

    mixed = compute_local_factor(
        SquarefreeLocalFactorRequest(
            source=_family(_form("c", 0, 5), _form("n", 1, 0)), prime=2
        )
    )
    assert (mixed.bad_count, mixed.valid_count) == (1, 3)


def test_divisible_coefficient_and_overlap_shapes() -> None:
    invertible = compute_local_factor(
        SquarefreeLocalFactorRequest(source=_family(_form("f", 2, 1)), prime=2)
    )
    assert (invertible.bad_count, invertible.valid_count) == (0, 4)
    assert invertible.local_factor.as_integer_ratio() == (1, 1)

    overlap = compute_local_factor(
        SquarefreeLocalFactorRequest(
            source=_family(_form("n", 1, 0), _form("two_n", 2, 0)), prime=3
        )
    )
    assert [(row.residue, row.form_ids) for row in overlap.bad_residues] == [
        (0, ("n", "two_n"))
    ]
    assert overlap.overlap_count == 1
    assert (overlap.bad_count, overlap.valid_count) == (1, 8)


def test_full_residue_coset_at_admitted_prime_bound() -> None:
    prime = 997
    result = local_factor(_family(_form("constant", 0, prime * prime)), prime)

    assert result.covers_all_residues
    assert result.bad_residues == ()
    assert result.bad_count == prime * prime
    assert result.valid_count == 0


def test_obstructed_family_covers_every_residue() -> None:
    family = _family(
        _form("r0", 1, 0),
        _form("r1", 1, 1),
        _form("r2", 1, 2),
        _form("r3", 1, 3),
    )
    result = compute_local_factor(SquarefreeLocalFactorRequest(source=family, prime=2))
    assert not result.covers_all_residues
    assert (result.bad_count, result.valid_count) == (4, 0)
    assert result.has_local_obstruction
    assert result.local_factor.as_integer_ratio() == (0, 1)
    assert len(result.bad_residues) == 4


def test_large_prime_positivity_bound() -> None:
    family = _family(
        _form("a", 6, 1),
        _form("b", 10, 3),
        _form("c", 15, 7),
    )
    for prime in (17, 19, 23, 97):
        result = compute_local_factor(
            SquarefreeLocalFactorRequest(source=family, prime=prime)
        )
        # p divides no coefficient, so each form excludes exactly one residue.
        assert all(row.bad_count == 1 for row in result.form_rows)
        assert result.valid_count >= prime * prime - family.form_count
        assert not result.has_local_obstruction


@pytest.mark.parametrize("prime", [2, 3, 5, 7, 11, 13, 29, 31])
def test_ledger_matches_brute_force_enumeration(prime: int) -> None:
    family = _family(
        _form("f0", 1, 0),
        _form("f1", 3, -2),
        _form("f2", prime, 5),
        _form("f3", 0, 7),
        _form("f4", -4, 11),
    )
    result = compute_local_factor(
        SquarefreeLocalFactorRequest(source=family, prime=prime)
    )
    expected = _brute_force_ledger(family, prime)
    assert {(row.residue, row.form_ids) for row in result.bad_residues} == set(
        expected.items()
    )
    assert result.bad_count == len(expected)
    assert result.valid_count == prime * prime - len(expected)
    assert (
        result.local_factor.as_integer_ratio()
        == Fraction(prime * prime - len(expected), prime * prime).as_integer_ratio()
    )
    # Defining partition invariant: valid + bad covers every residue once.
    assert result.bad_count + result.valid_count == prime * prime
    for row in result.bad_residues:
        for form in family.forms:
            if form.form_id in row.form_ids:
                assert (form.coefficient * row.residue + form.constant) % (
                    prime * prime
                ) == 0


def test_random_families_satisfy_defining_invariants() -> None:
    generator = random.Random(1705)
    for _ in range(40):
        forms = tuple(
            _form(
                f"f{index}",
                generator.randint(-30, 30),
                generator.randint(-30, 30),
            )
            for index in range(generator.randint(1, 4))
        )
        pairs = {(form.coefficient, form.constant) for form in forms}
        if len(pairs) != len(forms) or any(a == 0 and b == 0 for a, b in pairs):
            continue
        family = _family(*forms)
        prime = generator.choice([2, 3, 5, 7, 11])
        result = compute_local_factor(
            SquarefreeLocalFactorRequest(source=family, prime=prime)
        )
        expected = _brute_force_ledger(family, prime)
        assert {(row.residue, row.form_ids) for row in result.bad_residues} == set(
            expected.items()
        )
        assert result.valid_count == prime * prime - len(expected)


def test_euler_product_is_exact_over_its_prime_set() -> None:
    result = compute_euler_product(
        SquarefreeEulerProductRequest(source=CONSECUTIVE_PAIR, primes=(2, 3, 5))
    )
    expected = Fraction(1, 1)
    for row in result.rows:
        assert row.modulus == row.prime**2
        assert row.bad_count + row.valid_count == row.modulus
        expected *= Fraction(*row.local_factor.as_integer_ratio())
    assert result.product.as_integer_ratio() == expected.as_integer_ratio()
    assert (
        result.product.as_integer_ratio()
        == (Fraction(1, 2) * Fraction(7, 9) * Fraction(23, 25)).as_integer_ratio()
    )
    assert not result.has_local_obstruction

    single = compute_euler_product(
        SquarefreeEulerProductRequest(source=SINGLE_N, primes=(2,))
    )
    assert single.product.as_integer_ratio() == (3, 4)
    assert len(single.rows) == 1

    # A union of bad cosets covers every residue modulo 4 without any single
    # form vanishing identically.
    covered = compute_euler_product(
        SquarefreeEulerProductRequest(
            source=_family(
                _form("r0", 1, 0),
                _form("r1", 1, 1),
                _form("r2", 1, 2),
                _form("r3", 1, 3),
            ),
            primes=(2, 3),
        )
    )
    assert covered.has_local_obstruction
    assert covered.first_obstructing_prime == 2
    assert covered.product.as_integer_ratio() == (0, 1)
    assert covered.rows[0].valid_count == 0
    assert covered.rows[1].valid_count > 0

    identically = compute_euler_product(
        SquarefreeEulerProductRequest(source=_family(_form("c", 0, 4)), primes=(2, 3))
    )
    assert identically.has_local_obstruction
    assert identically.first_obstructing_prime == 2
    assert identically.rows[1].valid_count == 9


def test_infinite_product_enclosure_uses_checked_rational_tail_bound() -> None:
    result = compute_infinite_product(
        SquarefreeInfiniteProductRequest(source=CONSECUTIVE_PAIR, prime_cutoff=100)
    )
    lower = result.enclosure.lower.as_fraction()
    upper = result.enclosure.upper.as_fraction()
    assert result.source == CONSECUTIVE_PAIR
    assert result.prime_cutoff == 100

    # For n and n+1, the two bad residues are distinct modulo p^2, so every
    # local factor is independently 1 - 2/p^2. A longer finite product must
    # remain inside the returned enclosure by the summable-factor tail bound.
    longer_prefix = Fraction(1, 1)
    for prime in primerange(2, 2_001):
        longer_prefix *= Fraction(prime * prime - 2, prime * prime)
    assert lower <= longer_prefix <= upper

    cutoff_error = Fraction(2, 100)
    prefix_at_cutoff = Fraction(1, 1)
    for prime in primerange(2, 101):
        prefix_at_cutoff *= Fraction(prime * prime - 2, prime * prime)
    assert lower == prefix_at_cutoff * (1 - cutoff_error)
    assert upper == prefix_at_cutoff

    obstructed = _family(*(_form(f"r{i}", 1, i) for i in range(4)))
    zero = infinite_product_enclosure(obstructed, 2)
    assert zero.enclosure.lower.as_fraction() == 0
    assert zero.enclosure.upper.as_fraction() == 0

    too_small = _family(_form("large_slope", 5, 1))
    with pytest.raises(OperationDomainValidationError) as error:
        infinite_product_enclosure(too_small, 4)
    assert error.value.errors()[0]["type"].endswith("infinite_product_cutoff_too_small")


def test_metamorphic_row_order_relabeling_and_translation() -> None:
    shuffled = SquarefreeAffineFamily(
        forms=(CONSECUTIVE_PAIR.forms[1], CONSECUTIVE_PAIR.forms[0])
    )
    assert shuffled == CONSECUTIVE_PAIR
    assert compute_local_factor(
        SquarefreeLocalFactorRequest(source=shuffled, prime=2)
    ) == compute_local_factor(
        SquarefreeLocalFactorRequest(source=CONSECUTIVE_PAIR, prime=2)
    )

    relabelled = _family(_form("m", 1, 0), _form("m_plus_1", 1, 1))
    relabelled_result = compute_local_factor(
        SquarefreeLocalFactorRequest(source=relabelled, prime=2)
    )
    assert (relabelled_result.bad_count, relabelled_result.valid_count) == (2, 2)
    assert relabelled_result.local_factor.as_integer_ratio() == (1, 2)

    # n -> n+5 permutes residues and preserves local-factor counts.
    translated = _family(_form("n", 1, 5), _form("n_plus_1", 1, 6))
    translated_result = compute_local_factor(
        SquarefreeLocalFactorRequest(source=translated, prime=3)
    )
    original_result = compute_local_factor(
        SquarefreeLocalFactorRequest(source=CONSECUTIVE_PAIR, prime=3)
    )
    assert translated_result.bad_count == original_result.bad_count
    assert translated_result.valid_count == original_result.valid_count
    assert translated_result.local_factor == original_result.local_factor


def test_adversarial_inputs_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError) as composite:
        compute_local_factor(SquarefreeLocalFactorRequest(source=SINGLE_N, prime=15))
    assert composite.value.errors()[0]["type"] == (
        "number_theory.squarefree_affine.prime_required"
    )
    with pytest.raises(ValidationError):
        SquarefreeLocalFactorRequest(source=SINGLE_N, prime=1)
    with pytest.raises(OperationDomainValidationError) as unordered:
        compute_euler_product(
            SquarefreeEulerProductRequest(source=SINGLE_N, primes=(3, 2))
        )
    assert unordered.value.errors()[0]["type"] == (
        "number_theory.squarefree_affine.prime_order"
    )
    with pytest.raises(OperationDomainValidationError):
        compute_euler_product(
            SquarefreeEulerProductRequest(source=SINGLE_N, primes=(2, 2))
        )
    with pytest.raises(ValidationError):
        SquarefreeEulerProductRequest(source=SINGLE_N, primes=())

    with pytest.raises(ValidationError):
        _family(_form("same", 1, 0), _form("same", 1, 2))
    with pytest.raises(ValidationError):
        _family(_form("first", 1, 0), _form("second", 1, 0))
    with pytest.raises(ValidationError):
        _form("zero", 0, 0)
    with pytest.raises(ValidationError):
        _form("wide", 10**8, 0)
    with pytest.raises(ValidationError):
        _family(
            *(_form(f"f{index}", 1, index) for index in range(MAX_SQUAREFREE_FORMS + 1))
        )


def test_forged_results_fail_verification() -> None:
    genuine = compute_local_factor(
        SquarefreeLocalFactorRequest(source=CONSECUTIVE_PAIR, prime=2)
    )
    payload = json.loads(genuine.model_dump_json())
    payload["bad_residues"][0]["residue"] = "1"
    with pytest.raises(ValidationError):
        SquarefreeLocalFactorResult.model_validate_json(json.dumps(payload))

    payload = json.loads(genuine.model_dump_json())
    payload["bad_residues"] = payload["bad_residues"][:1]
    payload["bad_count"] = "1"
    payload["valid_count"] = "3"
    payload["local_factor"] = {"num": "3", "den": "4"}
    with pytest.raises(ValidationError):
        SquarefreeLocalFactorResult.model_validate_json(json.dumps(payload))

    payload = json.loads(genuine.model_dump_json())
    payload["form_rows"][0]["root"] = "2"
    with pytest.raises(ValidationError):
        SquarefreeLocalFactorResult.model_validate_json(json.dumps(payload))

    payload = json.loads(genuine.model_dump_json())
    payload["has_local_obstruction"] = True
    with pytest.raises(ValidationError):
        SquarefreeLocalFactorResult.model_validate_json(json.dumps(payload))

    product = compute_euler_product(
        SquarefreeEulerProductRequest(source=CONSECUTIVE_PAIR, primes=(2, 3))
    )
    payload = json.loads(product.model_dump_json())
    payload["product"] = {"num": "1", "den": "1"}
    with pytest.raises(ValidationError):
        SquarefreeEulerProductResult.model_validate_json(json.dumps(payload))

    payload = json.loads(product.model_dump_json())
    payload["rows"][0]["valid_count"] = "3"
    with pytest.raises(ValidationError):
        SquarefreeEulerProductResult.model_validate_json(json.dumps(payload))


def test_envelope_accepts_and_rejects_at_boundaries() -> None:
    largest_prime = max(int(p) for p in primerange(2, MAX_LOCAL_FACTOR_PRIME + 1))
    assert largest_prime == 997
    accepted = compute_local_factor(
        SquarefreeLocalFactorRequest(source=SINGLE_N, prime=largest_prime)
    )
    assert accepted.bad_count == 1
    assert accepted.valid_count == largest_prime**2 - 1

    with pytest.raises(OperationResourceAdmissionError) as too_large:
        compute_local_factor(SquarefreeLocalFactorRequest(source=SINGLE_N, prime=1_009))
    assert too_large.value.errors()[0]["type"] == (
        "number_theory.squarefree_affine.prime_budget"
    )

    full_family = _family(
        *(_form(f"f{index}", 1, index) for index in range(MAX_SQUAREFREE_FORMS))
    )
    compute_local_factor(
        SquarefreeLocalFactorRequest(source=full_family, prime=largest_prime)
    )

    small_primes = tuple(int(p) for p in list(primerange(2, 200))[:MAX_EULER_PRIMES])
    assert len(small_primes) == MAX_EULER_PRIMES
    compute_euler_product(
        SquarefreeEulerProductRequest(source=SINGLE_N, primes=small_primes)
    )
    with pytest.raises(ValidationError):
        SquarefreeEulerProductRequest(
            source=SINGLE_N,
            primes=tuple(
                int(p) for p in list(primerange(2, 200))[: MAX_EULER_PRIMES + 1]
            ),
        )
    oversized_batch = SquarefreeEulerProductRequest.model_construct(
        source=SINGLE_N,
        primes=tuple(int(p) for p in list(primerange(2, 200))[: MAX_EULER_PRIMES + 1]),
    )
    with pytest.raises(OperationResourceAdmissionError) as too_many:
        compute_euler_product(oversized_batch)
    assert too_many.value.errors()[0]["type"] == (
        "number_theory.squarefree_affine.prime_batch_budget"
    )

    with pytest.raises(OperationResourceAdmissionError) as too_much_work:
        compute_euler_product(
            SquarefreeEulerProductRequest(
                source=full_family,
                primes=tuple(
                    int(p)
                    for p in list(primerange(900, MAX_LOCAL_FACTOR_PRIME + 1))[
                        :MAX_EULER_PRIMES
                    ]
                ),
            )
        )
    assert too_much_work.value.errors()[0]["type"] == (
        "number_theory.squarefree_affine.work_budget"
    )


def test_native_and_catalog_paths_agree_and_round_trip() -> None:
    native = local_factor(CONSECUTIVE_PAIR, 3)
    catalog = compute_local_factor(
        SquarefreeLocalFactorRequest(source=CONSECUTIVE_PAIR, prime=3)
    )
    assert native == catalog
    restored = SquarefreeLocalFactorResult.model_validate_json(native.model_dump_json())
    assert restored == native

    native_product = euler_product(CONSECUTIVE_PAIR, (2, 3))
    catalog_product = compute_euler_product(
        SquarefreeEulerProductRequest(source=CONSECUTIVE_PAIR, primes=(2, 3))
    )
    assert native_product == catalog_product
    restored_product = SquarefreeEulerProductResult.model_validate_json(
        native_product.model_dump_json()
    )
    assert restored_product == native_product

    covered = local_factor(_family(_form("c", 0, 9)), 3)
    assert covered.covers_all_residues
    assert (
        SquarefreeLocalFactorResult.model_validate_json(covered.model_dump_json())
        == covered
    )


def test_family_claim_verification() -> None:
    assert verify_squarefree_affine_family(CONSECUTIVE_PAIR)
    assert (
        verify_squarefree_affine_family(
            SquarefreeAffineFamily.model_construct(
                forms=(_form("dup", 1, 0), _form("dup2", 1, 0))
            )
        )
        is False
    )
    assert (
        verify_squarefree_affine_family(
            SquarefreeAffineFamily.model_construct(forms=())
        )
        is False
    )
    assert (
        verify_squarefree_affine_family(
            SquarefreeAffineFamily.model_construct(
                forms=(
                    SquarefreeAffineForm.model_construct(
                        form_id="wide", coefficient=10**9, constant=0
                    ),
                )
            )
        )
        is False
    )

    # A malformed constructed family is still rejected by operation admission.
    malformed = SquarefreeAffineFamily.model_construct(
        forms=(
            SquarefreeAffineForm.model_construct(
                form_id="zero", coefficient=0, constant=0
            ),
        )
    )
    with pytest.raises(OperationDomainValidationError):
        local_factor(malformed, 2)


def test_constructed_malformed_sources_reject_stably() -> None:
    malformed_sources = (
        SquarefreeAffineFamily.model_construct(
            forms=(
                SquarefreeAffineForm.model_construct(
                    form_id="n", coefficient="1", constant=0
                ),
            )
        ),
        SquarefreeAffineFamily.model_construct(
            forms=(
                SquarefreeAffineForm.model_construct(
                    form_id="n", coefficient=1, constant=True
                ),
            )
        ),
        SquarefreeAffineFamily.model_construct(
            forms=(
                SquarefreeAffineForm.model_construct(
                    form_id=7, coefficient=1, constant=0
                ),
            )
        ),
        SquarefreeAffineFamily.model_construct(forms=(object(),)),
        SquarefreeAffineFamily.model_construct(forms=object()),
        SquarefreeAffineFamily.model_construct(),
        SquarefreeAffineFamily.model_construct(
            forms=(SquarefreeAffineForm.model_construct(coefficient=1, constant=0),)
        ),
    )
    boundary_codes = {
        "number_theory.squarefree_affine.family_source",
        "number_theory.squarefree_affine.family_form_source",
    }
    for malformed in malformed_sources:
        for call in (
            lambda source: local_factor(source, 2),
            lambda source: euler_product(source, (2,)),
            lambda source: infinite_product_enclosure(source, 100),
            lambda source: local_admissibility(source),
            lambda source: interval_count(source, 1, 10),
        ):
            with pytest.raises(OperationDomainValidationError) as exc_info:
                call(malformed)
            assert exc_info.value.errors()[0]["type"] in boundary_codes

    def is_squarefree(value: int) -> bool:
        value = abs(value)
        return all(
            value % (prime * prime) != 0 for prime in primerange(2, isqrt(value) + 1)
        )

    twin_pair = _family(_form("n", 1, 0), _form("n_plus_2", 1, 2))
    result = interval_count(twin_pair, 1, 30)
    assert result.count == sum(
        1 for n in range(1, 31) if is_squarefree(n) and is_squarefree(n + 2)
    )
    assert result.count > 0
    assert local_admissibility(twin_pair).status == "LOCALLY_ADMISSIBLE"
    enclosure = infinite_product_enclosure(twin_pair, 100).enclosure
    assert 0 < enclosure.lower.as_fraction() <= enclosure.upper.as_fraction() <= 1


def test_tool_declarations_are_published() -> None:
    operation_ids = {tool.operation_id for tool in TOOLS}
    assert operation_ids == {
        "number_theory.squarefree_affine_forms.local_factor.compute",
        "number_theory.squarefree_affine_forms.euler_product.compute",
        "number_theory.squarefree_affine_forms.infinite_product.enclose",
        "number_theory.squarefree_affine_forms.local_admissibility.decide",
        "number_theory.squarefree_affine_forms.interval_count.compute",
    }
    for tool in TOOLS:
        assert 1 <= len(tool.discovery_terms) <= 8
        assert tool.examples
        for example in tool.examples:
            request = tool.request_type.model_validate_json(json.dumps(example.input))
            result = tool.run(request)
            assert result is not None


def _twin_pair() -> SquarefreeAffineFamily:
    return _family(_form("n", 1, 0), _form("n_plus_2", 1, 2))


def _brute_force_admissible(family: SquarefreeAffineFamily, bound: int) -> bool:
    """Independent obstruction search over small primes for the tests."""

    from sympy import isprime

    for prime in range(2, bound + 1):
        if not isprime(prime):
            continue
        modulus = prime * prime
        if all(
            any(
                (form.coefficient * residue + form.constant) % modulus == 0
                for form in family.forms
            )
            for residue in range(modulus)
        ):
            return False
    return True


def test_twin_pair_is_locally_admissible() -> None:
    family = _twin_pair()
    result = local_admissibility(family)

    assert result.status == "LOCALLY_ADMISSIBLE"
    assert result.cutoff == 1
    assert result.rows == ()
    assert result.obstruction is None


def test_single_form_is_locally_admissible() -> None:
    result = local_admissibility(SINGLE_N)

    assert result.status == "LOCALLY_ADMISSIBLE"
    assert result.cutoff == 1


def test_constant_four_is_obstructed_at_two() -> None:
    family = _family(_form("c", 0, 4))
    result = local_admissibility(family)

    assert result.status == "LOCALLY_OBSTRUCTED"
    assert result.obstruction is not None
    assert result.obstruction.prime == 2
    assert result.obstruction.valid_count == 0


def test_admissibility_cutoff_and_exact_rows() -> None:
    family = _family(_form("a", 6, 1), _form("b", 10, 3), _form("c", 15, 7))
    result = local_admissibility(family)

    # Cutoff: A = 15, M = max(3 forms, no constant forms) = 3, B = 15.
    assert (result.cutoff, result.max_abs_coefficient, result.large_prime_bound) == (
        15,
        15,
        3,
    )
    assert [row.prime for row in result.rows] == list(primerange(2, 16))
    assert result.status == "LOCALLY_ADMISSIBLE"
    assert all(row.valid_count > 0 for row in result.rows)
    assert _brute_force_admissible(family, 15)


def test_admissibility_matches_brute_force() -> None:
    family = _family(_form("n", 2, 1), _form("m", 3, 2))
    result = local_admissibility(family)

    assert result.status == "LOCALLY_ADMISSIBLE"
    assert _brute_force_admissible(family, result.cutoff)


def test_admissibility_native_and_catalog_paths_agree() -> None:
    family = _twin_pair()
    request = LocalAdmissibilityRequest(source=family)

    assert compute_local_admissibility(request) == local_admissibility(family)


def test_admissibility_round_trip_and_forgery() -> None:
    family = _family(_form("c", 0, 4))
    result = local_admissibility(family)
    restored = LocalAdmissibilityResult.model_validate_json(result.model_dump_json())

    assert restored == result
    forged = json.loads(restored.model_dump_json())
    forged["status"] = "LOCALLY_ADMISSIBLE"
    forged["obstruction"] = None
    with pytest.raises(ValidationError):
        LocalAdmissibilityResult.model_validate_json(json.dumps(forged))
    forged_rows = json.loads(restored.model_dump_json())
    forged_rows["rows"][0]["bad_count"] = 1
    forged_rows["rows"][0]["valid_count"] = 3
    with pytest.raises(ValidationError):
        LocalAdmissibilityResult.model_validate_json(json.dumps(forged_rows))


def test_cutoff_type_errors_are_domain_rejections() -> None:
    for cutoff in (True, "100"):
        with pytest.raises(OperationDomainValidationError):
            admit_admissibility_cutoff(cutoff)  # type: ignore[arg-type]
        with pytest.raises(OperationDomainValidationError):
            infinite_product_enclosure(SINGLE_N, cutoff)  # type: ignore[arg-type]


def test_decimal_digit_admission_short_circuits_giant_integers() -> None:
    giant = 1 << 1_000_000
    assert _decimal_digits(giant) > 8


def test_admissibility_cutoff_budget_is_a_resource_boundary() -> None:
    family = _family(_form("big", 10**7, 1))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        local_admissibility(family)
    assert "cutoff" in exc_info.value.errors()[0]["type"]


def test_interval_count_one_to_twenty() -> None:
    result = interval_count(SINGLE_N, 1, 20, True)

    assert result.count == 13
    assert result.matching == (1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19)
    assert [(row.n, row.form_id, row.prime) for row in result.obstructions] == [
        (4, "n", 2),
        (8, "n", 2),
        (9, "n", 3),
        (12, "n", 2),
        (16, "n", 2),
        (18, "n", 3),
        (20, "n", 2),
    ]


def test_interval_count_without_ledger() -> None:
    result = interval_count(SINGLE_N, 1, 20, False)

    assert result.count == 13
    assert result.matching == ()
    assert result.obstructions == ()


def test_interval_count_matches_brute_force() -> None:
    import math

    family = _family(_form("a", 2, 1), _form("b", 3, 1))

    def is_squarefree(value: int) -> bool:
        root = math.isqrt(abs(value))
        return all(value % (prime * prime) for prime in range(2, root + 1))

    expected = sum(
        1
        for n in range(-10, 31)
        if all(
            is_squarefree(form.coefficient * n + form.constant) for form in family.forms
        )
    )
    result = interval_count(family, -10, 30, True)

    assert result.count == expected
    for point in result.matching:
        assert all(
            is_squarefree(form.coefficient * point + form.constant)
            for form in family.forms
        )
    for row in result.obstructions:
        form = next(f for f in family.forms if f.form_id == row.form_id)
        value = form.coefficient * row.n + form.constant
        assert value % (row.prime * row.prime) == 0


def test_interval_count_square_coefficient_does_not_materialize_residues() -> None:
    # 9973^2 | coefficient: the congruence holds for every residue mod 9973^2.
    # The sieve must mark the interval directly instead of building ~10^8
    # residues; the least prime is 2 at n=0 and 9973 at n=1.
    family = _family(_form("sq", 9973**2, 0))

    result = interval_count(family, 0, 1, True)

    assert result.count == 0
    assert result.matching == ()
    assert [(row.n, row.prime) for row in result.obstructions] == [(0, 2), (1, 9973)]


def test_interval_sieve_residue_budget_is_a_resource_boundary() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        admit_interval_sieve_residues(MAX_INTERVAL_SIEVE_RESIDUES + 1)


def test_interval_sieve_charges_point_visits_before_congruence_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each form coefficient contains a distinct prime near 200,000 and the
    # factors 2, 3, 5, 7.  Over this 20,000-point interval, class enumeration
    # fits its 4,000,000 bound while the bounded progression-visit estimate
    # exceeds its separate work envelope.
    high_primes = tuple(primerange(200_000, 200_200))[:MAX_SQUAREFREE_FORMS]
    family = _family(*(_form(f"p{prime}", 210 * prime, 0) for prime in high_primes))
    maximum = max(abs(form.coefficient * 19_999) for form in family.forms)
    primes = tuple(primerange(2, isqrt(maximum) + 1))
    class_work, visit_work = _interval_sieve_work(family, primes, 20_000)
    assert class_work < MAX_INTERVAL_SIEVE_RESIDUES
    assert visit_work > MAX_INTERVAL_SIEVE_VISITS

    def sieve_must_not_expand(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("congruence classes expanded before work admission")

    monkeypatch.setattr(
        "jacobian.math.number_theory.squarefree_affine_forms._interval_count._congruence_classes",
        sieve_must_not_expand,
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        interval_count(family, 0, 19_999, False)
    assert "interval_sieve_visit_budget" in exc_info.value.errors()[0]["type"]


def test_interval_sieve_charges_full_residue_branch_once() -> None:
    family = _family(_form("multiple", 4, 0))
    class_work, visit_work = _interval_sieve_work(family, (2,), 20_000)
    # When a coefficient and constant vanish modulo p^2, the kernel marks the
    # interval directly instead of enumerating p^2 residue classes.
    assert (class_work, visit_work) == (20_000, 20_000)


def test_interval_sieve_visit_envelope_covers_brute_force_congruence_points() -> None:
    family = _family(_form("even", 2, 0))
    class_work, visit_work = _interval_sieve_work(family, (2,), 20)
    # Independent oracle: 4 divides 2*n precisely at the ten even integers
    # in [0, 19].  The preflight safely allows for either class meeting the
    # interval's maximum per-progression count.
    actual_hits = sum((2 * n) % 4 == 0 for n in range(20))
    assert class_work == 2
    assert actual_hits == 10
    assert visit_work >= actual_hits


def test_interval_count_zero_form_is_obstructed() -> None:
    family = SINGLE_N
    # n = 0 makes the single form vanish; it is rejected with prime 2.
    result = interval_count(family, 0, 0, True)

    assert result.count == 0
    assert result.matching == ()
    (obstruction,) = result.obstructions
    assert (obstruction.n, obstruction.prime) == (0, 2)


def test_interval_count_interior_zero_is_obstructed() -> None:
    # A zero value is divisible by every square.  When it is an interior point
    # of a short interval the governing magnitude (taken at the endpoints) can
    # be below 4, so the least prime must still be sieved.
    result = interval_count(SINGLE_N, 0, 3, True)

    assert result.count == 3
    assert result.matching == (1, 2, 3)
    assert [(row.n, row.prime) for row in result.obstructions] == [(0, 2)]

    shifted = _family(_form("n_minus_one", 1, -1))
    shifted_result = interval_count(shifted, 0, 4, True)

    assert shifted_result.count == 4
    assert shifted_result.matching == (0, 2, 3, 4)
    assert [(row.n, row.prime) for row in shifted_result.obstructions] == [(1, 2)]


def test_interval_count_native_and_catalog_paths_agree() -> None:
    request = IntervalCountRequest(
        source=SINGLE_N, lower=1, upper=20, include_ledger=True
    )

    assert compute_interval_count(request) == interval_count(SINGLE_N, 1, 20, True)


def test_interval_count_round_trip_and_forgery() -> None:
    result = interval_count(SINGLE_N, 1, 20, True)
    restored = IntervalCountResult.model_validate_json(result.model_dump_json())

    assert restored == result
    forged = json.loads(restored.model_dump_json())
    forged["count"] = 14
    with pytest.raises(ValidationError):
        IntervalCountResult.model_validate_json(json.dumps(forged))
    forged_moved = json.loads(restored.model_dump_json())
    # Move n = 4 from rejected to accepted with a consistent count. The
    # decoded object preserves the caller's claim; a consumer that relies on
    # membership must check the relevant square-divisibility relation.
    forged_moved["matching"] = sorted([*forged_moved["matching"], 4])
    forged_moved["obstructions"] = [
        row for row in forged_moved["obstructions"] if row["n"] != 4
    ]
    forged_moved["count"] = len(forged_moved["matching"])
    forged_claim = IntervalCountResult.model_validate_json(json.dumps(forged_moved))
    assert 4 in forged_claim.matching
    assert any(
        (form.coefficient * 4 + form.constant) % 4 == 0
        for form in forged_claim.source.forms
    )


def test_interval_reversed_bounds_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError):
        interval_count(SINGLE_N, 20, 1, False)
    with pytest.raises(ValidationError):
        IntervalCountRequest(source=SINGLE_N, lower=20, upper=1)


def test_interval_length_budget_is_a_resource_boundary() -> None:
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        interval_count(SINGLE_N, 0, 20_000, False)
    assert "interval_length" in exc_info.value.errors()[0]["type"]


def test_interval_value_budget_is_a_resource_boundary() -> None:
    family = _family(_form("big", 99_999_999, 0))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        interval_count(family, 0, 19999, False)
    assert "interval_value" in exc_info.value.errors()[0]["type"]
