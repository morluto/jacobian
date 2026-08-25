"""Exact asymmetric local-lemma numerical-witness contracts."""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from math import prod

import pytest
from pydantic import ValidationError

from jacobian.math.probability._local_lemma import (
    ASYMMETRIC_LOCAL_LEMMA_OPERATION,
    AsymmetricLocalLemmaWitnessCheckResult,
    AsymmetricLocalLemmaWitnessRequest,
    compute_asymmetric_local_lemma_witness_check,
)
from jacobian.math.probability.local_lemma import (
    MAX_LOCAL_LEMMA_EVENTS,
    MAX_LOCAL_LEMMA_INCIDENCES,
    MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS,
    MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS,
    AsymmetricLocalLemmaWitness,
    check_asymmetric_local_lemma_witness,
)


def _rational(value: Fraction | int) -> dict[str, str]:
    fraction = Fraction(value)
    return {"num": str(fraction.numerator), "den": str(fraction.denominator)}


def _payload(
    labels: tuple[str, ...],
    probability_bounds: tuple[Fraction, ...],
    witnesses: tuple[Fraction, ...],
    neighborhoods: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    return {
        "event_labels": list(labels),
        "probability_upper_bounds": [_rational(value) for value in probability_bounds],
        "witness_parameters": [_rational(value) for value in witnesses],
        "neighborhoods": [list(neighbors) for neighbors in neighborhoods],
    }


def _compute(payload: dict[str, object]) -> AsymmetricLocalLemmaWitnessCheckResult:
    request = AsymmetricLocalLemmaWitnessRequest.model_validate(payload)
    return compute_asymmetric_local_lemma_witness_check(request)


def test_source_backed_tangent_collision_numerics_reconstruct_exactly() -> None:
    # TangentLLLNumerics.lean at the issue's pinned source revision uses
    # x_E = 2*P(E) and neighborhood probability mass <= 1/4.  Three events
    # with P(E)=1/8 and the other two as neighbors attain that mass bound.
    payload = _payload(
        ("E_ab", "E_ac", "E_bc"),
        (Fraction(1, 8),) * 3,
        (Fraction(1, 4),) * 3,
        ((1, 2), (0, 2), (0, 1)),
    )

    result = _compute(payload)
    native = result.as_native()

    assert native.valid is True
    assert native.failed_event_indices == ()
    assert native.witness_product == Fraction(27, 64)
    for row in native.inequalities:
        assert row.neighborhood_product == Fraction(9, 16)
        assert row.right_hand_side == Fraction(9, 64)
        assert row.slack == Fraction(1, 64)
        assert row.inequality_holds is True

    # Independent Fraction replay of the defining relation from retained source.
    source = native.source
    for row in native.inequalities:
        expected_product = prod(
            (
                1 - source.witness_parameters[index]
                for index in source.neighborhoods[row.event_index]
            ),
            start=Fraction(1),
        )
        expected_rhs = source.witness_parameters[row.event_index] * expected_product
        assert row.neighborhood_product == expected_product
        assert row.right_hand_side == expected_rhs
        assert row.slack == (
            expected_rhs - source.probability_upper_bounds[row.event_index]
        )


def test_empty_family_and_isolated_equality_are_exact() -> None:
    empty = _compute(_payload((), (), (), ())).as_native()
    assert empty.valid is True
    assert empty.inequalities == ()
    assert empty.failed_event_indices == ()
    assert empty.witness_product == 1

    isolated = _compute(
        _payload(
            ("A",),
            (Fraction(1, 3),),
            (Fraction(1, 3),),
            ((),),
        )
    ).as_native()
    assert isolated.valid is True
    assert isolated.inequalities[0].neighborhood_product == 1
    assert isolated.inequalities[0].right_hand_side == Fraction(1, 3)
    assert isolated.inequalities[0].slack == 0


def test_one_unit_rational_mutation_makes_the_boundary_fail() -> None:
    equal = _compute(
        _payload(
            ("A",),
            (Fraction(3, 8),),
            (Fraction(3, 8),),
            ((),),
        )
    ).as_native()
    failed = _compute(
        _payload(
            ("A",),
            (Fraction(4, 8),),
            (Fraction(3, 8),),
            ((),),
        )
    ).as_native()

    assert equal.valid is True
    assert equal.inequalities[0].slack == 0
    assert failed.valid is False
    assert failed.failed_event_indices == (0,)
    assert failed.inequalities[0].slack == Fraction(-1, 8)


def test_directed_neighborhood_reversal_changes_the_decision() -> None:
    forward = _compute(
        _payload(
            ("A", "B"),
            (Fraction(1, 5), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(1, 2)),
            ((1,), ()),
        )
    ).as_native()
    reversed_relation = _compute(
        _payload(
            ("A", "B"),
            (Fraction(1, 5), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(1, 2)),
            ((), (0,)),
        )
    ).as_native()

    assert forward.valid is True
    assert reversed_relation.valid is False
    assert reversed_relation.failed_event_indices == (1,)


def test_listed_self_neighbor_is_included_once() -> None:
    result = _compute(
        _payload(
            ("A",),
            (Fraction(1, 4),),
            (Fraction(1, 2),),
            ((0,),),
        )
    ).as_native()

    assert result.valid is True
    assert result.inequalities[0].neighborhood_product == Fraction(1, 2)
    assert result.inequalities[0].right_hand_side == Fraction(1, 4)


@pytest.mark.parametrize(
    "neighborhoods",
    (
        ((0, 0), ()),
        ((1, 0), ()),
        ((2,), ()),
    ),
    ids=("repeated", "not-increasing", "outside-axis"),
)
def test_neighborhoods_must_be_canonical_axis_subsets(
    neighborhoods: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(
            _payload(
                ("A", "B"),
                (Fraction(), Fraction()),
                (Fraction(), Fraction()),
                neighborhoods,
            )
        )


@pytest.mark.parametrize(
    ("probability", "witness", "message"),
    (
        (Fraction(-1, 10), Fraction(1, 2), "probability upper bounds"),
        (Fraction(11, 10), Fraction(1, 2), "probability upper bounds"),
        (Fraction(), Fraction(-1, 10), "witness parameters"),
        (Fraction(), Fraction(1), "witness parameters"),
    ),
    ids=("negative-p", "p-over-one", "negative-x", "x-equals-one"),
)
def test_probability_and_witness_domains_are_admitted_before_multiplication(
    probability: Fraction,
    witness: Fraction,
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(
            _payload(("A",), (probability,), (witness,), ((),))
        )


def test_zero_witness_has_the_exact_expected_boundary_behavior() -> None:
    valid = _compute(_payload(("A",), (Fraction(),), (Fraction(),), ((),))).as_native()
    invalid = _compute(
        _payload(("A",), (Fraction(1, 10),), (Fraction(),), ((),))
    ).as_native()

    assert valid.valid is True
    assert valid.inequalities[0].right_hand_side == 0
    assert invalid.valid is False
    assert invalid.inequalities[0].slack == Fraction(-1, 10)


def test_axis_aligned_fields_and_labels_are_canonical() -> None:
    base = _payload(
        ("A", "B"),
        (Fraction(), Fraction()),
        (Fraction(), Fraction()),
        ((), ()),
    )
    misaligned = deepcopy(base)
    misaligned["witness_parameters"] = [_rational(Fraction())]
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(misaligned)

    duplicated = deepcopy(base)
    duplicated["event_labels"] = ["A", "A"]
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(duplicated)

    non_nfc = _payload(
        ("e\N{COMBINING ACUTE ACCENT}",),
        (Fraction(),),
        (Fraction(),),
        ((),),
    )
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(non_nfc)


def test_result_rejects_independent_source_and_conclusion_forgeries() -> None:
    result = _compute(
        _payload(
            ("A", "B"),
            (Fraction(1, 5), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(1, 2)),
            ((1,), ()),
        )
    )
    serialized = result.model_dump(mode="json")

    source_forgery = deepcopy(serialized)
    source_forgery["source"]["probability_upper_bounds"][0] = _rational(Fraction(1, 4))
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessCheckResult.model_validate(source_forgery)

    conclusion_forgery = deepcopy(serialized)
    conclusion_forgery["inequalities"][0]["slack"] = _rational(Fraction())
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessCheckResult.model_validate(conclusion_forgery)

    validity_forgery = deepcopy(serialized)
    validity_forgery["valid"] = False
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessCheckResult.model_validate(validity_forgery)

    failure_forgery = deepcopy(serialized)
    failure_forgery["failed_event_indices"] = [0]
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessCheckResult.model_validate(failure_forgery)

    product_forgery = deepcopy(serialized)
    product_forgery["witness_product"] = _rational(Fraction(1, 3))
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessCheckResult.model_validate(product_forgery)


def test_numerical_validity_is_not_labeled_dependency_graph_correctness() -> None:
    result = _compute(_payload(("A",), (Fraction(),), (Fraction(),), ((),)))

    assert result.valid is True
    assert "dependency" not in AsymmetricLocalLemmaWitnessCheckResult.model_fields
    assert "does not establish" in ASYMMETRIC_LOCAL_LEMMA_OPERATION.description
    assert "dependency graph" in ASYMMETRIC_LOCAL_LEMMA_OPERATION.description


def test_native_function_returns_the_source_bound_canonical_value() -> None:
    source = AsymmetricLocalLemmaWitness(
        event_labels=("A",),
        probability_upper_bounds=(Fraction(1, 4),),
        witness_parameters=(Fraction(1, 2),),
        neighborhoods=((0,),),
    )

    result = check_asymmetric_local_lemma_witness(source)

    assert result.source is source
    assert result.valid is True
    assert result.inequalities[0].slack == 0


def test_native_result_rejects_boolean_failure_index_alias() -> None:
    source = AsymmetricLocalLemmaWitness(
        event_labels=("A",),
        probability_upper_bounds=(Fraction(1, 2),),
        witness_parameters=(Fraction(1, 3),),
        neighborhoods=((),),
    )
    result = check_asymmetric_local_lemma_witness(source)

    with pytest.raises(TypeError):
        replace(result, failed_event_indices=(False,))


def test_raw_input_rational_digit_bound_precedes_canonical_integer_parsing() -> None:
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(sys.int_info.default_max_str_digits)
    try:
        for oversized in (
            "1" * (MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS + 1),
            "-" * (MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS + 1) + "1",
        ):
            payload = _payload(("A",), (Fraction(),), (Fraction(),), ((),))
            payload["probability_upper_bounds"] = [{"num": oversized, "den": "1"}]
            with pytest.raises(ValidationError):
                AsymmetricLocalLemmaWitnessRequest.model_validate(payload)
    finally:
        sys.set_int_max_str_digits(previous_limit)


def test_raw_result_digit_bound_precedes_source_replay() -> None:
    serialized = _compute(
        _payload(("A",), (Fraction(),), (Fraction(),), ((),))
    ).model_dump(mode="json")
    serialized["inequalities"][0]["neighborhood_product"] = {
        "num": "1" * (MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS + 1),
        "den": "1",
    }

    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessCheckResult.model_validate(serialized)


def test_event_count_is_rejected_in_raw_preflight() -> None:
    payload = {
        "event_labels": [f"E{index}" for index in range(MAX_LOCAL_LEMMA_EVENTS + 1)],
        "probability_upper_bounds": [],
        "witness_parameters": [],
        "neighborhoods": [],
    }
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(payload)


def test_incidence_count_is_rejected_in_raw_preflight() -> None:
    labels = tuple(f"E{index}" for index in range(MAX_LOCAL_LEMMA_EVENTS))
    full_neighborhood = tuple(range(MAX_LOCAL_LEMMA_EVENTS))
    full_rows = MAX_LOCAL_LEMMA_INCIDENCES // MAX_LOCAL_LEMMA_EVENTS + 1
    payload = _payload(
        labels,
        (Fraction(),) * MAX_LOCAL_LEMMA_EVENTS,
        (Fraction(),) * MAX_LOCAL_LEMMA_EVENTS,
        (full_neighborhood,) * full_rows + ((),) * (MAX_LOCAL_LEMMA_EVENTS - full_rows),
    )
    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(payload)


def test_preflight_rejects_one_overgrown_exact_result_component() -> None:
    event_count = 130
    denominator = 10**255
    payload = _payload(
        tuple(f"E{index}" for index in range(event_count)),
        (Fraction(),) * event_count,
        (Fraction(1, denominator),) * event_count,
        (tuple(range(1, event_count)),) + ((),) * (event_count - 1),
    )

    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(payload)


def test_preflight_rejects_overgrown_complete_result_ledger() -> None:
    event_count = 200
    denominator = 10**255
    payload = _payload(
        tuple(f"E{index}" for index in range(event_count)),
        (Fraction(),) * event_count,
        (Fraction(1, denominator),) * event_count,
        (tuple(range(8)),) * event_count,
    )

    with pytest.raises(ValidationError):
        AsymmetricLocalLemmaWitnessRequest.model_validate(payload)


def test_operation_declares_the_exact_public_contract() -> None:
    assert ASYMMETRIC_LOCAL_LEMMA_OPERATION.operation_id == (
        "probability.local_lemma.asymmetric_witness.check"
    )
    example_request = AsymmetricLocalLemmaWitnessRequest.model_validate(
        ASYMMETRIC_LOCAL_LEMMA_OPERATION.examples[0].input
    )
    example_result = ASYMMETRIC_LOCAL_LEMMA_OPERATION.run(example_request)
    assert isinstance(example_result, AsymmetricLocalLemmaWitnessCheckResult)
    assert example_result.valid is True
