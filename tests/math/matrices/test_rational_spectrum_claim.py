"""Exact complete rational spectrum-claim tests."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

import pytest
from pydantic import ValidationError

from jacobian.math.matrices.analysis._models import (
    MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
    MAX_RATIONAL_SPECTRUM_RESULT_BYTES,
    RationalSpectrumClaimRequest,
    RationalSpectrumClaimResult,
)
from jacobian.math.matrices.analysis._operations import (
    check_rational_spectrum_claim,
)


def _rational(value: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(value), "den": str(denominator)}


def _matrix(entries: list[list[dict[str, str]]]) -> dict[str, object]:
    return {"entries": entries}


def _claim(
    eigenvalue: int,
    multiplicity: int,
    denominator: int = 1,
) -> dict[str, object]:
    return {
        "eigenvalue": _rational(eigenvalue, denominator),
        "multiplicity": multiplicity,
    }


def _request(
    entries: list[list[dict[str, str]]],
    claims: list[dict[str, object]],
) -> RationalSpectrumClaimRequest:
    return RationalSpectrumClaimRequest.model_validate(
        {
            "matrix": _matrix(entries),
            "claimed_profile": claims,
        }
    )


def _mutate_source_matrix(data: dict[str, Any]) -> None:
    matrix = cast(dict[str, Any], data["matrix"])
    entries = cast(list[list[Any]], matrix["entries"])
    entries[0][0] = _rational(3)


def _mutate_claim(data: dict[str, Any]) -> None:
    profile = cast(list[dict[str, Any]], data["claimed_profile"])
    profile[0]["multiplicity"] = 2


def _mutate_nullity(data: dict[str, Any]) -> None:
    ledger = cast(list[dict[str, Any]], data["nullity_ledger"])
    ledger[0]["exact_nullity"] = 0


def _mutate_validity(data: dict[str, Any]) -> None:
    data["valid_complete_rational_spectrum"] = False


def _mutate_outcome(data: dict[str, Any]) -> None:
    data["outcome"] = "INVALID"


def test_complete_repeated_rational_spectrum_binds_exact_nullities() -> None:
    request = _request(
        [
            [_rational(2), _rational(0), _rational(0)],
            [_rational(0), _rational(2), _rational(0)],
            [_rational(0), _rational(0), _rational(-1)],
        ],
        [_claim(2, 2), _claim(-1, 1)],
    )

    result = check_rational_spectrum_claim(request)

    assert result.outcome == "VALID"
    assert result.valid_complete_rational_spectrum is True
    assert result.first_failed_condition is None
    assert result.first_failed_claim_index is None
    assert [entry.exact_nullity for entry in result.nullity_ledger] == [2, 1]
    assert result.claimed_multiplicity_sum == 3
    assert result.established_multiplicity_sum == 3
    assert (
        RationalSpectrumClaimResult.model_validate(result.model_dump(mode="json"))
        == result
    )


def test_zero_matrix_and_nonintegral_rational_spectrum() -> None:
    zero = check_rational_spectrum_claim(
        _request(
            [
                [_rational(0), _rational(0), _rational(0)],
                [_rational(0), _rational(0), _rational(0)],
                [_rational(0), _rational(0), _rational(0)],
            ],
            [_claim(0, 3)],
        )
    )
    assert zero.valid_complete_rational_spectrum is True
    assert zero.nullity_ledger[0].exact_nullity == 3

    rational = check_rational_spectrum_claim(
        _request(
            [
                [_rational(1, 2), _rational(0)],
                [_rational(0), _rational(-2, 3)],
            ],
            [_claim(1, 1, 2), _claim(-2, 1, 3)],
        )
    )
    assert rational.valid_complete_rational_spectrum is True
    assert [entry.exact_nullity for entry in rational.nullity_ledger] == [1, 1]


def test_irrational_spectrum_is_an_exact_invalid_claim() -> None:
    # [[0, 1], [1, 1]] has characteristic polynomial x^2 - x - 1.
    result = check_rational_spectrum_claim(
        _request(
            [[_rational(0), _rational(1)], [_rational(1), _rational(1)]],
            [_claim(0, 2)],
        )
    )

    assert result.outcome == "INVALID"
    assert result.valid_complete_rational_spectrum is False
    assert result.nullity_ledger[0].exact_nullity == 0
    assert result.established_multiplicity_sum == 0
    assert result.first_failed_condition == "MULTIPLICITY_MISMATCH"
    assert result.first_failed_claim_index == 0


def test_correct_subset_fails_only_the_completeness_sum() -> None:
    result = check_rational_spectrum_claim(
        _request(
            [
                [_rational(1), _rational(0), _rational(0)],
                [_rational(0), _rational(1), _rational(0)],
                [_rational(0), _rational(0), _rational(2)],
            ],
            [_claim(1, 2)],
        )
    )

    assert result.nullity_ledger[0].multiplicity_matches is True
    assert result.first_failed_condition == (
        "CLAIMED_MULTIPLICITY_SUM_DOES_NOT_EQUAL_MATRIX_ORDER"
    )
    assert result.first_failed_claim_index is None
    assert result.established_multiplicity_sum == 2


def test_first_wrong_multiplicity_is_reported_in_submission_order() -> None:
    result = check_rational_spectrum_claim(
        _request(
            [
                [_rational(1), _rational(0), _rational(0)],
                [_rational(0), _rational(1), _rational(0)],
                [_rational(0), _rational(0), _rational(2)],
            ],
            [_claim(1, 1), _claim(2, 2)],
        )
    )

    assert [entry.exact_nullity for entry in result.nullity_ledger] == [2, 1]
    assert [entry.multiplicity_matches for entry in result.nullity_ledger] == [
        False,
        False,
    ]
    assert result.first_failed_condition == "MULTIPLICITY_MISMATCH"
    assert result.first_failed_claim_index == 0


@pytest.mark.parametrize(
    ("entries", "claims", "message"),
    [
        (
            [[_rational(1), _rational(1)], [_rational(0), _rational(1)]],
            [_claim(1, 2)],
            "symmetric",
        ),
        (
            [[_rational(1), _rational(0)], [_rational(0), _rational(2)]],
            [_claim(1, 1), _claim(1, 1)],
            "pairwise distinct",
        ),
        (
            [[_rational(1)]],
            [_claim(1, 0)],
            "greater than or equal to 1",
        ),
    ],
)
def test_invalid_request_domain_is_rejected_before_backend(
    entries: list[list[dict[str, str]]],
    claims: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _request(entries, claims)


def test_source_claim_ledger_and_conclusion_mutations_cannot_revalidate() -> None:
    result = check_rational_spectrum_claim(
        _request(
            [[_rational(1), _rational(0)], [_rational(0), _rational(2)]],
            [_claim(1, 1), _claim(2, 1)],
        )
    )

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        _mutate_source_matrix,
        _mutate_claim,
        _mutate_nullity,
        _mutate_validity,
        _mutate_outcome,
    )
    for mutate in mutations:
        forged = deepcopy(result.model_dump(mode="json"))
        mutate(forged)
        with pytest.raises(ValidationError, match="does not match exact replay"):
            RationalSpectrumClaimResult.model_validate(forged)


def test_simultaneous_row_column_permutation_preserves_claim() -> None:
    entries = [
        [_rational(2), _rational(1), _rational(0)],
        [_rational(1), _rational(2), _rational(0)],
        [_rational(0), _rational(0), _rational(4)],
    ]
    claims = [_claim(1, 1), _claim(3, 1), _claim(4, 1)]
    original = check_rational_spectrum_claim(_request(entries, claims))
    permutation = (2, 0, 1)
    permuted_entries = [
        [entries[row][column] for column in permutation] for row in permutation
    ]
    permuted = check_rational_spectrum_claim(_request(permuted_entries, claims))

    assert original.valid_complete_rational_spectrum is True
    assert permuted.valid_complete_rational_spectrum is True
    assert permuted.nullity_ledger == original.nullity_ledger


def test_order_claim_count_digit_and_result_boundaries() -> None:
    order = 32
    diagonal = [
        [_rational(row if row == column else 0) for column in range(order)]
        for row in range(order)
    ]
    claims = [_claim(index, 1) for index in range(order)]
    boundary = check_rational_spectrum_claim(_request(diagonal, claims))
    assert boundary.valid_complete_rational_spectrum is True
    assert len(boundary.model_dump_json()) < MAX_RATIONAL_SPECTRUM_RESULT_BYTES

    dense = [[_rational(1) for _ in range(order)] for _ in range(order)]
    dense_boundary = check_rational_spectrum_claim(
        _request(dense, [_claim(order, 1), _claim(0, order - 1)])
    )
    assert dense_boundary.valid_complete_rational_spectrum is True

    too_many_claims = [*claims, _claim(order, 1)]
    with pytest.raises(ValidationError, match="at most 32 items"):
        _request(diagonal, too_many_claims)

    max_scalar = int("9" * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS)
    scalar_boundary = check_rational_spectrum_claim(
        _request([[_rational(max_scalar)]], [_claim(max_scalar, 1)])
    )
    assert scalar_boundary.valid_complete_rational_spectrum is True

    over_scalar = int("1" + "0" * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS)
    with pytest.raises(ValidationError, match="limited to 64 decimal digits"):
        _request([[_rational(over_scalar)]], [_claim(0, 1)])
