"""Tests for exact partition and tableau operations added for #1825."""

from __future__ import annotations

import json
from typing import NoReturn

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic import (
    check_semistandard_tableau as public_check_semistandard_tableau,
)
from jacobian.math.combinatorics.algebraic import (
    check_standard_tableau as public_check_standard_tableau,
)
from jacobian.math.combinatorics.algebraic import (
    hook_content_count as public_hook_content_count,
)
from jacobian.math.combinatorics.algebraic import operations as native
from jacobian.math.combinatorics.algebraic import (
    partition_dominance as public_partition_dominance,
)
from jacobian.math.combinatorics.algebraic._models import (
    HookContentCountRequest,
    HookContentCountResult,
    PartitionDominanceRequest,
    SemistandardTableauCheckRequest,
    StandardTableauCheckRequest,
)
from jacobian.math.combinatorics.algebraic._tools import (
    check_semistandard_tableau,
    check_standard_tableau,
    hook_content_count,
    partition_dominance,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)


def test_hook_content_count_and_factors() -> None:
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [2, 1]}, "alphabet_size": 2}
        )
    )
    assert result.count == 2
    assert result.numerators == (2, 3, 1)
    assert result.hook_product == 3


def test_hook_content_zero_when_alphabet_is_too_small() -> None:
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [1, 1]}, "alphabet_size": 1}
        )
    )
    assert result.count == 0


def test_hook_content_empty_shape() -> None:
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": []}, "alphabet_size": 3}
        )
    )
    assert result.count == 1
    assert result.numerators == ()
    assert result.hook_product == 1


def test_hook_content_alphabet_is_independent_of_partition_size() -> None:
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [1]}, "alphabet_size": 501}
        )
    )
    assert result.count == 501
    assert result.numerators == (501,)
    assert result.hook_product == 1


def test_hook_content_admits_wire_bound_but_rejects_work_before_expansion() -> None:
    request = HookContentCountRequest.model_validate_json(
        json.dumps(
            {
                "partition": {"parts": [1]},
                "alphabet_size": "1" + "0" * (MAX_CANONICAL_INTEGER_DIGITS - 1),
            }
        ),
        strict=True,
    )
    with pytest.raises(OperationResourceAdmissionError):
        hook_content_count(request)

    with pytest.raises(ValidationError):
        HookContentCountRequest.model_validate_json(
            json.dumps(
                {
                    "partition": {"parts": [1]},
                    "alphabet_size": "1" + "0" * MAX_CANONICAL_INTEGER_DIGITS,
                }
            ),
            strict=True,
        )


def test_hook_content_large_exact_integers_roundtrip_strict_json() -> None:
    alphabet_size = (1 << 53) + 1
    request = HookContentCountRequest.model_validate_json(
        json.dumps({"partition": {"parts": [1]}, "alphabet_size": str(alphabet_size)}),
        strict=True,
    )
    result = hook_content_count(request)
    wire = json.loads(result.model_dump_json())

    assert wire["alphabet_size"] == str(alphabet_size)
    assert wire["numerators"] == [str(alphabet_size)]
    assert (
        HookContentCountResult.model_validate_json(json.dumps(wire), strict=True)
        == result
    )


def test_hook_content_charges_growing_product_work() -> None:
    request = HookContentCountRequest.model_validate(
        {"partition": {"parts": [1] * 500}, "alphabet_size": 10_000_000}
    )
    with pytest.raises(
        OperationResourceAdmissionError,
        match="hook-content arithmetic exceeds the admitted work bound",
    ):
        hook_content_count(request)


@pytest.mark.parametrize(
    ("left", "right", "relation"),
    [
        ((2, 1), (1, 1, 1), "LEFT_DOMINATES"),
        ((1, 1, 1), (2, 1), "RIGHT_DOMINATES"),
        ((2, 2), (2, 2), "EQUAL"),
        ((3, 1, 1, 1), (2, 2, 2), "INCOMPARABLE"),
    ],
)
def test_partition_dominance_ledger(
    left: tuple[int, ...], right: tuple[int, ...], relation: str
) -> None:
    result = partition_dominance(
        PartitionDominanceRequest.model_validate(
            {"left": {"parts": left}, "right": {"parts": right}}
        )
    )
    assert result.relation == relation
    assert len(result.left_prefix_sums) == max(len(left), len(right))
    assert result.left_prefix_sums[-1] == result.right_prefix_sums[-1]


def test_partition_dominance_different_sizes_is_distinct() -> None:
    result = partition_dominance(
        PartitionDominanceRequest.model_validate(
            {"left": {"parts": [2]}, "right": {"parts": [1]}}
        )
    )
    assert result.relation == "NOT_COMPARABLE_DIFFERENT_SIZE"
    assert result.left_prefix_sums == ()


def test_tableau_checkers_replay_membership() -> None:
    standard = check_standard_tableau(
        StandardTableauCheckRequest.model_validate({"tableau": {"rows": [[1, 2], [3]]}})
    )
    semistandard = check_semistandard_tableau(
        SemistandardTableauCheckRequest.model_validate(
            {"tableau": {"rows": [[1, 1], [2]]}}
        )
    )
    assert standard.rows == ((1, 2), (3,))
    assert semistandard.rows == ((1, 1), (2,))


@pytest.mark.parametrize(
    "candidate",
    [
        StandardTableauCheckRequest.model_validate(
            {"tableau": {"rows": [[1, 1], [2]]}}
        ),
        StandardTableauCheckRequest.model_validate(
            {"tableau": {"rows": [[1, 2], [2]]}}
        ),
    ],
)
def test_standard_tableau_checker_rejects_row_or_column_failure(
    candidate: StandardTableauCheckRequest,
) -> None:
    with pytest.raises(ValueError):
        check_standard_tableau(candidate)


def test_semistandard_tableau_checker_rejects_column_failure() -> None:
    request = SemistandardTableauCheckRequest.model_validate(
        {"tableau": {"rows": [[1, 2], [1]]}}
    )
    with pytest.raises(ValueError):
        check_semistandard_tableau(request)


def test_tableau_checker_propagates_operational_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = StandardTableauCheckRequest.model_validate({"tableau": {"rows": [[1]]}})

    def fail(_tableau: StandardYoungTableau) -> NoReturn:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(native, "check_standard_tableau", fail)
    with pytest.raises(RuntimeError, match="backend failure"):
        check_standard_tableau(request)


def test_hook_content_alphabet_bound_is_published() -> None:
    with pytest.raises(ValidationError):
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [1]}, "alphabet_size": 0}
        )


def test_native_operations_are_published_from_algebraic_package() -> None:
    partition = IntegerPartition(parts=(1,))
    standard = StandardYoungTableau(rows=((1,),))
    semistandard = SemistandardYoungTableau(rows=((1,),))

    assert public_hook_content_count(partition, 501) == (501, (501,), 1)
    assert (
        public_partition_dominance(
            IntegerPartition(parts=(2, 1)), IntegerPartition(parts=(1, 1, 1))
        )[0]
        == "LEFT_DOMINATES"
    )
    assert public_check_standard_tableau(standard) == standard
    assert public_check_semistandard_tableau(semistandard) == semistandard
