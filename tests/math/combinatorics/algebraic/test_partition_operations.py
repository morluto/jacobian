"""Tests for exact partition and tableau operations added for #1825."""

from __future__ import annotations

import json
from typing import NoReturn

import pytest
from jsonschema import Draft202012Validator
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


def test_hook_content_count() -> None:
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [2, 1]}, "alphabet_size": 2}
        )
    )
    assert result.count == 2


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


def test_hook_content_alphabet_is_independent_of_partition_size() -> None:
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [1]}, "alphabet_size": 501}
        )
    )
    assert result.count == 501


def test_hook_content_accepts_maximum_digit_one_cell_alphabet() -> None:
    alphabet_size = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    result = hook_content_count(
        HookContentCountRequest.model_validate(
            {"partition": {"parts": [1]}, "alphabet_size": alphabet_size}
        )
    )
    assert result.count == alphabet_size


def test_hook_content_rejects_wire_bound_growth_before_expansion() -> None:
    request = HookContentCountRequest.model_validate_json(
        json.dumps(
            {
                "partition": {"parts": [1] * 500},
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
                    "partition": {"parts": [1] * 500},
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
    assert set(wire) == {"partition", "alphabet_size", "count"}
    assert (
        HookContentCountResult.model_validate_json(json.dumps(wire), strict=True)
        == result
    )


def test_hook_content_result_rejects_derivation_artifacts() -> None:
    payload = {
        "partition": {"parts": [2, 1]},
        "alphabet_size": "2",
        "count": "2",
        "numerators": ["2", "3", "1"],
        "hook_product": "3",
    }
    with pytest.raises(ValidationError):
        HookContentCountResult.model_validate_json(json.dumps(payload), strict=True)


def test_hook_content_charges_growing_product_work() -> None:
    # 500 cells over a 64-digit alphabet fit the 32,768-digit output bound
    # (32,000 digits) but exceed the admitted bigint work envelope.
    request = HookContentCountRequest.model_validate(
        {"partition": {"parts": [1] * 500}, "alphabet_size": 10**63}
    )
    with pytest.raises(
        OperationResourceAdmissionError,
        match="hook-content arithmetic exceeds the admitted work bound",
    ):
        hook_content_count(request)


def test_hook_content_admits_two_cell_large_alphabet() -> None:
    # One 3000-digit multiplication plus an exact division: microseconds of
    # Karatsuba work returning a 6000-digit count within the output bound.
    alphabet_size = 10**3000
    request = HookContentCountRequest.model_validate(
        {"partition": {"parts": [2]}, "alphabet_size": alphabet_size}
    )
    result = hook_content_count(request)
    assert result.count == alphabet_size * (alphabet_size + 1) // 2
    assert result.alphabet_size == alphabet_size


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
    assert standard.tableau.rows == ((1, 2), (3,))
    assert standard.is_member is True
    assert semistandard.tableau.rows == ((1, 1), (2,))
    assert semistandard.is_member is True


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
def test_standard_tableau_checker_returns_source_bound_false_for_nonmembers(
    candidate: StandardTableauCheckRequest,
) -> None:
    result = check_standard_tableau(candidate)
    assert result.tableau == candidate.tableau
    assert result.is_member is False


def test_semistandard_tableau_checker_returns_source_bound_false_for_nonmembers() -> (
    None
):
    request = SemistandardTableauCheckRequest.model_validate(
        {"tableau": {"rows": [[1, 2], [1]]}}
    )
    result = check_semistandard_tableau(request)
    assert result.tableau == request.tableau
    assert result.is_member is False


def test_tableau_checkers_return_false_for_non_partition_shapes() -> None:
    # Row lengths (1, 2) do not form a Young diagram; the shared partition
    # validator wraps that shape failure, which must read as nonmembership.
    standard = check_standard_tableau(
        StandardTableauCheckRequest.model_validate({"tableau": {"rows": [[1], [2, 3]]}})
    )
    assert standard.is_member is False
    semistandard = check_semistandard_tableau(
        SemistandardTableauCheckRequest.model_validate(
            {"tableau": {"rows": [[1], [1, 1]]}}
        )
    )
    assert semistandard.is_member is False


def test_tableau_checker_propagates_operational_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = StandardTableauCheckRequest.model_validate({"tableau": {"rows": [[1]]}})

    def fail(_tableau: StandardYoungTableau) -> NoReturn:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(native, "check_standard_tableau", fail)
    with pytest.raises(RuntimeError, match="backend failure"):
        check_standard_tableau(request)


@pytest.mark.parametrize("value", ["0", "-1", "01", "1\n"])
def test_hook_content_positive_exact_integer_schema_matches_runtime(
    value: str,
) -> None:
    request_schema = HookContentCountRequest.model_json_schema()
    result_schema = HookContentCountResult.model_json_schema()
    request_payload = {"partition": {"parts": [1]}, "alphabet_size": value}
    result_payload = {
        "partition": {"parts": [1]},
        "alphabet_size": value,
        "count": "0",
    }

    assert list(Draft202012Validator(request_schema).iter_errors(request_payload))
    assert list(Draft202012Validator(result_schema).iter_errors(result_payload))
    with pytest.raises(ValidationError):
        HookContentCountRequest.model_validate_json(
            json.dumps(request_payload), strict=True
        )
    with pytest.raises(ValidationError):
        HookContentCountResult.model_validate_json(
            json.dumps(result_payload), strict=True
        )


def test_hook_content_count_schema_publishes_nonnegative_count() -> None:
    schema = HookContentCountResult.model_json_schema()
    count_schema = schema["properties"]["count"]
    assert count_schema["examples"] == ["0", "2"]
    assert "(?:0|" in count_schema["pattern"]
    assert (
        HookContentCountResult.model_validate_json(
            json.dumps(
                {
                    "partition": {"parts": [1, 1]},
                    "alphabet_size": "1",
                    "count": "0",
                }
            ),
            strict=True,
        ).count
        == 0
    )


def test_native_operations_are_published_from_algebraic_package() -> None:
    partition = IntegerPartition(parts=(1,))
    standard = StandardYoungTableau(rows=((1,),))
    semistandard = SemistandardYoungTableau(rows=((1,),))

    assert public_hook_content_count(partition, 501) == 501
    assert (
        public_partition_dominance(
            IntegerPartition(parts=(2, 1)), IntegerPartition(parts=(1, 1, 1))
        )[0]
        == "LEFT_DOMINATES"
    )
    assert public_check_standard_tableau(standard) == standard
    assert public_check_semistandard_tableau(semistandard) == semistandard
