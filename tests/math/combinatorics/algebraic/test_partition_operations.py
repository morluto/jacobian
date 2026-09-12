"""Tests for exact partition and tableau operations added for #1825."""

from __future__ import annotations

import json
import math
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
from jacobian.math.combinatorics.algebraic import operations as native
from jacobian.math.combinatorics.algebraic import (
    partition_dominance as public_partition_dominance,
)
from jacobian.math.combinatorics.algebraic import (
    semistandard_young_tableaux_count as public_semistandard_young_tableaux_count,
)
from jacobian.math.combinatorics.algebraic._models import (
    PartitionDominanceRequest,
    SemistandardTableauCheckRequest,
    SemistandardTableauCheckResult,
    SemistandardYoungTableauCountRequest,
    SemistandardYoungTableauCountResult,
    StandardTableauCheckRequest,
    StandardTableauCheckResult,
)
from jacobian.math.combinatorics.algebraic._tools import (
    check_semistandard_tableau,
    check_standard_tableau,
    partition_dominance,
    semistandard_young_tableaux_count,
)
from jacobian.math.combinatorics.algebraic.operations import (
    _MAX_SSYT_COUNT_DIGITS,
    _ssyt_count_digit_bound,
    _upper_decimal_digits,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)


def test_semistandard_young_tableaux_count() -> None:
    result = semistandard_young_tableaux_count(
        SemistandardYoungTableauCountRequest.model_validate(
            {"partition": {"parts": [2, 1]}, "alphabet_size": 2}
        )
    )
    assert result.count == 2


def test_hook_content_zero_when_alphabet_is_too_small() -> None:
    result = semistandard_young_tableaux_count(
        SemistandardYoungTableauCountRequest.model_validate(
            {"partition": {"parts": [1, 1]}, "alphabet_size": 1}
        )
    )
    assert result.count == 0


def test_hook_content_empty_shape() -> None:
    result = semistandard_young_tableaux_count(
        SemistandardYoungTableauCountRequest.model_validate(
            {"partition": {"parts": []}, "alphabet_size": 3}
        )
    )
    assert result.count == 1


def test_hook_content_alphabet_is_independent_of_partition_size() -> None:
    result = semistandard_young_tableaux_count(
        SemistandardYoungTableauCountRequest.model_validate(
            {"partition": {"parts": [1]}, "alphabet_size": 501}
        )
    )
    assert result.count == 501


def test_hook_content_accepts_maximum_digit_one_cell_alphabet() -> None:
    alphabet_size = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    result = semistandard_young_tableaux_count(
        SemistandardYoungTableauCountRequest.model_validate(
            {"partition": {"parts": [1]}, "alphabet_size": alphabet_size}
        )
    )
    assert result.count == alphabet_size


def test_hook_content_rejects_wire_bound_growth_before_expansion() -> None:
    request = SemistandardYoungTableauCountRequest.model_validate_json(
        json.dumps(
            {
                "partition": {"parts": [1] * 500},
                "alphabet_size": "1" + "0" * (MAX_CANONICAL_INTEGER_DIGITS - 1),
            }
        ),
        strict=True,
    )
    with pytest.raises(OperationResourceAdmissionError):
        semistandard_young_tableaux_count(request)

    with pytest.raises(ValidationError):
        SemistandardYoungTableauCountRequest.model_validate_json(
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
    request = SemistandardYoungTableauCountRequest.model_validate_json(
        json.dumps({"partition": {"parts": [1]}, "alphabet_size": str(alphabet_size)}),
        strict=True,
    )
    result = semistandard_young_tableaux_count(request)
    wire = json.loads(result.model_dump_json())

    assert wire["alphabet_size"] == str(alphabet_size)
    assert set(wire) == {"partition", "alphabet_size", "count"}
    assert (
        SemistandardYoungTableauCountResult.model_validate_json(
            json.dumps(wire), strict=True
        )
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
        SemistandardYoungTableauCountResult.model_validate_json(
            json.dumps(payload), strict=True
        )


def test_hook_content_admits_growing_product_within_work_bound() -> None:
    # 500 cells over a 64-digit alphabet fit the 32,768-digit output bound.
    # A balanced product tree keeps this request inside the work envelope.
    alphabet_size = 10**63
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [1] * 500}, "alphabet_size": alphabet_size}
    )
    result = semistandard_young_tableaux_count(request)
    assert result.count == math.comb(alphabet_size, 500)
    assert result.alphabet_size == alphabet_size


def test_hook_content_admits_two_cell_large_alphabet() -> None:
    # One 3000-digit multiplication plus an exact division: microseconds of
    # Karatsuba work returning a 6000-digit count within the output bound.
    alphabet_size = 10**3000
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [2]}, "alphabet_size": alphabet_size}
    )
    result = semistandard_young_tableaux_count(request)
    assert result.count == alphabet_size * (alphabet_size + 1) // 2
    assert result.alphabet_size == alphabet_size


def test_hook_content_admits_two_cell_count_near_digit_limit() -> None:
    # One ~16000-digit multiplication plus an exact division returns a
    # 32,000-digit count within the output bound; the level-by-level
    # Karatsuba estimate charges about 1.6M units against the 8M envelope
    # instead of recreating the large-scalar rejection at the boundary.
    alphabet_size = 10**16000
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [2]}, "alphabet_size": alphabet_size}
    )
    result = semistandard_young_tableaux_count(request)
    assert result.count == alphabet_size * (alphabet_size + 1) // 2
    assert result.alphabet_size == alphabet_size


def test_hook_content_admits_multi_cell_count_near_digit_limit() -> None:
    # Fifty ~655-digit factors yield a 32,636-digit count. A balanced product
    # tree keeps the work well below the 8M envelope; left-to-right charging
    # of growing accumulators previously rejected this cheap request.
    alphabet_size = 10**654
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [50]}, "alphabet_size": alphabet_size}
    )
    result = semistandard_young_tableaux_count(request)
    expected = 1
    for column in range(50):
        expected *= alphabet_size + column
    expected //= math.factorial(50)
    assert result.count == expected
    assert result.alphabet_size == alphabet_size


def test_hook_content_admits_column_count_after_hook_cancellation() -> None:
    # The 500-row column with alphabet 10**67 has an uncancelled numerator
    # bound past 32,768 digits, but C(10**67, 500) has 32,366 digits and is
    # cheap once hook cancellation is charged in admission.
    alphabet_size = 10**67
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [1] * 500}, "alphabet_size": alphabet_size}
    )
    result = semistandard_young_tableaux_count(request)
    assert result.count == math.comb(alphabet_size, 500)
    assert result.alphabet_size == alphabet_size


def test_ssyt_count_admits_power_of_two_row_after_tighter_log_bound() -> None:
    """C(2**225 + 499, 500) has 32,732 digits and stays inside the envelope."""

    alphabet_size = 2**225
    partition = IntegerPartition(parts=(500,))
    bound = _ssyt_count_digit_bound(
        partition,
        alphabet_size,
        _upper_decimal_digits(alphabet_size),
    )
    exact = math.comb(alphabet_size + 499, 500)
    exact_digits = _upper_decimal_digits(exact)
    assert exact_digits == 32732
    assert bound >= exact_digits
    assert bound <= _MAX_SSYT_COUNT_DIGITS
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [500]}, "alphabet_size": alphabet_size}
    )
    result = semistandard_young_tableaux_count(request)
    assert result.count == exact


def test_hook_content_rejects_column_count_beyond_digit_limit() -> None:
    # C(10**68, 500) has 32,866 digits. ceil(log10)+1 must exceed the
    # 32,768-digit ExactInteger envelope so admission rejects first.
    alphabet_size = 10**68
    request = SemistandardYoungTableauCountRequest.model_validate(
        {"partition": {"parts": [1] * 500}, "alphabet_size": alphabet_size}
    )
    with pytest.raises(OperationResourceAdmissionError, match="digit bound"):
        semistandard_young_tableaux_count(request)
    exact_digits = _upper_decimal_digits(math.comb(alphabet_size, 500))
    bound = _ssyt_count_digit_bound(
        IntegerPartition(parts=(1,) * 500),
        alphabet_size,
        _upper_decimal_digits(alphabet_size),
    )
    assert exact_digits == 32866
    assert bound >= exact_digits
    assert bound > _MAX_SSYT_COUNT_DIGITS


@pytest.mark.parametrize(
    ("left", "right", "relation"),
    [
        ((2, 1), (1, 1, 1), "LEFT_DOMINATES"),
        ((1, 1, 1), (2, 1), "RIGHT_DOMINATES"),
        ((2, 2), (2, 2), "EQUAL"),
        ((3, 1, 1, 1), (2, 2, 2), "INCOMPARABLE"),
    ],
)
def test_partition_dominance_relation(
    left: tuple[int, ...], right: tuple[int, ...], relation: str
) -> None:
    result = partition_dominance(
        PartitionDominanceRequest.model_validate(
            {"left": {"parts": left}, "right": {"parts": right}}
        )
    )
    assert result.relation == relation
    assert not hasattr(result, "left_prefix_sums")
    assert result.left.parts[: len(left)] == left
    assert result.right.parts[: len(right)] == right


def test_partition_dominance_different_sizes_is_distinct() -> None:
    result = partition_dominance(
        PartitionDominanceRequest.model_validate(
            {"left": {"parts": [2]}, "right": {"parts": [1]}}
        )
    )
    assert result.relation == "NOT_COMPARABLE_DIFFERENT_SIZE"


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


def test_tableau_checkers_reject_oversized_carriers_at_admission() -> None:
    # Rows [[1, ..., 500], [501]] are mathematically a standard tableau of
    # shape (500, 1), but 501 cells exceed the canonical budget: request
    # admission must fail rather than report a false nonmembership.
    oversized = [list(range(1, 501)), [501]]
    with pytest.raises(ValidationError) as standard_error:
        StandardTableauCheckRequest.model_validate({"tableau": {"rows": oversized}})
    assert (
        standard_error.value.errors()[0]["type"]
        == "symmetric_function.tableau_size_exceeded"
    )
    with pytest.raises(ValidationError) as semistandard_error:
        SemistandardTableauCheckRequest.model_validate({"tableau": {"rows": oversized}})
    assert (
        semistandard_error.value.errors()[0]["type"]
        == "symmetric_function.tableau_size_exceeded"
    )


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
    request_schema = SemistandardYoungTableauCountRequest.model_json_schema()
    result_schema = SemistandardYoungTableauCountResult.model_json_schema()
    request_payload = {"partition": {"parts": [1]}, "alphabet_size": value}
    result_payload = {
        "partition": {"parts": [1]},
        "alphabet_size": value,
        "count": "0",
    }

    assert list(Draft202012Validator(request_schema).iter_errors(request_payload))
    assert list(Draft202012Validator(result_schema).iter_errors(result_payload))
    with pytest.raises(ValidationError):
        SemistandardYoungTableauCountRequest.model_validate_json(
            json.dumps(request_payload), strict=True
        )
    with pytest.raises(ValidationError):
        SemistandardYoungTableauCountResult.model_validate_json(
            json.dumps(result_payload), strict=True
        )


def test_semistandard_young_tableaux_count_schema_publishes_nonnegative_count() -> None:
    schema = SemistandardYoungTableauCountResult.model_json_schema()
    count_schema = schema["properties"]["count"]
    assert count_schema["examples"] == ["0", "2"]
    assert "(?:0|" in count_schema["pattern"]
    assert (
        SemistandardYoungTableauCountResult.model_validate_json(
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

    assert public_semistandard_young_tableaux_count(partition, 501) == 501
    assert (
        public_partition_dominance(
            IntegerPartition(parts=(2, 1)), IntegerPartition(parts=(1, 1, 1))
        )
        == "LEFT_DOMINATES"
    )
    assert public_check_standard_tableau(standard) == StandardTableauCheckResult(
        tableau=standard, is_member=True
    )
    assert public_check_semistandard_tableau(
        semistandard
    ) == SemistandardTableauCheckResult(tableau=semistandard, is_member=True)


def test_native_tableau_checks_return_typed_nonmembership() -> None:
    standard = StandardYoungTableau(rows=((1, 1), (2,)))
    semistandard = SemistandardYoungTableau(rows=((1, 2), (1,)))
    assert native.check_standard_tableau(standard) == StandardTableauCheckResult(
        tableau=standard, is_member=False
    )
    assert native.check_semistandard_tableau(
        semistandard
    ) == SemistandardTableauCheckResult(tableau=semistandard, is_member=False)
