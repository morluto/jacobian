from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics import (
    bell_number,
    bernoulli_number,
    evaluate_linear_recurrence,
    integer_partitions,
    stirling_second,
)
from jacobian.math.combinatorics.operations import MAX_COUNTING_INDEX


def _bell_by_stirling_recurrence(n: int) -> int:
    """Count set partitions by their number of blocks, without the Bell kernel."""

    stirling_row = [1]
    for size in range(1, n + 1):
        next_row = [0] * (size + 1)
        for blocks in range(1, size + 1):
            next_row[blocks] = (
                blocks * stirling_row[blocks] if blocks < len(stirling_row) else 0
            ) + stirling_row[blocks - 1]
        stirling_row = next_row
    return sum(stirling_row)


def _bell_modulo_by_touchard_recurrence(limit: int, prime: int) -> list[int]:
    """Compute Bell residues from small Stirling rows and Touchard's congruence."""

    residues = [_bell_by_stirling_recurrence(n) % prime for n in range(prime)]
    for n in range(prime, limit + 1):
        residues.append((residues[n - prime] + residues[n - prime + 1]) % prime)
    return residues


def test_classical_numbers_remain_available_without_catalog_slots() -> None:
    assert bell_number(6) == 203
    assert bernoulli_number(4) == Fraction(-1, 30)
    assert stirling_second(5, 2) == 15
    assert integer_partitions(4, max_parts=2) == ((4,), (3, 1), (2, 2))


def test_bell_numbers_match_the_independent_stirling_recurrence() -> None:
    assert [bell_number(n) for n in range(20)] == [
        _bell_by_stirling_recurrence(n) for n in range(20)
    ]


def test_large_bell_numbers_match_touchard_residues() -> None:
    prime = 13
    n = MAX_COUNTING_INDEX - prime
    residues = _bell_modulo_by_touchard_recurrence(MAX_COUNTING_INDEX, prime)

    bell_n = bell_number(n)
    bell_n_plus_one = bell_number(n + 1)
    bell_n_plus_prime = bell_number(n + prime)

    assert bell_n % prime == residues[n]
    assert bell_n_plus_one % prime == residues[n + 1]
    assert bell_n_plus_prime % prime == residues[n + prime]
    assert bell_n_plus_prime % prime == (bell_n + bell_n_plus_one) % prime


def test_largest_bell_number_projects_as_a_canonical_public_json_integer() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("combinatorics.compute.bell")
    assert operation is not None

    result = invoke_operation(
        operation.operation_id, {"n": MAX_COUNTING_INDEX}, catalog
    )

    assert isinstance(result.output["value"], str)
    assert len(result.output["value"]) == 27_665
    assert (
        operation.result_type.model_validate_json(json.dumps(result.output)).model_dump(
            mode="json"
        )
        == result.output
    )


def test_native_classical_numbers_reject_noninteger_or_negative_indices() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        bell_number(-1)
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        bell_number(True)


def test_bell_number_enforces_the_native_counting_index_bound() -> None:
    with pytest.raises(OperationDomainValidationError) as raised:
        bell_number(MAX_COUNTING_INDEX + 1)

    assert (
        raised.value.errors()[0]["type"] == "combinatorics.counting_index_out_of_range"
    )


def test_native_recurrence_admission_uses_typed_domain_errors() -> None:
    rational = CanonicalRational(num=1, den=1)

    with pytest.raises(OperationDomainValidationError, match="convention"):
        evaluate_linear_recurrence(
            (rational,),
            (rational,),
            "unsupported",  # type: ignore[arg-type]
            "PREFIX",
            term_count=1,
        )
