"""Counting-family model contracts."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._counting_models import (
    IntegerListRequest,
    SparseCountingPairRequest,
)
from jacobian.math.combinatorics._counting_tools import (
    binomial as compute_binomial,
)
from jacobian.math.combinatorics._counting_tools import (
    compositions as compute_compositions,
)
from jacobian.math.combinatorics._counting_tools import (
    permutations as compute_permutations,
)
from jacobian.math.combinatorics._models import IntegerResult
from jacobian.math.combinatorics.operations import (
    MAX_COUNTING_MULTIPLICATIVE_STEPS,
    MAX_SPARSE_COUNTING_INDEX,
    _admit_multiplicative_count,
    _binomial_coefficient_digit_bound,
    _falling_factorial_digit_bound,
)


def test_sparse_counting_request_publishes_the_wider_exact_bound() -> None:
    schema = SparseCountingPairRequest.model_json_schema()

    assert schema["properties"]["n"]["maximum"] == MAX_SPARSE_COUNTING_INDEX
    assert SparseCountingPairRequest(n=MAX_SPARSE_COUNTING_INDEX, k=0).model_dump() == {
        "n": MAX_SPARSE_COUNTING_INDEX,
        "k": 0,
    }

    with pytest.raises(ValidationError):
        SparseCountingPairRequest(n=MAX_SPARSE_COUNTING_INDEX + 1, k=0)


@pytest.mark.parametrize(
    ("operation", "k", "expected"),
    (
        (compute_binomial, 0, "1"),
        (compute_binomial, 1, "1000000000000"),
        (compute_binomial, 2, "499999999999500000000000"),
        (compute_permutations, 0, "1"),
        (compute_permutations, 1, "1000000000000"),
        (compute_permutations, 2, "999999999999000000000000"),
        (compute_compositions, 1, "1"),
        (compute_compositions, 2, "999999999999"),
    ),
)
def test_sparse_large_counts_are_admitted(
    operation: Callable[[SparseCountingPairRequest], IntegerResult],
    k: int,
    expected: str,
) -> None:
    request = SparseCountingPairRequest(n=10**12, k=k)
    assert operation(request).value == expected


@pytest.mark.parametrize(
    "operation", (compute_binomial, compute_permutations, compute_compositions)
)
def test_large_zero_cases_skip_coefficient_construction(
    operation: Callable[[SparseCountingPairRequest], IntegerResult],
) -> None:
    request = SparseCountingPairRequest(n=10**12, k=10**12 + 1)
    assert operation(request).value == "0"


def test_composition_zero_endpoint_conventions_are_preserved() -> None:
    assert compute_compositions(SparseCountingPairRequest(n=0, k=0)).value == "1"
    assert compute_compositions(SparseCountingPairRequest(n=10**12, k=0)).value == "0"


@pytest.mark.parametrize(
    "operation", (compute_binomial, compute_permutations, compute_compositions)
)
def test_long_multiplicative_profiles_reject_before_kernel(
    operation: Callable[[SparseCountingPairRequest], IntegerResult],
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        operation(
            SparseCountingPairRequest(
                n=10**12,
                k=MAX_COUNTING_MULTIPLICATIVE_STEPS + 2,
            )
        )
    assert error.value.errors()[0]["type"] == "combinatorics.counting_work_exceeded"


def test_central_binomial_admission_accounts_for_cancellation() -> None:
    result = compute_binomial(SparseCountingPairRequest(n=100_000, k=50_000))
    assert len(result.value) == 30_101


def test_off_center_binomial_admission_accounts_for_cancellation() -> None:
    result = compute_binomial(SparseCountingPairRequest(n=1_000_000, k=80_000))
    assert len(result.value) == 121_066


@pytest.mark.parametrize("operation", (compute_binomial, compute_compositions))
def test_oversized_binomial_steps_reject_before_digit_bound_iteration(
    operation: Callable[[SparseCountingPairRequest], IntegerResult],
) -> None:
    request = SparseCountingPairRequest(
        n=MAX_SPARSE_COUNTING_INDEX,
        k=MAX_SPARSE_COUNTING_INDEX // 2,
    )
    started = time.perf_counter()
    with pytest.raises(OperationDomainValidationError) as error:
        operation(request)
    elapsed = time.perf_counter() - started

    assert error.value.errors()[0]["type"] == "combinatorics.counting_work_exceeded"
    assert elapsed < 1.0


def test_long_permutation_uses_the_actual_string_transport_envelope() -> None:
    result = compute_permutations(SparseCountingPairRequest(n=20_000, k=20_000))
    assert len(result.value) == 77_338


def test_decimal_result_construction_is_included_in_work_admission() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        compute_permutations(
            SparseCountingPairRequest(n=MAX_SPARSE_COUNTING_INDEX, k=100_000)
        )

    assert error.value.errors()[0]["type"] == "combinatorics.counting_work_exceeded"


def test_binomial_digit_bound_rejects_underestimated_lgamma_envelope() -> None:
    n, k = 663_856_028_889_952, 45_932
    bound = _binomial_coefficient_digit_bound(n, k)
    assert bound >= 486_613
    with pytest.raises(OperationDomainValidationError) as error:
        compute_binomial(SparseCountingPairRequest(n=n, k=k))
    assert error.value.errors()[0]["type"] == "combinatorics.counting_work_exceeded"


def test_permutation_falling_product_admission_fits_the_stated_ledger() -> None:
    n, k = 1_048_576, 58_741
    digits = _falling_factorial_digit_bound(n, k)
    formatting_steps = (digits + 8) // 9
    assert k + formatting_steps <= MAX_COUNTING_MULTIPLICATIVE_STEPS
    _admit_multiplicative_count(
        maximum_factor=n,
        steps=k,
        result_digit_bound=digits,
    )


def test_medium_counts_retain_defining_identities() -> None:
    n, k = 200, 37
    assert compute_binomial(SparseCountingPairRequest(n=n, k=k)).value == str(
        math.comb(n, n - k)
    )
    assert compute_permutations(SparseCountingPairRequest(n=n, k=k)).value == str(
        math.factorial(n) // math.factorial(n - k)
    )


def test_integer_list_accepts_canonical_values_beyond_python_limit() -> None:
    value = "1" + ("0" * 5_000)

    request = IntegerListRequest(values=(value,))

    assert request.values == (value,)


def test_integer_list_retains_the_nonnegative_parts_contract() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IntegerListRequest(values=("-1",))

    assert exc_info.value.errors()[0]["type"] == "combinatorics.invariant"
