from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics import (
    bell_number,
    bernoulli_number,
    evaluate_linear_recurrence,
    integer_partitions,
    stirling_second,
)


def test_classical_numbers_remain_available_without_catalog_slots() -> None:
    assert bell_number(6) == 203
    assert bernoulli_number(4) == Fraction(-1, 30)
    assert stirling_second(5, 2) == 15
    assert integer_partitions(4, max_parts=2) == ((4,), (3, 1), (2, 2))


def test_native_classical_numbers_reject_noninteger_or_negative_indices() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        bell_number(-1)
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        bell_number(True)


def test_native_recurrence_admission_uses_typed_domain_errors() -> None:
    rational = CanonicalRational(num="1", den="1")

    with pytest.raises(OperationDomainValidationError, match="convention"):
        evaluate_linear_recurrence(
            (rational,),
            (rational,),
            "unsupported",  # type: ignore[arg-type]
            "PREFIX",
            term_count=1,
        )
