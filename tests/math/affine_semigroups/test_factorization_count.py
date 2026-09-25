from __future__ import annotations

import importlib
import itertools
from collections.abc import Callable
from fractions import Fraction
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups import (
    AffineConfiguration,
    AffineFactorizationCount,
    PositiveAffineSemigroup,
    factorization_count,
)
from jacobian.math.affine_semigroups.factorization_count import (
    AffineFactorizationCountRequest,
)


def _semigroup(
    vectors: tuple[tuple[int, ...], ...], grading: tuple[Fraction, ...]
) -> PositiveAffineSemigroup:
    rows = len(grading)
    configuration = AffineConfiguration(
        row_labels=tuple(f"r{row}" for row in range(rows)),
        generator_labels=tuple(f"g{column}" for column in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(rows)),
    )
    return PositiveAffineSemigroup(
        configuration=configuration,
        grading=tuple(
            CanonicalRational.from_fraction(component) for component in grading
        ),
    )


def _recursive_oracle(
    vectors: tuple[tuple[int, ...], ...],
    grading: tuple[Fraction, ...],
    target: tuple[int, ...],
) -> int:
    """Exhaust the exact grading-bounded coefficient box independently."""
    grades = tuple(
        sum(
            (grading[row] * vector[row] for row in range(len(grading))),
            Fraction(0),
        )
        for vector in vectors
    )
    target_grade = sum(
        (grading[row] * target[row] for row in range(len(grading))), Fraction(0)
    )
    if target_grade < 0:
        return 0
    maxima = tuple(int(target_grade // grade) for grade in grades)
    result = 0
    for coefficients in itertools.product(*(range(maximum + 1) for maximum in maxima)):
        if all(
            sum(
                coefficients[column] * vectors[column][row]
                for column in range(len(vectors))
            )
            == target[row]
            for row in range(len(grading))
        ):
            result += 1
    return result


def test_one_row_duplicate_columns_and_zero_target() -> None:
    semigroup = _semigroup(((1,), (1,)), (Fraction(1),))

    assert factorization_count(semigroup, (4,)).count == 5
    assert factorization_count(semigroup, (0,)).count == 1


def test_one_row_gcd_nonmember_and_negative_ray() -> None:
    positive = _semigroup(((6,), (10,)), (Fraction(1),))
    negative = _semigroup(((-2,), (-3,)), (Fraction(-1),))

    assert factorization_count(positive, (8,)).count == 0
    assert factorization_count(positive, (-1,)).count == 0
    assert factorization_count(negative, (-12,)).count == 3
    assert factorization_count(negative, (12,)).count == 0


def test_general_positive_grading_counts_signed_multivariate_generators() -> None:
    vectors = ((1, 0), (-1, 1), (0, 1))
    grading = (Fraction(1), Fraction(2))
    semigroup = _semigroup(vectors, grading)

    assert factorization_count(semigroup, (0, 2)).count == 3
    assert factorization_count(semigroup, (0, -1)).count == 0
    assert factorization_count(semigroup, (0, 2)).count == _recursive_oracle(
        vectors, grading, (0, 2)
    )


def test_univariate_dynamic_program_matches_exhaustive_small_oracle() -> None:
    for count in (1, 2, 3):
        for weights in itertools.combinations_with_replacement(range(1, 5), count):
            vectors = tuple((weight,) for weight in weights)
            semigroup = _semigroup(vectors, (Fraction(1),))
            for target in range(13):
                expected = _recursive_oracle(vectors, (Fraction(1),), (target,))
                assert factorization_count(semigroup, (target,)).count == expected


def test_univariate_dp_admits_a_large_fiber_without_materializing_it() -> None:
    semigroup = _semigroup(((1,), (1,)), (Fraction(1),))

    result = factorization_count(semigroup, (100_000,))

    assert result.count == 100_001


def test_zero_generator_is_rejected_by_positive_semigroup_carrier() -> None:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("zero", "unit"),
        entries=((0, 1), (0, 0)),
    )
    with pytest.raises(ValidationError, match="strictly positive grading"):
        PositiveAffineSemigroup(
            configuration=configuration,
            grading=(
                CanonicalRational.from_fraction(Fraction(1)),
                CanonicalRational.from_fraction(Fraction(1)),
            ),
        )


def test_state_preflight_runs_before_univariate_array_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = importlib.import_module(
        "jacobian.math.affine_semigroups.factorization_count"
    )
    monkeypatch.setattr(operation, "MAX_AFFINE_FACTOR_COUNT_STATES", 10)
    monkeypatch.setattr(
        operation,
        "_univariate_coin_change",
        lambda *_args: pytest.fail("dynamic program allocated before state admission"),
    )
    semigroup = _semigroup(((1,), (2,)), (Fraction(1),))

    with pytest.raises(OperationResourceAdmissionError, match="requires 21 states"):
        factorization_count(semigroup, (20,))


def test_count_digit_preflight_runs_before_univariate_array_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = importlib.import_module(
        "jacobian.math.affine_semigroups.factorization_count"
    )
    monkeypatch.setattr(operation, "MAX_AFFINE_FACTOR_COUNT_DIGITS", 1)
    monkeypatch.setattr(
        operation,
        "_univariate_coin_change",
        lambda *_args: pytest.fail("dynamic program ran before count-height admission"),
    )
    semigroup = _semigroup(((1,), (1,)), (Fraction(1),))

    with pytest.raises(OperationResourceAdmissionError, match="result envelope"):
        factorization_count(semigroup, (4,))


def test_general_count_preflight_runs_before_coefficient_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = importlib.import_module(
        "jacobian.math.affine_semigroups.factorization_count"
    )
    semigroup_module = importlib.import_module(
        "jacobian.math.affine_semigroups.semigroup"
    )
    monkeypatch.setattr(semigroup_module, "MAX_AFFINE_FIBER_WORK", 0)
    monkeypatch.setattr(
        operation,
        "_count_general_fiber",
        lambda *_args: pytest.fail("coefficient search started before admission"),
    )
    semigroup = _semigroup(((1, 0), (0, 1)), (Fraction(1), Fraction(1)))

    with pytest.raises(OperationResourceAdmissionError, match="state envelope"):
        factorization_count(semigroup, (1, 1))


def test_forged_typed_source_is_shaped_before_revalidation_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = importlib.import_module(
        "jacobian.math.affine_semigroups.factorization_count"
    )
    malformed_configuration = AffineConfiguration.model_construct(
        row_labels=("r",),
        generator_labels=tuple(f"g{i}" for i in range(11)),
        entries=(tuple(1 for _ in range(11)),),
    )
    forged = PositiveAffineSemigroup.model_construct(
        configuration=malformed_configuration,
        grading=(CanonicalRational.from_fraction(Fraction(1)),),
    )
    monkeypatch.setattr(
        operation,
        "_admit_semigroup",
        lambda *_args: pytest.fail("forged axes reached model revalidation"),
    )

    with pytest.raises(OperationDomainValidationError, match="axes exceed"):
        factorization_count(forged, (1,))


def test_public_request_and_catalog_example_round_trip() -> None:
    from jacobian.math.affine_semigroups.semigroup_tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.factorization_count.compute"
    )
    request = AffineFactorizationCountRequest.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    run = cast(
        Callable[[AffineFactorizationCountRequest], AffineFactorizationCount],
        tool.run,
    )
    result = run(request)

    assert result.count == 5
    assert (
        AffineFactorizationCount.model_validate_json(result.model_dump_json()) == result
    )
