from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    AffineFactorization,
    PositiveAffineSemigroup,
    evaluate_factorization,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS


def _semigroup() -> PositiveAffineSemigroup:
    return PositiveAffineSemigroup(
        configuration=AffineConfiguration(
            row_labels=("x", "y"),
            generator_labels=("a", "b", "c"),
            entries=((1, -1, 0), (-1, 2, 1)),
        ),
        grading=(
            CanonicalRational(num=3, den=1),
            CanonicalRational(num=2, den=1),
        ),
    )


def test_factorization_operation_returns_exact_parent_bound_element() -> None:
    semigroup = _semigroup()
    result = evaluate_factorization(semigroup, (2, 3, 4))

    # Independent hand-computed column combination:
    # 2(1,-1) + 3(-1,2) + 4(0,1) = (-1,8).
    assert result == AffineFactorization(
        semigroup=semigroup,
        coordinates=(2, 3, 4),
        target=(-1, 8),
    )
    restored = AffineFactorization.model_validate_json(result.model_dump_json())
    assert restored == result


def test_factorization_rejects_wrong_axis_negative_or_oversized_coefficients() -> None:
    semigroup = _semigroup()
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.factorization.evaluate"
    )
    with pytest.raises(OperationDomainValidationError):
        tool.run(tool.request_type(semigroup=semigroup, coordinates=(1, 2)))
    with pytest.raises(OperationDomainValidationError):
        tool.run(tool.request_type(semigroup=semigroup, coordinates=(1, -1, 0)))
    with pytest.raises(OperationResourceAdmissionError):
        tool.run(
            tool.request_type(
                semigroup=semigroup,
                coordinates=(10**32, 0, 0),
            )
        )


def test_factorization_accepts_maximum_coordinate_digits() -> None:
    semigroup = _semigroup()
    value = 10**32 - 1
    result = evaluate_factorization(semigroup, (value, 0, 0))
    assert result.target == (value, -value)


def test_native_factorization_admission_rejects_forged_or_oversized_values() -> None:
    semigroup = _semigroup()
    with pytest.raises(OperationResourceAdmissionError):
        evaluate_factorization(semigroup, (10**32, 0, 0))

    forged = PositiveAffineSemigroup.model_construct(
        configuration=semigroup.configuration,
        grading=(CanonicalRational(num=1, den=1), CanonicalRational(num=1, den=1)),
    )
    with pytest.raises(OperationDomainValidationError):
        evaluate_factorization(forged, (1, 0, 0))


def test_public_tool_preserves_factorization_output_admission_error() -> None:
    semigroup = PositiveAffineSemigroup(
        configuration=AffineConfiguration(
            row_labels=("r" * 200_000,),
            generator_labels=("g",),
            entries=((1,),),
        ),
        grading=(CanonicalRational(num=1, den=1),),
    )
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.factorization.evaluate"
    )
    request = tool.request_type(semigroup=semigroup, coordinates=(0,))
    with pytest.raises(OperationResourceAdmissionError):
        tool.run(request)


def test_deserialized_factorization_checks_its_authored_relation() -> None:
    semigroup = _semigroup()
    result = evaluate_factorization(semigroup, (2, 3, 4)).model_dump(mode="json")
    result["target"] = [0, 0]
    with pytest.raises(ValidationError):
        AffineFactorization.model_validate(result)


def test_public_operation_example_dispatches_and_round_trips() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.factorization.evaluate"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    decoded = tool.result_type.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.target == (2, 3)
