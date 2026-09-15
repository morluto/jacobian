"""Exact class and unit groups (#3717), backed by the PARI worker."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._models import (
    MAX_CLASS_GROUP_DEGREE,
    NumberFieldClassGroupRequest,
    NumberFieldUnitGroupRequest,
)
from jacobian.math.number_theory.number_fields.operations import (
    class_group,
    unit_group,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
)

pytestmark = pytest.mark.requires_backend("pari")


def _field(*coefficients: int) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(
        coefficients_descending=tuple(coefficients)
    )


def test_imaginary_quadratic_class_two() -> None:
    """QQ(sqrt(-5)) has class number 2 with cyclic structure C2."""
    result = class_group(_field(1, 0, 5))
    assert result.class_number == 2
    assert result.abelian_invariants == (2,)
    assert result.field_discriminant == -20
    assert (result.real_embedding_count, result.complex_embedding_pair_count) == (0, 1)
    assert len(result.ideal_representatives) == 1
    # The retained HNF basis vectors generate an ideal of norm two.
    basis = result.ideal_representatives[0].entries
    assert basis == ((2, 0), (1, 1))


def test_imaginary_quadratic_class_three() -> None:
    result = class_group(_field(1, 0, 23))
    assert result.class_number == 3
    assert result.abelian_invariants == (3,)


def test_real_quadratic_trivial_class_group() -> None:
    result = class_group(_field(1, 0, -2))
    assert result.class_number == 1
    assert result.abelian_invariants == ()
    assert result.ideal_representatives == ()
    assert result.field_discriminant == 8
    assert (result.real_embedding_count, result.complex_embedding_pair_count) == (2, 0)


def test_real_quadratic_fundamental_unit() -> None:
    """QQ(sqrt(2)) has rank 1 and fundamental unit up to sign 1 + sqrt(2)."""
    result = unit_group(_field(1, 0, -2))
    assert result.rank == 1
    assert result.torsion_order == 2
    assert len(result.fundamental_units) == 1
    coefficients = tuple(
        component.as_fraction()
        for component in result.fundamental_units[0].coefficients_ascending
    )
    assert coefficients in ((Fraction(1), Fraction(1)), (Fraction(-1), Fraction(-1)))


def test_imaginary_quadratic_unit_rank_zero() -> None:
    result = unit_group(_field(1, 0, 5))
    assert result.rank == 0
    assert result.torsion_order == 2


def test_cubic_signature_and_torsion() -> None:
    """x^3 - x - 1 has one real and one complex pair, with a single unit."""
    result = unit_group(_field(1, 0, -1, -1))
    assert (result.real_embedding_count, result.complex_embedding_pair_count) == (1, 1)
    assert result.rank == 1
    assert result.torsion_order == 2


def test_reducible_polynomial_rejected() -> None:
    """x^2 - 1 is reducible, so it does not define a field."""
    with pytest.raises(OperationDomainValidationError, match="irreducible"):
        class_group(_field(1, 0, -1))


def test_degree_bound_rejected_natively() -> None:
    coefficients = (1,) + (0,) * MAX_CLASS_GROUP_DEGREE + (1,)
    with pytest.raises(OperationDomainValidationError):
        class_group(_field(*coefficients))


def test_request_validators_gate_degree() -> None:
    from pydantic import ValidationError

    coefficients = (1,) + (0,) * MAX_CLASS_GROUP_DEGREE + (1,)
    with pytest.raises(ValidationError):
        NumberFieldClassGroupRequest(
            field=SimpleNumberFieldPresentation(
                coefficients_descending=coefficients
            )
        )


def test_serialized_results_round_trip() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.number_theory.number_fields._models import (
        NumberFieldClassGroupResult,
        NumberFieldUnitGroupResult,
    )

    class_result = class_group(_field(1, 0, 5))
    restored = NumberFieldClassGroupResult.model_validate_json(
        encode_strict_json(class_result.model_dump(mode="json")), strict=True
    )
    assert restored.class_number == 2

    unit_result = unit_group(_field(1, 0, -2))
    restored_unit = NumberFieldUnitGroupResult.model_validate_json(
        encode_strict_json(unit_result.model_dump(mode="json")), strict=True
    )
    assert restored_unit.rank == 1
    assert restored_unit.fundamental_units == unit_result.fundamental_units


def test_forged_rank_signature_rejected() -> None:
    from pydantic import ValidationError

    from jacobian.canonical import encode_strict_json
    from jacobian.math.number_theory.number_fields._models import (
        NumberFieldUnitGroupResult,
    )

    result = unit_group(_field(1, 0, -2))
    payload = result.model_dump(mode="json")
    payload["rank"] = 0
    payload["fundamental_units"] = []
    with pytest.raises(ValidationError):
        NumberFieldUnitGroupResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )


def test_unit_request_validates_field_type() -> None:
    with pytest.raises(OperationDomainValidationError, match="SimpleNumberFieldPresentation"):
        unit_group("not a field")


def test_class_group_request_admits_native_field() -> None:
    request = NumberFieldClassGroupRequest(field=_field(1, 0, 5))
    assert request.field.degree == 2
    unit_request = NumberFieldUnitGroupRequest(field=_field(1, 0, 5))
    assert unit_request.field.degree == 2
