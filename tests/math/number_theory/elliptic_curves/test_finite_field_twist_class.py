from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldShortWeierstrassCurve,
    finite_field_quadratic_twist,
)
from jacobian.math.number_theory.elliptic_curves.twist_class.operations import (
    FiniteFieldTwistClassResult,
    finite_field_twist_class,
)


def _f5_curve(a: int, b: int) -> FiniteFieldShortWeierstrassCurve:
    field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    return FiniteFieldShortWeierstrassCurve(
        field=field,
        coefficient_a=FiniteFieldElement(presentation=field, coordinates=(a,)),
        coefficient_b=FiniteFieldElement(presentation=field, coordinates=(b,)),
    )


def _j(a: int, b: int) -> int:
    # j=1728*4a^3/(4a^3+27b^2) in F5, computed independently.
    denominator = (4 * a**3 + 27 * b**2) % 5
    return (1728 * (4 * a**3) * pow(denominator, -1, 5)) % 5


def _isomorphic_f5(source: tuple[int, int], target: tuple[int, int]) -> bool:
    a, b = source
    target_a, target_b = target
    return any(
        (pow(u, 4, 5) * a) % 5 == target_a and (pow(u, 6, 5) * b) % 5 == target_b
        for u in range(1, 5)
    )


def test_generic_j_twist_class_matches_independent_f5_oracle_and_roundtrips() -> None:
    source = _f5_curve(1, 1)
    twist = finite_field_quadratic_twist(source)
    target_coefficients = (
        twist.coefficient_a.coordinates[0],
        twist.coefficient_b.coordinates[0],
    )
    result = finite_field_twist_class(source, twist)

    assert result.relation == "QUADRATIC_TWIST"
    assert result.twist is not None
    assert result.twist.twisted_curve == twist
    assert result.twist_to_target is not None
    assert result.twist_to_target.isomorphic
    assert _j(1, 1) == _j(*target_coefficients)
    assert not _isomorphic_f5((1, 1), target_coefficients)
    assert (
        FiniteFieldTwistClassResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_twist_class_distinguishes_isomorphic_and_unequal_j_models() -> None:
    source = _f5_curve(1, 1)
    isomorphic = _f5_curve(1, 4)
    different_j = _f5_curve(2, 1)

    same_class = finite_field_twist_class(source, isomorphic)
    different = finite_field_twist_class(source, different_j)

    assert same_class.relation == "ISOMORPHIC"
    assert same_class.isomorphism is not None
    assert _isomorphic_f5((1, 1), (1, 4))
    assert different.relation == "DIFFERENT_J"
    assert (
        different.source_j_invariant.coordinates
        != different.target_j_invariant.coordinates
    )
    assert _j(1, 1) != _j(2, 1)


def test_twist_class_works_over_a_nonprime_field_against_coordinate_oracle() -> None:
    field = FiniteFieldPresentation(
        characteristic=5,
        modulus_coefficients=(2, 0, 1),
        generator="a",
    )

    def element(value: tuple[int, int]) -> FiniteFieldElement:
        return FiniteFieldElement(presentation=field, coordinates=value)

    source = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=element((1, 0)), coefficient_b=element((1, 0))
    )
    target = finite_field_quadratic_twist(source)

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        # In this presentation a^2 = -2 = 3 in F5.
        return (
            (left[0] * right[0] + 3 * left[1] * right[1]) % 5,
            (left[0] * right[1] + left[1] * right[0]) % 5,
        )

    def power(value: tuple[int, int], exponent: int) -> tuple[int, int]:
        result, base = (1, 0), value
        while exponent:
            if exponent & 1:
                result = multiply(result, base)
            base = multiply(base, base)
            exponent >>= 1
        return result

    source_a = tuple(source.coefficient_a.coordinates)
    source_b = tuple(source.coefficient_b.coordinates)
    target_a = tuple(target.coefficient_a.coordinates)
    target_b = tuple(target.coefficient_b.coordinates)
    oracle_isomorphic = any(
        multiply(power((u0, u1), 4), source_a) == target_a
        and multiply(power((u0, u1), 6), source_b) == target_b
        for u0 in range(5)
        for u1 in range(5)
        if (u0, u1) != (0, 0)
    )

    result = finite_field_twist_class(source, target)
    assert not oracle_isomorphic
    assert result.relation == "QUADRATIC_TWIST"
    assert result.twist_to_target is not None
    assert result.twist_to_target.scaling is not None
    assert any(result.twist_to_target.scaling.coordinates)


def test_twist_class_roundtrip_rejects_a_forged_square_twist_parameter() -> None:
    source = _f5_curve(1, 1)
    target = finite_field_quadratic_twist(source)
    result = finite_field_twist_class(source, target)
    assert result.twist is not None
    payload = result.model_dump(mode="python")
    payload["twist"]["parameter"] = FiniteFieldElement(
        presentation=source.field, coordinates=(1,)
    )

    with pytest.raises(ValidationError, match="do not define the claimed twist"):
        FiniteFieldTwistClassResult.model_validate(payload)


def test_twist_class_result_rejects_forged_j_invariant() -> None:
    source = _f5_curve(1, 1)
    result = finite_field_twist_class(source, source)
    payload = result.model_dump(mode="python")
    payload["source_j_invariant"] = FiniteFieldElement(
        presentation=source.field, coordinates=(0,)
    )

    with pytest.raises(ValidationError, match="exact j values of the curves"):
        FiniteFieldTwistClassResult.model_validate(payload)


@pytest.mark.parametrize("exceptional", [_f5_curve(0, 1), _f5_curve(1, 0)])
def test_twist_class_rejects_automorphism_rich_j_values(exceptional) -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        finite_field_twist_class(exceptional, exceptional)
    assert exc_info.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.twist_class_exceptional_j"
    )


def test_twist_class_rejects_field_above_complete_search_bound() -> None:
    field = FiniteFieldPresentation(
        characteristic=5003, modulus_coefficients=(0, 1), generator="a"
    )
    one = FiniteFieldElement(presentation=field, coordinates=(1,))
    curve = FiniteFieldShortWeierstrassCurve(
        field=field, coefficient_a=one, coefficient_b=one
    )

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        finite_field_twist_class(curve, curve)
    assert exc_info.value.errors()[0]["type"] == (
        "elliptic_curve.finite_field.twist_class_order_bound"
    )
