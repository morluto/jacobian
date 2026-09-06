"""Canonical decimal coordinates remain exact across public operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.groups.abelian import (
    FiniteAbelianProductGroup,
    element_order,
    reduce_element,
)
from jacobian.math.groups.abelian._models import ElementReduceRequest
from jacobian.math.groups.abelian._tools import TOOLS
from jacobian.math.groups.abelian.values import MAX_GROUP_INTEGER_DIGITS

pytestmark = pytest.mark.requires_backend("flint")


def test_group_moduli_are_native_integers_with_canonical_json_encoding() -> None:
    group = FiniteAbelianProductGroup(moduli=(97,))
    assert group.model_dump() == {"moduli": (97,)}
    assert group.model_dump(mode="json") == {"moduli": ["97"]}
    encoded = encode_strict_json(group.model_dump(mode="json"))
    restored = FiniteAbelianProductGroup.model_validate_json(encoded)
    assert restored.moduli == (97,)
    assert element_order(reduce_element(restored, (1,))).order == 97


@pytest.mark.parametrize("exponent", [200, 6000])
def test_large_signed_coordinate_reduces_through_public_contract(exponent: int) -> None:
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "abelian_group.element.reduce"
    )
    coordinate = -(10**exponent + 7)
    request = operation.request_type.model_validate_json(
        encode_strict_json(
            {
                "group": {"moduli": ["97"]},
                "coordinates": [format_canonical_integer(coordinate)],
            }
        )
    )
    result = operation.run(request)
    restored = operation.result_type.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert restored.coordinates == (coordinate % 97,)
    assert restored == reduce_element(
        FiniteAbelianProductGroup(moduli=(97,)), (coordinate,)
    )
    assert element_order(restored).order == 97


@pytest.mark.parametrize("malformed", ["01", "-0", "+1", "1e3", " 1", 1, True])
def test_coordinate_wire_encoding_is_strict(malformed: object) -> None:
    with pytest.raises(ValidationError):
        ElementReduceRequest.model_validate_json(
            encode_strict_json(
                {"group": {"moduli": ["97"]}, "coordinates": [malformed]}
            )
        )


def test_decimal_digit_guard_precedes_coordinate_arithmetic() -> None:
    group = FiniteAbelianProductGroup(moduli=(97,))
    for sign in ("", "-"):
        with pytest.raises(ValidationError):
            ElementReduceRequest.model_validate_json(
                encode_strict_json(
                    {
                        "group": group.model_dump(mode="json"),
                        "coordinates": [sign + "1" * (MAX_GROUP_INTEGER_DIGITS + 1)],
                    }
                )
            )
    with pytest.raises(ValueError, match="bound"):
        reduce_element(group, (10**MAX_GROUP_INTEGER_DIGITS,))
