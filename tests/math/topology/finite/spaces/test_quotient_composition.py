"""Canonical Kolmogorov quotient values compose without rebuilding arrays."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.finite.spaces import (
    FiniteTopologicalSpace,
    closure,
    continuous_check,
    kolmogorov_quotient,
    verify_kolmogorov_quotient,
)


def test_quotient_composes() -> None:
    space = FiniteTopologicalSpace(
        points=("a", "b", "c"), preorder=((0, 1), (0, 1), (0, 1, 2))
    )
    result = kolmogorov_quotient(space)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_kolmogorov_quotient(decoded)
    assert decoded.quotient_map.source == space
    assert closure(decoded.quotient_map.target, frozenset({1})) == frozenset({0, 1})
    assert continuous_check(decoded.quotient_map)
    payload = result.model_dump(mode="json")
    payload["quotient_map"]["point_map"][0] = 99
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)
    payload["quotient_map"]["point_map"][0] = 1
    assert not verify_kolmogorov_quotient(type(result).model_validate(payload))


def test_authored_preorder_checked_on_consumption() -> None:
    claim = FiniteTopologicalSpace(
        points=("a", "b", "c"), preorder=((0,), (0, 1), (1, 2))
    )
    decoded = FiniteTopologicalSpace.model_validate_json(claim.model_dump_json())
    with pytest.raises(OperationDomainValidationError, match="transitive"):
        kolmogorov_quotient(decoded)
