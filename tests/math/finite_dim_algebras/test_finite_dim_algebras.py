"""Tests for finite-dimensional algebra operations."""

from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    RadicalRequest,
    StructureConstants,
)
from jacobian.math.finite_dim_algebras._operations import (
    compute_center,
    compute_radical,
)
from jacobian.math.finite_dim_algebras._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "algebra.center.compute",
        "algebra.radical.compute",
    }


def test_center_of_zero_algebra() -> None:
    request = CenterRequest(
        algebra=StructureConstants(
            dimension=2, field_order=2, multiplication=((0, 0), (0, 0))
        )
    )
    result = compute_center(request)
    assert result.center_dimension > 0


def test_radical_basic() -> None:
    request = RadicalRequest(
        algebra=StructureConstants(
            dimension=2, field_order=2, multiplication=((0, 0), (0, 0))
        )
    )
    result = compute_radical(request)
    assert result.is_semisimple is True
    assert result.dimension == 0
