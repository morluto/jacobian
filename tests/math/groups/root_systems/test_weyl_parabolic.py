"""Embedded standard parabolic Weyl subgroup profiles."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    WeylParabolicRequest,
    WeylParabolicResult,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import (
    weyl_group_order,
    weyl_parabolic,
)

A3 = ((2, -1, 0), (-1, 2, -1), (0, -1, 2))


@pytest.mark.parametrize(
    ("indices", "submatrix", "order"),
    (
        ((), (), 1),
        ((0,), ((2,),), 2),
        ((0, 1), ((2, -1), (-1, 2)), 6),
        ((0, 2), ((2, 0), (0, 2)), 4),
        ((0, 1, 2), A3, 24),
    ),
)
def test_parabolic_value_retains_parent_embedding(indices, submatrix, order):
    result = weyl_parabolic(A3, indices)
    assert result.matrix == CartanMatrix.model_validate(A3)
    assert result.simple_root_indices == indices
    assert result.parabolic_cartan_matrix == submatrix
    assert result.group_order == order
    assert WeylParabolicResult.model_validate_json(result.model_dump_json()) == result
    if indices:
        assert weyl_group_order(submatrix).group_order == result.group_order


def test_parabolic_request_rejects_noncanonical_or_out_of_range_indices():
    with pytest.raises(ValueError):
        WeylParabolicRequest(
            matrix=CartanMatrix.model_validate(A3), simple_root_indices=(1, 0)
        )
    with pytest.raises(ValueError):
        WeylParabolicRequest(
            matrix=CartanMatrix.model_validate(A3), simple_root_indices=(3,)
        )
    with pytest.raises(OperationDomainValidationError):
        weyl_parabolic(A3, (0, 0))


def test_public_parabolic_operation_uses_the_typed_request_and_result():
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "weyl_group.parabolic.compute"
    )
    result = tool.run(
        WeylParabolicRequest(
            matrix=CartanMatrix.model_validate(A3), simple_root_indices=(0, 1)
        )
    )
    assert isinstance(result, WeylParabolicResult)
    assert result.group_order == 6
