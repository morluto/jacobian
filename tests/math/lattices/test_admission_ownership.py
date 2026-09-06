"""Rank admission is performed once on authored lattice bases."""

import sys
from types import FrameType
from typing import Any

import pytest
from sympy.matrices.matrixbase import MatrixBase

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices import IntegerLattice, compute_saturation
from jacobian.math.matrices.values import IntegerMatrix


@pytest.mark.parametrize("dependent", [False, True])
def test_saturation_checks_authored_rank_once(dependent: bool) -> None:
    entries = ((2, 2), (4, 4)) if dependent else ((2, 2),)
    calls = 0

    def observe(frame: FrameType, event: str, arg: Any) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is MatrixBase.rank.__code__:
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        lattice = IntegerLattice(
            ambient_dimension=2, basis=IntegerMatrix(entries=entries)
        )
        lattice = IntegerLattice.model_validate_json(lattice.model_dump_json())
        assert calls == 0
        if dependent:
            with pytest.raises(OperationDomainValidationError, match="full row rank"):
                compute_saturation(lattice)
        else:
            result = compute_saturation(lattice)
            assert result.saturated_basis.entries == ((1, 1),)
            assert result.inclusion_transform.entries == ((2,),)
            assert result.saturation_index == 2
            assert type(result).model_validate_json(result.model_dump_json()) == result
        assert calls == 1
    finally:
        sys.setprofile(previous)
