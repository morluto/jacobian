"""Rank admission is performed once on authored lattice bases."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices import IntegerLattice, compute_saturation
from jacobian.math.matrices.values import IntegerMatrix


@pytest.mark.parametrize("dependent", [False, True])
def test_saturation_checks_authored_rank_once(
    dependent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = ((2, 2), (4, 4)) if dependent else ((2, 2),)
    from jacobian.math.lattices import operations

    calls = 0
    original = getattr(operations, "integer_rank")

    def observe(matrix: list[list[int]]) -> int:
        nonlocal calls
        calls += 1
        return int(original(matrix))

    monkeypatch.setattr(operations, "integer_rank", observe)
    lattice = IntegerLattice(ambient_dimension=2, basis=IntegerMatrix(entries=entries))
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
