import json

import pytest
from pydantic import ValidationError
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form as sympy_column_hnf

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.affine_semigroups.group_lattice import (
    AffineGroupLattice,
    compute_group_lattice,
)
from jacobian.math.affine_semigroups.group_lattice_models import (
    AffineGroupLatticeRequest,
)
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration


def _configuration(entries: tuple[tuple[int, ...], ...]) -> AffineConfiguration:
    return AffineConfiguration(
        row_labels=tuple(f"r{i}" for i in range(len(entries))),
        generator_labels=tuple(f"g{i}" for i in range(len(entries[0]))),
        entries=entries,
    )


def _sympy_image_hnf(configuration: AffineConfiguration) -> tuple[tuple[int, ...], ...]:
    matrix = Matrix([[int(value) for value in row] for row in configuration.entries])
    column_basis = sympy_column_hnf(matrix)
    return tuple(
        tuple(int(column_basis[row, column]) for row in range(column_basis.rows))
        for column in range(column_basis.cols)
    )


def test_group_lattice_matches_independent_column_hnf_oracle() -> None:
    for entries in (
        ((2, 0, 2), (0, 2, 2)),
        ((2, 4, 6), (0, 0, 0)),
        ((2, 1, 3), (4, 2, 6)),
        ((0, 0), (0, 0)),
        ((1, 0, 2), (0, 1, 3), (0, 0, 0)),
    ):
        config = _configuration(entries)
        result = compute_group_lattice(config)
        actual = tuple(
            tuple(int(value) for value in row) for row in result.lattice.basis.entries
        )
        assert actual == _sympy_image_hnf(config)
        assert result.configuration == config
        assert result.lattice.ambient_dimension == config.rows


def test_group_lattice_carrier_rejects_wrong_ambient_dimension() -> None:
    config = _configuration(((1,), (0,)))
    result = compute_group_lattice(config)
    payload = result.model_dump(mode="python")
    payload["lattice"]["ambient_dimension"] = 1
    with pytest.raises(ValidationError):
        AffineGroupLattice.model_validate(payload)


def test_group_lattice_carrier_checks_both_integer_inclusions() -> None:
    result = compute_group_lattice(_configuration(((2, 0), (0, 2))))
    payload = result.model_dump(mode="python")
    payload["generator_lattice_coordinates"]["entries"] = ((2, 0), (0, 1))
    with pytest.raises(ValidationError):
        AffineGroupLattice.model_validate(payload)


def test_group_lattice_catalog_example_executes() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "affine_semigroup.group_lattice.compute"
    )
    request = AffineGroupLatticeRequest.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.lattice.basis.entries == ((2, 0), (0, 2))
