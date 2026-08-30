"""Exact homogeneous monomial systems on complex algebraic tori."""

from __future__ import annotations

import json
from math import prod

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.algebraic_tori import (
    AlgebraicTorusSolutionSubgroup,
    HomogeneousMonomialSystem,
    TorsionCharacterGroup,
    homogeneous_monomial_solution_subgroup,
)
from jacobian.math.algebraic_tori._models import (
    HomogeneousMonomialSolutionRequest,
)
from jacobian.math.algebraic_tori._tools import TOOLS
from jacobian.math.matrices.certified_snf import (
    CertifiedIntegerMatrix,
    verify_smith_normal_form_certificate,
)


def _matrix(
    entries: list[list[int | str]],
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> CertifiedIntegerMatrix:
    row_count = len(entries) if rows is None else rows
    column_count = (len(entries[0]) if entries else 0) if columns is None else columns
    return CertifiedIntegerMatrix(
        row_count=row_count,
        column_count=column_count,
        entries=tuple(tuple(str(value) for value in row) for row in entries),
    )


def _system(
    entries: list[list[int | str]],
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> HomogeneousMonomialSystem:
    matrix = _matrix(entries, rows=rows, columns=columns)
    return HomogeneousMonomialSystem(
        exponent_matrix=matrix,
        equation_axis=tuple(f"equation_{index}" for index in range(matrix.row_count)),
        coordinate_axis=tuple(f"x_{index}" for index in range(matrix.column_count)),
    )


def _integers(matrix: CertifiedIntegerMatrix) -> list[list[int]]:
    return [[parse_canonical_integer(value) for value in row] for row in matrix.entries]


def _multiply(
    left: list[list[int]], right: list[list[int]], columns: int
) -> list[list[int]]:
    return [
        [
            sum(left[row][index] * right[index][column] for index in range(len(right)))
            for column in range(columns)
        ]
        for row in range(len(left))
    ]


def _assert_defining_invariants(result: AlgebraicTorusSolutionSubgroup) -> None:
    certificate = result.smith_certificate
    assert verify_smith_normal_form_certificate(certificate)
    source = _integers(result.source.exponent_matrix)
    free = _integers(result.reduced_free_exponent_map)
    assert _multiply(source, free, result.free_rank) == [
        [0] * result.free_rank for _ in source
    ]
    factors = tuple(
        parse_canonical_integer(value)
        for value in result.torsion_character_group.invariant_factors
    )
    torsion = _integers(result.torsion_exponent_map)
    torsion_relations = _multiply(source, torsion, len(factors))
    assert all(
        value % factors[column] == 0
        for row in torsion_relations
        for column, value in enumerate(row)
    )
    assert parse_canonical_integer(result.connected_component_count) == prod(
        factors, start=1
    )


def test_nonsymmetric_smith_basis_uses_the_right_transformation_back_map() -> None:
    result = homogeneous_monomial_solution_subgroup(_system([[1, 1]]))

    assert result.smith_certificate.right_transformation.entries == (
        ("1", "-1"),
        ("0", "1"),
    )
    assert result.smith_free_exponent_map.entries == (("-1",), ("1",))
    _assert_defining_invariants(result)


def test_one_power_equation_has_d_zero_dimensional_components() -> None:
    result = homogeneous_monomial_solution_subgroup(_system([[7]]))

    assert result.smith_certificate.rank == 1
    assert result.free_rank == 0
    assert result.torsion_character_group.invariant_factors == ("7",)
    assert result.connected_component_count == "7"
    assert result.torsion_exponent_map.entries == (("1",),)
    _assert_defining_invariants(result)


def test_mixed_invariant_factors_and_free_coordinate_stay_compact() -> None:
    result = homogeneous_monomial_solution_subgroup(_system([[2, 0, 0], [0, 6, 0]]))

    assert result.torsion_character_group.invariant_factors == ("2", "6")
    assert result.connected_component_count == "12"
    assert result.free_rank == 1
    assert result.reduced_free_exponent_map.entries == (("0",), ("0",), ("1",))
    assert "components" not in result.model_dump()
    assert "exactness" not in result.model_dump()
    assert "completeness" not in result.model_dump()
    _assert_defining_invariants(result)


def test_lll_reduced_basis_retains_the_exact_parameter_transport() -> None:
    result = homogeneous_monomial_solution_subgroup(
        _system([[-5, 4, 4, 0], [-1, 4, 5, 6]])
    )

    assert result.smith_free_exponent_map.entries == (
        ("4", "-24"),
        ("21", "-120"),
        ("-16", "90"),
        ("0", "1"),
    )
    assert result.reduced_free_exponent_map.entries == (
        ("4", "0"),
        ("3", "6"),
        ("2", "-6"),
        ("-3", "1"),
    )
    assert result.smith_free_parameters_from_reduced.entries == (
        ("-17", "6"),
        ("-3", "1"),
    )
    smith_free = _integers(result.smith_free_exponent_map)
    parameter_transport = _integers(result.smith_free_parameters_from_reduced)
    assert _multiply(smith_free, parameter_transport, result.free_rank) == _integers(
        result.reduced_free_exponent_map
    )
    _assert_defining_invariants(result)


def test_redundant_and_sign_reversed_equations_preserve_the_group_profile() -> None:
    original = homogeneous_monomial_solution_subgroup(_system([[2, -2, 0]]))
    redundant = homogeneous_monomial_solution_subgroup(
        _system([[-2, 2, 0], [4, -4, 0]])
    )

    assert (
        original.smith_certificate.rank,
        original.free_rank,
        original.torsion_character_group,
    ) == (
        redundant.smith_certificate.rank,
        redundant.free_rank,
        redundant.torsion_character_group,
    )
    _assert_defining_invariants(redundant)


def test_empty_equation_and_coordinate_degeneracies_retain_ambient_axes() -> None:
    full_torus = homogeneous_monomial_solution_subgroup(_system([], rows=0, columns=3))
    singleton = homogeneous_monomial_solution_subgroup(
        _system([[], []], rows=2, columns=0)
    )

    assert (full_torus.smith_certificate.rank, full_torus.free_rank) == (0, 3)
    assert full_torus.reduced_free_exponent_map.entries == (
        ("1", "0", "0"),
        ("0", "1", "0"),
        ("0", "0", "1"),
    )
    assert (singleton.smith_certificate.rank, singleton.free_rank) == (0, 0)
    assert singleton.connected_component_count == "1"
    _assert_defining_invariants(full_torus)
    _assert_defining_invariants(singleton)


def test_large_component_family_is_compact_not_materialized() -> None:
    factor = "9" * 32
    result = homogeneous_monomial_solution_subgroup(_system([[factor]]))

    assert result.connected_component_count == factor
    assert len(json.dumps(result.model_dump(mode="json"))) < 10_000


def test_full_dimension_and_digit_boundary_returns_a_compact_exact_result() -> None:
    factor = "9" * 32
    system = _system(
        [
            [factor if row == column else "0" for column in range(16)]
            for row in range(16)
        ]
    )

    result = homogeneous_monomial_solution_subgroup(system)

    assert result.smith_certificate.rank == 16
    assert result.free_rank == 0
    assert len(result.torsion_character_group.invariant_factors) == 16
    assert result.connected_component_count == str(int(factor) ** 16)
    assert len(result.model_dump_json()) < 12_000


def test_result_round_trips_and_source_composes_unchanged() -> None:
    produced = homogeneous_monomial_solution_subgroup(_system([[2, 6, 0]]))
    decoded = AlgebraicTorusSolutionSubgroup.model_validate_json(
        produced.model_dump_json()
    )
    consumed = homogeneous_monomial_solution_subgroup(decoded.source)

    assert decoded == produced
    assert consumed == produced


@pytest.mark.parametrize("factors", [("1",), ("6", "2"), ("2", "3")])
def test_torsion_character_group_rejects_noncanonical_factors(
    factors: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="divisibility chain"):
        TorsionCharacterGroup(invariant_factors=factors)


def test_result_rejects_component_count_that_contradicts_torsion_group() -> None:
    encoded = homogeneous_monomial_solution_subgroup(
        _system([[2, 0], [0, 6]])
    ).model_dump(mode="json")
    encoded["connected_component_count"] = "11"

    with pytest.raises(ValidationError, match="product of torsion invariant factors"):
        AlgebraicTorusSolutionSubgroup.model_validate(encoded)


def test_result_rejects_a_certificate_bound_to_another_source() -> None:
    produced = homogeneous_monomial_solution_subgroup(_system([[2, 6]]))
    forged = produced.model_dump(mode="json")
    forged["source"]["exponent_matrix"]["entries"][0][0] = "3"

    with pytest.raises(ValidationError) as error:
        AlgebraicTorusSolutionSubgroup.model_validate(forged)

    assert error.value.errors()[0]["type"] == (
        "algebraic_torus.solution_source_binding"
    )


def test_raw_result_rejects_map_shape_before_nested_matrix_decoding() -> None:
    forged = homogeneous_monomial_solution_subgroup(_system([[2, 6, 0]])).model_dump(
        mode="json"
    )
    forged["reduced_free_exponent_map"]["row_count"] = 16

    with pytest.raises(ValidationError) as error:
        AlgebraicTorusSolutionSubgroup.model_validate(forged)

    assert error.value.errors()[0]["type"] == "algebraic_torus.solution_map_shape"


def test_system_bounds_nested_dimensions_and_digits_before_smith_work() -> None:
    with pytest.raises(ValidationError, match="at most 16"):
        HomogeneousMonomialSystem.model_validate(
            {
                "exponent_matrix": {
                    "row_count": 1,
                    "column_count": 17,
                    "entries": [["0"] * 17],
                },
                "equation_axis": ["e"],
                "coordinate_axis": [f"x{i}" for i in range(17)],
            }
        )
    with pytest.raises(ValidationError, match="32 decimal digits"):
        HomogeneousMonomialSystem.model_validate(
            {
                "exponent_matrix": {
                    "row_count": 1,
                    "column_count": 1,
                    "entries": [["9" * 33]],
                },
                "equation_axis": ["e"],
                "coordinate_axis": ["x"],
            }
        )


def test_public_operation_example_crosses_the_json_request_boundary() -> None:
    operation = TOOLS[0]
    request = HomogeneousMonomialSolutionRequest.model_validate(
        operation.examples[0].input
    )

    result = operation.run(request)

    assert isinstance(result, AlgebraicTorusSolutionSubgroup)
    assert result.connected_component_count == "2"
    assert result.free_rank == 1


def test_exact_public_api_symbols() -> None:
    from jacobian.math import algebraic_tori

    assert tuple(algebraic_tori.__all__) == (
        "AlgebraicTorusSolutionSubgroup",
        "HomogeneousMonomialSystem",
        "TorsionCharacterGroup",
        "homogeneous_monomial_solution_subgroup",
    )
