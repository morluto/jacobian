"""Contract tests for finite-order crystallographic mapping tori."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.crystallographic._models import (
    CrystallographicMappingTorusRequest,
)
from jacobian.math.geometry.crystallographic.operations import (
    mapping_torus_chain_complex,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.chain_complexes.operations import homology_groups
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    IntegralHomologyGroupValue,
)


def _matrix(rows: tuple[tuple[int, ...], ...]) -> IntegerMatrix:
    return IntegerMatrix(entries=rows)


def _integral_groups(
    complex_value: ChainComplexValue,
) -> tuple[IntegralHomologyGroupValue, ...]:
    groups = homology_groups(complex_value).homology_groups
    assert all(isinstance(group, IntegralHomologyGroupValue) for group in groups)
    return tuple(
        group for group in groups if isinstance(group, IntegralHomologyGroupValue)
    )


def test_point_mapping_torus_is_the_circle_complex() -> None:
    result = mapping_torus_chain_complex(
        IntegerMatrix(row_count=0, column_count=0, entries=()),
        1,
    )

    assert result.basis_sizes == (1, 1)
    assert result.differential_matrices == (((0,),),)
    assert tuple(group.free_rank for group in _integral_groups(result)) == (1, 1)


def test_identity_circle_mapping_torus_is_the_two_torus() -> None:
    result = mapping_torus_chain_complex(_matrix(((1,),)), 1)

    assert result.basis_sizes == (1, 2, 1)
    assert result.differential_matrices == (
        ((0, 0),),
        ((0,), (0,)),
    )
    groups = _integral_groups(result)
    assert tuple(group.free_rank for group in groups) == (1, 2, 1)
    assert all(not group.torsion_invariant_factors for group in groups)


def test_reflection_mapping_torus_has_klein_bottle_homology() -> None:
    result = mapping_torus_chain_complex(_matrix(((-1,),)), 2)

    assert result.basis_sizes == (1, 2, 1)
    assert result.differential_matrices[0] == ((0, 0),)
    assert result.differential_matrices[1] == ((0,), (-2,))
    degree_zero, degree_one, degree_two = _integral_groups(result)
    assert (degree_zero.free_rank, degree_zero.torsion_invariant_factors) == (
        1,
        (),
    )
    assert (degree_one.free_rank, degree_one.torsion_invariant_factors) == (
        1,
        (2,),
    )
    assert (degree_two.free_rank, degree_two.torsion_invariant_factors) == (
        0,
        (),
    )


def test_quarter_turn_mapping_torus_uses_exterior_power_differentials() -> None:
    result = mapping_torus_chain_complex(_matrix(((0, -1), (1, 0))), 4)

    assert result.basis_sizes == (1, 3, 3, 1)
    # Λ^0 A - I and Λ^2 A - I vanish; the middle block is A-I.
    assert result.differential_matrices[0] == ((0, 0, 0),)
    assert result.differential_matrices[2] == ((0,), (0,), (0,))
    assert result.differential_matrices[1] == (
        (0, 0, 0),
        (-1, -1, 0),
        (1, -1, 0),
    )


def test_false_finite_order_claim_is_a_domain_error() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        mapping_torus_chain_complex(_matrix(((2,),)), 3)

    assert error.value.errors()[0]["type"] == (
        "crystallographic.mapping_torus.finite_order_relation"
    )


def test_native_boundary_rejects_wrong_and_forged_carriers() -> None:
    with pytest.raises(OperationDomainValidationError):
        mapping_torus_chain_complex("not-a-matrix", 1)  # type: ignore[arg-type]
    forged = IntegerMatrix.model_construct(
        domain="ZZ",
        row_count=1,
        column_count=1,
        entries=((object(),),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        mapping_torus_chain_complex(forged, 1)
    assert error.value.errors()[0]["type"] == (
        "crystallographic.mapping_torus.matrix_shape"
    )
    with pytest.raises(OperationDomainValidationError):
        mapping_torus_chain_complex(_matrix(((1,),)), True)


def test_full_rank_and_exponent_boundary_remain_accepted() -> None:
    rank = 6
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(rank))
        for row in range(rank)
    )

    result = mapping_torus_chain_complex(_matrix(identity), 64)

    assert result.basis_sizes == (1, 7, 21, 35, 35, 21, 7, 1)


def test_rank_and_height_are_admitted_before_exact_powering() -> None:
    rank = 7
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(rank))
        for row in range(rank)
    )
    with pytest.raises(OperationResourceAdmissionError) as rank_error:
        mapping_torus_chain_complex(_matrix(identity), 1)
    assert rank_error.value.errors()[0]["type"] == (
        "crystallographic.mapping_torus.rank_bound"
    )

    with pytest.raises(OperationResourceAdmissionError) as height_error:
        mapping_torus_chain_complex(_matrix(((10**32,),)), 1)
    assert height_error.value.errors()[0]["type"] == (
        "crystallographic.mapping_torus.coefficient_height_bound"
    )


def test_exact_result_growth_is_rejected_before_exterior_expansion() -> None:
    large = 10**20
    involution = (
        (1, -2 * large, 0, 0, 0, 0),
        (0, -1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 1),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        mapping_torus_chain_complex(_matrix(involution), 2)

    assert error.value.errors()[0]["type"] == (
        "crystallographic.mapping_torus.result_size_bound"
    )


def test_serialized_result_composes_into_integral_homology() -> None:
    produced = mapping_torus_chain_complex(_matrix(((-1,),)), 2)
    restored = ChainComplexValue.model_validate_json(produced.model_dump_json())

    groups = _integral_groups(restored)
    assert groups[1].torsion_invariant_factors == (2,)


def test_request_schema_and_catalog_example_are_executable() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "crystallographic.mapping_torus.chain_complex.compute"
    )
    example = tool.examples[0]
    request = CrystallographicMappingTorusRequest.model_validate_json(
        json.dumps(example.input)
    )

    assert tool.run(request) == mapping_torus_chain_complex(_matrix(((-1,),)), 2)
    schema = CrystallographicMappingTorusRequest.model_json_schema()
    assert "A^m = I" in schema["properties"]["finite_order_exponent"]["description"]


def test_result_round_trip_rejects_noncanonical_integer_entries() -> None:
    result = mapping_torus_chain_complex(_matrix(((-1,),)), 2)
    payload = json.loads(result.model_dump_json())
    payload["differential_matrices"][1][1][0] = "-02"

    with pytest.raises(ValidationError):
        ChainComplexValue.model_validate_json(json.dumps(payload))
