from __future__ import annotations

from importlib import import_module

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredChainComplexRequest,
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    associated_graded,
)
from jacobian.math.topology.chain_complexes.filtered_direct_sum import (
    filtered_direct_sum,
)
from jacobian.math.topology.chain_complexes.operations import chain_map_commutes
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)

_OPERATION_ID = "homological.filtered_chain_complex.direct_sum.compute"
_DIRECT_SUM_MODULE = import_module(
    "jacobian.math.topology.chain_complexes.filtered_direct_sum"
)


def _filtered(
    boundary: list[list[int]],
    bottom: tuple[list[list[int]], list[list[int]]],
) -> FilteredChainComplexRequest:
    complex_value = ChainComplexValue.model_validate(
        {
            "coefficient_ring": "QQ",
            "degree_min": 0,
            "degree_max": 1,
            "basis_sizes": [1, 1],
            "differential_matrices": [boundary],
        }
    )
    return FilteredChainComplexRequest(
        complex=complex_value,
        filtration=(
            FiltrationLevel(
                subspaces=tuple(
                    FilteredSubspace(vectors=tuple(tuple(row) for row in vectors))
                    for vectors in bottom
                )
            ),
            FiltrationLevel(
                subspaces=(
                    FilteredSubspace(vectors=((1,),)),
                    FilteredSubspace(vectors=((1,),)),
                )
            ),
        ),
    )


def _full_zero_filtered_complex(dimension: int) -> FilteredChainComplexRequest:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(dimension,),
        differential_matrices=(),
    )
    basis = tuple(
        tuple(1 if row == column else 0 for row in range(dimension))
        for column in range(dimension)
    )
    return FilteredChainComplexRequest(
        complex=complex_value,
        filtration=(FiltrationLevel(subspaces=(FilteredSubspace(vectors=basis),)),),
    )


def test_filtered_direct_sum_block_differential_filtration_and_composition() -> None:
    left = _filtered([[1]], ([[1]], []))
    right = _filtered([[2]], ([[1]], [[1]]))
    result = filtered_direct_sum(left, right)

    output = result.filtered_complex
    assert output.complex.basis_sizes == (2, 2)
    assert output.complex.differential_matrices == (((1, 0), (0, 2)),)
    assert output.filtration[0].subspaces[0].vectors == ((1, 0), (0, 1))
    assert output.filtration[0].subspaces[1].vectors == ((0, 1),)
    assert result.left_inclusions == (
        ((1,), (0,)),
        ((1,), (0,)),
    )
    assert result.right_inclusions == (
        ((0,), (1,)),
        ((0,), (1,)),
    )
    assert chain_map_commutes(
        result.left.complex, output.complex, result.left_inclusions
    ).is_valid
    assert chain_map_commutes(
        result.right.complex, output.complex, result.right_inclusions
    ).is_valid

    restored = FilteredChainComplexRequest.model_validate_json(output.model_dump_json())
    graded = associated_graded(restored.complex, restored.filtration)
    assert graded.graded_dimensions == ((2, 1), (0, 1))
    assert graded.graded_differentials[0] == (((0,), (2,)),)
    assert graded.graded_differentials[1] == ((),)


def test_filtered_direct_sum_keeps_zero_dimensional_degenerate_case() -> None:
    empty = FilteredChainComplexRequest(
        complex=ChainComplexValue(
            coefficient_ring=CoefficientRing.RATIONAL,
            degree_min=0,
            degree_max=0,
            basis_sizes=(0,),
            differential_matrices=(),
        ),
        filtration=(FiltrationLevel(subspaces=(FilteredSubspace(vectors=()),)),),
    )
    result = filtered_direct_sum(empty, empty)
    assert result.filtered_complex.complex.basis_sizes == (0,)
    assert result.left_inclusions == ((),)
    assert result.right_inclusions == ((),)


def test_filtered_direct_sum_bounds_output_dimension_before_expansion() -> None:
    left = _full_zero_filtered_complex(17)
    right = _full_zero_filtered_complex(17)
    with pytest.raises(OperationResourceAdmissionError) as error:
        filtered_direct_sum(left, right)
    assert (
        error.value.errors()[0]["type"]
        == "filtered_direct_sum.ambient_dimension_exceeded"
    )


def test_filtered_direct_sum_admits_validation_work_before_exact_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _filtered([[1]], ([[1]], []))
    right = _filtered([[2]], ([[1]], [[1]]))
    monkeypatch.setattr(_DIRECT_SUM_MODULE, "MAX_FILTERED_DIRECT_SUM_WORK", 1)
    with pytest.raises(OperationResourceAdmissionError) as error:
        filtered_direct_sum(left, right)
    assert error.value.errors()[0]["type"] == "filtered_direct_sum.work_budget_exceeded"


def test_filtered_direct_sum_rejects_misaligned_filtration_axes() -> None:
    one = _filtered([[0]], ([], []))
    two_levels = one.model_copy(
        update={"filtration": one.filtration + one.filtration[-1:]}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        filtered_direct_sum(one, two_levels)
    assert (
        error.value.errors()[0]["type"]
        == "filtered_direct_sum.filtration_axis_mismatch"
    )


def test_filtered_direct_sum_revalidates_model_constructed_input() -> None:
    valid = _filtered([[0]], ([], []))
    forged = FilteredChainComplexRequest.model_construct(
        complex=valid.complex,
        filtration=(FiltrationLevel.model_construct(subspaces=()),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        filtered_direct_sum(forged, valid)
    assert error.value.errors()[0]["type"] == "filtered_chain_complex.structure_invalid"


def test_direct_sum_example_executes_through_catalog() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_OPERATION_ID)
    assert operation is not None
    result = invoke_operation(_OPERATION_ID, operation.examples[0].input, catalog)
    assert result.output["filtered_complex"]["complex"]["basis_sizes"] == [2]
    assert result.output["left"]["complex"]["basis_sizes"] == [1]
    assert result.output["right"]["complex"]["basis_sizes"] == [1]
