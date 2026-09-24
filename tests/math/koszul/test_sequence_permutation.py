"""Exact exterior-power maps for permuted finite-module Koszul sequences."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

import jacobian.math.koszul.module_operations as operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul._tools import TOOLS
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleDifferential,
    ModuleKoszulRequest,
    ModuleKoszulSequencePermutationRequest,
    ModuleKoszulUnitContractionRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_sequence_permute,
    module_koszul_unit_contract,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _source(sequence):
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((q(1),),),))
    return module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=sequence)
    )


def _sparse_columns(matrix):
    columns = [{} for _ in range(matrix.column_count)]
    for row, column, coefficient in matrix.entries:
        columns[column][row] = Fraction(coefficient.num, coefficient.den)
    return columns


def _apply(matrix, vector):
    result = {}
    for row, column, coefficient in matrix.entries:
        scalar = vector.get(column, Fraction(0))
        if scalar:
            result[row] = result.get(row, Fraction(0)) + scalar * Fraction(
                coefficient.num, coefficient.den
            )
    return {index: value for index, value in result.items() if value}


def _replay_chain_isomorphism(result):
    for degree, (source_d, target_d) in enumerate(
        zip(
            result.source_complex.differentials,
            result.target_complex.differentials,
            strict=True,
        ),
        start=1,
    ):
        for column in range(source_d.column_count):
            vector = {column: Fraction(1)}
            left = _apply(result.source_to_target[degree - 1], _apply(source_d, vector))
            right = _apply(target_d, _apply(result.source_to_target[degree], vector))
            assert left == right
    for forward, backward in zip(
        result.source_to_target, result.target_to_source, strict=True
    ):
        forward_columns = _sparse_columns(forward)
        backward_columns = _sparse_columns(backward)
        for source_index, image in enumerate(forward_columns):
            assert len(image) == 1
            target_index, coefficient = next(iter(image.items()))
            assert backward_columns[target_index] == {source_index: coefficient}


def test_two_term_swap_reverses_exterior_sign_and_commutes_with_differential():
    source = _source(((q(1),), (q(2),)))
    result = module_koszul_sequence_permute(
        ModuleKoszulSequencePermutationRequest(complex=source, new_to_old=(1, 0))
    )
    assert result.target_complex.sequence == ((q(2),), (q(1),))
    assert result.source_to_target[1].entries == (
        (0, 1, q(1)),
        (1, 0, q(1)),
    )
    assert result.source_to_target[2].entries == ((0, 0, q(-1)),)
    assert result.target_to_source[2].entries == ((0, 0, q(-1)),)
    _replay_chain_isomorphism(result)


def test_three_cycle_uses_inverse_index_map_and_preserves_exterior_signs():
    source = _source(((q(1),), (q(2),), (q(3),)))
    result = module_koszul_sequence_permute(
        ModuleKoszulSequencePermutationRequest(complex=source, new_to_old=(1, 2, 0))
    )
    assert result.target_complex.sequence == ((q(2),), (q(3),), (q(1),))
    assert result.source_to_target[1].entries == (
        (0, 1, q(1)),
        (1, 2, q(1)),
        (2, 0, q(1)),
    )
    assert result.source_to_target[2].entries == (
        (0, 2, q(1)),
        (1, 0, q(-1)),
        (2, 1, q(-1)),
    )
    assert result.source_to_target[3].entries == ((0, 0, q(1)),)
    _replay_chain_isomorphism(result)


def test_empty_and_identity_permutations_are_valid():
    empty = module_koszul_sequence_permute(
        ModuleKoszulSequencePermutationRequest(complex=_source(()), new_to_old=())
    )
    assert empty.source_to_target[0].entries == ((0, 0, q(1)),)
    identity = module_koszul_sequence_permute(
        ModuleKoszulSequencePermutationRequest(
            complex=_source(((q(1),), (q(0),))), new_to_old=(0, 1)
        )
    )
    assert all(
        forward == backward
        for forward, backward in zip(
            identity.source_to_target, identity.target_to_source, strict=True
        )
    )


def test_source_differential_must_match_its_retained_sequence():
    source = _source(((q(1),),))
    forged = source.model_copy(
        update={
            "differentials": (
                ModuleDifferential(row_count=1, column_count=1, entries=()),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as caught:
        module_koszul_sequence_permute(
            ModuleKoszulSequencePermutationRequest(complex=forged, new_to_old=(0,))
        )
    assert caught.value.errors()[0]["type"] == "koszul.module.source_complex_mismatch"


def test_invalid_permutation_is_rejected():
    with pytest.raises(ValidationError):
        ModuleKoszulSequencePermutationRequest(
            complex=_source(((q(1),), (q(2),))), new_to_old=(0, 0)
        )


def test_transform_output_bound_rejects_before_rebuilding(monkeypatch):
    source = _source(((q(1),), (q(2),)))

    def build_must_not_run(_request):
        raise AssertionError("complex expansion ran before transform admission")

    monkeypatch.setattr(operations, "_build_module_koszul_complex", build_must_not_run)
    monkeypatch.setattr(operations, "MAX_KOSZUL_SEQUENCE_TRANSFORM_OUTPUT_BYTES", 1)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        module_koszul_sequence_permute(
            ModuleKoszulSequencePermutationRequest(complex=source, new_to_old=(1, 0))
        )
    assert (
        caught.value.errors()[0]["type"] == "koszul.module.sequence_permutation_budget"
    )


def test_published_permutation_example_runs():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "homological.koszul.sequence_permute.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.source_to_target[2].entries == ((0, 0, q(-1)),)
    _replay_chain_isomorphism(result)


def test_unit_contraction_returns_exact_homotopy_and_inverse_for_nonzero_index():
    algebra = FiniteCommutativeAlgebra(
        basis=("1", "e"),
        multiplication=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(1)), (q(0), q(0))),
        ),
        unit=(q(1), q(0)),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("m", "em"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    sequence = ((q(0), q(1)), (q(1), q(0)))
    result = module_koszul_unit_contract(
        ModuleKoszulUnitContractionRequest(
            complex=module_koszul_complex(
                ModuleKoszulRequest(algebra=algebra, module=module, sequence=sequence)
            ),
            unit_index=1,
        )
    )
    with pytest.raises(OperationDomainValidationError) as nonunit:
        module_koszul_unit_contract(
            ModuleKoszulUnitContractionRequest(complex=result.complex, unit_index=0)
        )
    assert nonunit.value.errors()[0]["type"] == "koszul.module.sequence_entry_not_unit"
    assert result.inverse == (q(1), q(0))
    assert result.homotopy[0].entries == ((2, 0, q(1)), (3, 1, q(1)))
    # Independently replay dH + Hd on every basis vector, including the top.
    for degree, size in enumerate(result.complex.basis_sizes):
        for column in range(size):
            basis = {column: Fraction(1)}
            dh = (
                _apply(
                    result.complex.differentials[degree],
                    _apply(result.homotopy[degree], basis),
                )
                if degree < len(result.complex.sequence)
                else {}
            )
            hd = (
                _apply(
                    result.homotopy[degree - 1],
                    _apply(result.complex.differentials[degree - 1], basis),
                )
                if degree > 0
                else {}
            )
            combined = {
                i: dh.get(i, Fraction(0)) + hd.get(i, Fraction(0))
                for i in set(dh) | set(hd)
            }
            assert {i: value for i, value in combined.items() if value} == basis


def test_nonunit_entry_is_rejected_and_published_example_runs():
    source = _source(((q(0),),))
    with pytest.raises(OperationDomainValidationError) as caught:
        module_koszul_unit_contract(
            ModuleKoszulUnitContractionRequest(complex=source, unit_index=0)
        )
    assert caught.value.errors()[0]["type"] == "koszul.module.sequence_entry_not_unit"
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "homological.koszul.unit_contraction.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.homotopy[0].entries == ((0, 0, q(1)),)
    assert result.homotopy[1].entries == ()


def test_contraction_rejects_a_forged_square_zero_claim():
    source = _source(((q(1),),)).model_copy(update={"square_zero": False})
    with pytest.raises(OperationDomainValidationError) as caught:
        module_koszul_unit_contract(
            ModuleKoszulUnitContractionRequest(complex=source, unit_index=0)
        )
    assert caught.value.errors()[0]["type"] == "koszul.module.source_complex_mismatch"


def test_contraction_budget_rejects_before_inverse_or_complex_rebuild(monkeypatch):
    source = _source(((q(1),),))

    def arithmetic_must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "unit solving or complex reconstruction ran before admission"
        )

    monkeypatch.setattr(operations, "_algebra_inverse", arithmetic_must_not_run)
    monkeypatch.setattr(
        operations, "_build_module_koszul_complex", arithmetic_must_not_run
    )
    monkeypatch.setattr(operations, "MAX_KOSZUL_UNIT_CONTRACTION_OUTPUT_BYTES", 1)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        module_koszul_unit_contract(
            ModuleKoszulUnitContractionRequest(complex=source, unit_index=0)
        )
    assert caught.value.errors()[0]["type"] == "koszul.module.unit_contraction_budget"
