"""Exact chain isomorphisms induced by rational Koszul sequence changes."""

from fractions import Fraction

import pytest

import jacobian.math.koszul.module_operations as operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulRequest,
    ModuleKoszulSequenceLinearChangeRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_sequence_linear_change,
)


def q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _source(sequence):
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((q(1),),),))
    return module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=sequence)
    )


def _apply(matrix, vector):
    result = {}
    for row, column, coefficient in matrix.entries:
        value = vector.get(column, Fraction(0))
        if value:
            result[row] = result.get(row, Fraction(0)) + value * Fraction(
                coefficient.num, coefficient.den
            )
    return {index: value for index, value in result.items() if value}


def _dense(matrix):
    rows = [
        [Fraction(0) for _ in range(matrix.column_count)]
        for _ in range(matrix.row_count)
    ]
    for row, column, coefficient in matrix.entries:
        rows[row][column] = Fraction(coefficient.num, coefficient.den)
    return rows


def test_shear_constructs_exact_target_and_inverse_exterior_chain_maps():
    source = _source(((q(1),), (q(0),)))
    change = ((q(Fraction(1, 2)), q(1)), (q(0), q(1)))
    result = module_koszul_sequence_linear_change(
        ModuleKoszulSequenceLinearChangeRequest(complex=source, change_matrix=change)
    )

    assert result.target_complex.sequence == ((q(Fraction(1, 2)),), (q(0),))
    assert _dense(result.source_to_target[1]) == [
        [Fraction(2), Fraction(0)],
        [Fraction(-2), Fraction(1)],
    ]
    assert _dense(result.source_to_target[2]) == [[Fraction(2)]]
    assert _dense(result.target_to_source[1]) == [
        [Fraction(1, 2), Fraction(0)],
        [Fraction(1), Fraction(1)],
    ]

    # Independently replay both chain-square identities on every basis vector.
    for from_complex, to_complex, maps in (
        (result.source_complex, result.target_complex, result.source_to_target),
        (result.target_complex, result.source_complex, result.target_to_source),
    ):
        for degree, differential in enumerate(from_complex.differentials, start=1):
            target_differential = to_complex.differentials[degree - 1]
            for column in range(differential.column_count):
                basis = {column: Fraction(1)}
                assert _apply(maps[degree - 1], _apply(differential, basis)) == _apply(
                    target_differential, _apply(maps[degree], basis)
                )

    # In each degree, the maps compose to the identity in both directions.
    for _degree, (forward, backward) in enumerate(
        zip(result.source_to_target, result.target_to_source, strict=True)
    ):
        for column in range(forward.column_count):
            basis = {column: Fraction(1)}
            assert _apply(backward, _apply(forward, basis)) == basis
        for column in range(backward.column_count):
            basis = {column: Fraction(1)}
            assert _apply(forward, _apply(backward, basis)) == basis


def test_singular_change_is_rejected():
    source = _source(((q(1),), (q(0),)))
    with pytest.raises(OperationDomainValidationError) as caught:
        module_koszul_sequence_linear_change(
            ModuleKoszulSequenceLinearChangeRequest(
                complex=source,
                change_matrix=((q(1), q(2)), (q(2), q(4))),
            )
        )
    assert caught.value.errors()[0]["type"] == "koszul.module.sequence_change_singular"


def test_row_pivoting_preserves_the_top_exterior_sign():
    source = _source(((q(1),), (q(0),)))
    result = module_koszul_sequence_linear_change(
        ModuleKoszulSequenceLinearChangeRequest(
            complex=source,
            change_matrix=((q(0), q(1)), (q(1), q(0))),
        )
    )
    assert result.target_complex.sequence == ((q(0),), (q(1),))
    assert _dense(result.source_to_target[2]) == [[Fraction(-1)]]


def test_empty_sequence_uses_the_empty_matrix_as_its_identity_change():
    source = _source(())
    result = module_koszul_sequence_linear_change(
        ModuleKoszulSequenceLinearChangeRequest(complex=source, change_matrix=())
    )
    assert result.target_complex.sequence == ()
    assert result.source_to_target == result.target_to_source
    assert _dense(result.source_to_target[0]) == [[Fraction(1)]]


def test_maximum_sequence_length_accepts_the_identity_change():
    source = _source(tuple((q(0),) for _ in range(6)))
    identity = tuple(
        tuple(q(int(row == column)) for column in range(6)) for row in range(6)
    )
    result = module_koszul_sequence_linear_change(
        ModuleKoszulSequenceLinearChangeRequest(complex=source, change_matrix=identity)
    )
    assert tuple(len(degree_map.entries) for degree_map in result.source_to_target) == (
        1,
        6,
        15,
        20,
        15,
        6,
        1,
    )
    assert result.target_complex.sequence == source.sequence


def test_large_denominator_is_admitted_without_decimal_string_conversion(monkeypatch):
    source = _source(((q(1),),))
    huge_denominator = 10**5000 + 1
    change = ((CanonicalRational(num=1, den=huge_denominator),),)

    def fail_if_inverted(_matrix):
        raise AssertionError("oversized coefficient reached matrix inversion")

    monkeypatch.setattr(operations, "_inverse_matrix", fail_if_inverted)
    with pytest.raises(OperationResourceAdmissionError):
        module_koszul_sequence_linear_change(
            ModuleKoszulSequenceLinearChangeRequest(
                complex=source, change_matrix=change
            )
        )


def test_resource_bound_is_checked_before_matrix_inversion(monkeypatch):
    source = _source(((q(1),), (q(0),)))

    def fail_if_inverted(_matrix):
        raise AssertionError("matrix inversion ran before transform admission")

    monkeypatch.setattr(operations, "MAX_KOSZUL_SEQUENCE_LINEAR_CHANGE_WORK", 0)
    monkeypatch.setattr(operations, "_inverse_matrix", fail_if_inverted)
    with pytest.raises(OperationResourceAdmissionError):
        module_koszul_sequence_linear_change(
            ModuleKoszulSequenceLinearChangeRequest(
                complex=source,
                change_matrix=((q(1), q(1)), (q(0), q(1))),
            )
        )
