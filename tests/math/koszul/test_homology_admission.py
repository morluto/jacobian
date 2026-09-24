"""Coefficient-aware admission tests for finite-module Koszul homology."""

from fractions import Fraction

import pytest

import jacobian.math.koszul.module_operations as koszul_operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_homology,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def test_large_component_rejected_before_source_replay_or_rank(monkeypatch) -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1",),
        multiplication=(((q(1),),),),
        unit=(q(1),),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("m",),
        action=(((q(1),),),),
    )
    # This is a canonical input at the shared 32,768-digit limit.
    huge = CanonicalRational.from_fraction(Fraction(10**32_767))
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(
            algebra=algebra,
            module=module,
            sequence=((huge,),),
        )
    )

    def arithmetic_must_not_run(_matrix):
        raise AssertionError("exact arithmetic ran before homology admission")

    monkeypatch.setattr(koszul_operations, "_action_matrix", arithmetic_must_not_run)
    monkeypatch.setattr(koszul_operations, "_matrix_rank", arithmetic_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as admission:
        module_koszul_homology(complex_value)
    assert admission.value.errors()[0]["type"] == (
        "koszul.module.homology_coefficient_budget"
    )


def test_dense_matrix_work_rejected_before_source_replay_or_rank(monkeypatch) -> None:
    zero = (q(0), q(0))
    e1 = (q(1), q(0))
    algebra = FiniteCommutativeAlgebra(
        basis=("e1", "e2"),
        multiplication=((e1, zero), (zero, (q(0), q(1)))),
        unit=(q(1), q(1)),
    )
    # P = [I; J] [I-2J, J] for 2 by 2 all-ones J, so P is a
    # dense idempotent and I-P is its complementary idempotent.
    projection = tuple(
        tuple(q(value) for value in row)
        for row in (
            (-1, -2, 1, 1),
            (-2, -1, 1, 1),
            (-3, -3, 2, 2),
            (-3, -3, 2, 2),
        )
    )
    complement = tuple(
        tuple(
            q(int(row == column) - projection[row][column].num) for column in range(4)
        )
        for row in range(4)
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=tuple(f"m{index}" for index in range(4)),
        action=(projection, complement),
    )
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=(e1,) * 6)
    )

    def arithmetic_must_not_run(_matrix):
        raise AssertionError("exact arithmetic ran before homology admission")

    monkeypatch.setattr(koszul_operations, "_action_matrix", arithmetic_must_not_run)
    monkeypatch.setattr(koszul_operations, "_matrix_rank", arithmetic_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as admission:
        module_koszul_homology(complex_value)
    assert admission.value.errors()[0]["type"] == "koszul.module.homology_work_budget"


def test_exact_homology_example_remains_admitted() -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1", "x"),
        multiplication=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(1)), (q(0), q(0))),
        ),
        unit=(q(1), q(0)),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "x"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(
            algebra=algebra,
            module=module,
            sequence=((q(0), q(1)),),
        )
    )

    result = module_koszul_homology(complex_value)
    assert result.dimensions == (1, 1)
    assert result.cycle_dimensions == (2, 1)
    assert result.boundary_dimensions == (1, 0)


def test_large_echoed_label_rejected_before_rank(monkeypatch) -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1",),
        multiplication=(((q(1),),),),
        unit=(q(1),),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("\x00" * 1_400_000,),
        action=(((q(1),),),),
    )
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=())
    )

    def rank_must_not_run(_matrix):
        raise AssertionError("RREF ran before output admission")

    monkeypatch.setattr(koszul_operations, "_matrix_rank", rank_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as admission:
        module_koszul_homology(complex_value)
    assert admission.value.errors()[0]["type"] == "koszul.module.homology_output_budget"
