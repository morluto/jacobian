"""Top Koszul homology and simultaneous annihilator contracts."""

import json
from fractions import Fraction

import pytest

import jacobian.math.koszul.module_operations as koszul_operations
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
    ModuleKoszulTopHomologyRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_top_homology,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def dual_numbers_regular_module() -> tuple[FiniteCommutativeAlgebra, BasedFiniteModule]:
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
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    return algebra, module


def _independent_common_kernel(
    module: BasedFiniteModule,
    sequence: tuple[tuple[CanonicalRational, ...], ...],
) -> tuple[tuple[CanonicalRational, ...], ...]:
    """Small exact oracle: stack the element-action matrices and row-reduce."""
    dimension = len(module.basis)
    matrix: list[list[Fraction]] = []
    for element in sequence:
        action = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
        for scalar, basis_action in zip(element, module.action, strict=True):
            for row in range(dimension):
                for column in range(dimension):
                    action[row][column] += Fraction(scalar.num, scalar.den) * Fraction(
                        basis_action[row][column].num, basis_action[row][column].den
                    )
        matrix.extend(action)

    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(dimension):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]), None
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [entry / scale for entry in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row != pivot_row and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [
                    left - factor * right
                    for left, right in zip(matrix[row], matrix[pivot_row], strict=True)
                ]
        pivot_columns.append(column)
        pivot_row += 1

    pivot_rows = dict(zip(pivot_columns, matrix[:pivot_row], strict=True))
    basis = []
    for free_column in (
        column for column in range(dimension) if column not in pivot_rows
    ):
        vector = [Fraction(0) for _ in range(dimension)]
        vector[free_column] = Fraction(1)
        for column, row in pivot_rows.items():
            vector[column] = -row[free_column]
        basis.append(tuple(CanonicalRational.from_fraction(item) for item in vector))
    return tuple(basis)


def _compute(module, sequence):
    algebra = module.algebra
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=sequence)
    )
    return module_koszul_top_homology(
        ModuleKoszulTopHomologyRequest(complex=complex_value)
    )


def test_dual_number_top_homology_is_epsilon_annihilator_by_independent_oracle():
    algebra, module = dual_numbers_regular_module()
    sequence = ((q(0), q(1)),)
    result = _compute(module, sequence)
    repeated = ((q(0), q(1)), (q(0), q(1)))
    repeated_result = _compute(module, repeated)

    # Multiplication by e kills e and does not kill 1.
    assert result.annihilator_basis == _independent_common_kernel(module, sequence)
    assert result.annihilator_basis == ((q(0), q(1)),)
    assert result.top_homology_basis == ((q(0), q(1)),)
    assert result.algebra == algebra
    assert result.module == module
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert repeated_result.annihilator_basis == _independent_common_kernel(
        module, repeated
    )
    assert repeated_result.annihilator_basis == ((q(0), q(1)),)
    assert repeated_result.top_homology_basis == repeated_result.annihilator_basis
    assert repeated_result.top_differential is not None
    assert repeated_result.top_differential.row_count == 4


def test_top_annihilator_handles_empty_zero_and_unit_sequences():
    _algebra, module = dual_numbers_regular_module()
    empty = _compute(module, ())
    zero = _compute(module, ((q(0), q(0)),))
    unit = _compute(module, ((q(1), q(0)),))

    assert empty.annihilator_basis == ((q(1), q(0)), (q(0), q(1)))
    assert empty.top_homology_basis == empty.annihilator_basis
    assert zero.annihilator_basis == _independent_common_kernel(module, ((q(0), q(0)),))
    assert len(zero.annihilator_basis) == 2
    assert unit.annihilator_basis == ()
    assert unit.top_homology_basis == ()


def test_top_homology_rejects_square_zero_but_noncanonical_source_differential():
    _algebra, module = dual_numbers_regular_module()
    valid = module_koszul_complex(
        ModuleKoszulRequest(
            algebra=module.algebra,
            module=module,
            sequence=((q(0), q(1)),),
        )
    )
    forged = type(valid).model_construct(
        algebra=valid.algebra,
        module=valid.module,
        sequence=valid.sequence,
        basis_sizes=valid.basis_sizes,
        differentials=(ModuleDifferential(row_count=2, column_count=2, entries=()),),
        square_zero=True,
    )

    with pytest.raises(OperationDomainValidationError) as error:
        module_koszul_top_homology(ModuleKoszulTopHomologyRequest(complex=forged))
    assert error.value.errors()[0]["type"] == "koszul.module.source_complex_mismatch"


def test_top_homology_rejects_oversized_coefficients_before_algebra_replay(
    monkeypatch: pytest.MonkeyPatch,
):
    huge = CanonicalRational.from_fraction(Fraction(10**128))
    algebra = FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((q(1),),),))
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=((q(1),),))
    )
    oversized = type(complex_value).model_construct(
        algebra=complex_value.algebra,
        module=complex_value.module,
        sequence=((huge,),),
        basis_sizes=complex_value.basis_sizes,
        differentials=complex_value.differentials,
        square_zero=True,
    )

    def arithmetic_must_not_run(*_args, **_kwargs):
        raise AssertionError("algebra validation ran before coefficient admission")

    monkeypatch.setattr(koszul_operations, "_admit", arithmetic_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as error:
        module_koszul_top_homology(ModuleKoszulTopHomologyRequest(complex=oversized))
    assert (
        error.value.errors()[0]["type"] == "koszul.module.homology_coefficient_budget"
    )


def test_top_homology_is_published_and_example_executes():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "homological.koszul.top_homology.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.annihilator_basis == ((q(1),),)
