from jacobian._exact import CanonicalRational
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulDifferentialRequest,
    ModuleKoszulRequest,
)
from jacobian.math.koszul.module_operations import module_koszul_differential


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_degree_two_differential_matches_independent_wedge_formula() -> None:
    # A=QQ[x]/(x^2), basis (1,x), and the regular A-module A.
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
    # d_2(1 tensor e_12) = x tensor e_2 - 1 tensor e_1.
    request = ModuleKoszulRequest(
        algebra=algebra,
        module=module,
        sequence=((q(0), q(1)), (q(1), q(0))),
    )
    result = module_koszul_differential(
        ModuleKoszulDifferentialRequest(request=request, degree=2)
    )

    assert result.source_wedges == ((0, 1),)
    assert result.target_wedges == ((0,), (1,))
    assert result.differential.row_count == 4
    assert result.differential.column_count == 2
    assert result.differential.entries == (
        (0, 0, q(-1)),
        (1, 1, q(-1)),
        (3, 0, q(1)),
    )
