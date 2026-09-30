"""Package-native Koszul composition uses mathematical values, not requests."""

from collections.abc import Iterator

import pytest

import jacobian.math.koszul as koszul
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_native_map_homology_change_and_top_composition() -> None:
    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    chain = koszul.module_koszul_map(algebra, module, module, ((q(0),),), ((q(2),),))
    assert chain.degree_maps[0].entries == ((0, 0, q(2)),)
    assert chain.degree_maps[1].entries == ((0, 0, q(2)),)
    decoded = koszul.ModuleKoszulChainMap.model_validate_json(chain.model_dump_json())
    homology = koszul.koszul_homology_map(decoded)
    assert homology.degree_maps == (((q(2),),), ((q(2),),))
    source = koszul.ModuleKoszulComplex.model_validate_json(
        chain.source_complex.model_dump_json()
    )
    changed = koszul.module_koszul_sequence_linear_change(source, ((q(2),),))
    assert changed.target_complex.sequence == ((q(0),),)
    top = koszul.module_koszul_top_homology(changed.target_complex)
    assert top.annihilator_basis == ((q(1),),)


def test_native_map_reports_invalid_domain_axes() -> None:
    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    with pytest.raises(OperationDomainValidationError):
        koszul.module_koszul_map(algebra, module, module, ((q(0),),), ())


@pytest.mark.parametrize("operation", ["map", "homology", "change", "top"])
def test_forged_nested_module_is_revalidated(operation: str) -> None:
    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    chain = koszul.module_koszul_map(algebra, module, module, ((q(0),),), ((q(1),),))
    forged = koszul.BasedFiniteModule.model_construct(
        algebra=algebra, basis=("m",), action=()
    )
    complex_value = chain.source_complex.model_copy(update={"module": forged})
    with pytest.raises(OperationDomainValidationError):
        if operation == "map":
            koszul.module_koszul_map(algebra, forged, module, ((q(0),),), ((q(1),),))
        elif operation == "homology":
            koszul.koszul_homology_map(
                chain.model_copy(
                    update={"source": forged, "source_complex": complex_value}
                )
            )
        elif operation == "change":
            koszul.module_koszul_sequence_linear_change(complex_value, ((q(1),),))
        else:
            koszul.module_koszul_top_homology(complex_value)


def test_unbounded_forged_axis_is_refused_without_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    forged = koszul.BasedFiniteModule.model_construct(
        algebra=algebra, basis=("m",), action=((),) * 100_000
    )

    def forbidden_dump(*args: object, **kwargs: object) -> None:
        pytest.fail("forged carrier was serialized before raw admission")

    monkeypatch.setattr(koszul.BasedFiniteModule, "model_dump", forbidden_dump)
    with pytest.raises(OperationResourceAdmissionError, match="dimension envelope"):
        koszul.module_koszul_map(algebra, forged, forged, (), ((q(1),),))


def test_native_axis_subclass_cannot_supply_callbacks() -> None:
    class PoisonTuple(tuple[object, ...]):
        def __iter__(self) -> Iterator[object]:
            raise AssertionError("untrusted tuple callback ran")

    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    forged = koszul.BasedFiniteModule.model_construct(
        algebra=algebra, basis=("m",), action=PoisonTuple(())
    )
    with pytest.raises(OperationDomainValidationError, match="exact tuples"):
        koszul.module_koszul_map(algebra, forged, forged, (), ((q(1),),))


def test_native_snapshot_shares_request_deadline() -> None:
    from time import monotonic

    from jacobian._execution import OperationExecutionTimeoutError, request_execution

    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    started = monotonic()
    with (
        request_execution(started, outer_deadline=started - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        koszul.module_koszul_map(algebra, module, module, (), ((q(1),),))


def test_native_snapshot_checks_cancellation_during_sparse_copy() -> None:
    from time import monotonic

    from jacobian._execution import OperationExecutionCancelledError, request_execution
    from jacobian.math.koszul.module_models import ModuleDifferential

    class CancelDuringCopy:
        checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 2

    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    value = koszul.ModuleKoszulComplex.model_construct(
        algebra=algebra,
        module=module,
        sequence=((q(0),),),
        basis_sizes=(64, 64),
        differentials=(
            ModuleDifferential.model_construct(
                row_count=64,
                column_count=64,
                entries=tuple((i, j, q(1)) for i in range(64) for j in range(64)),
            ),
        ),
        square_zero=True,
    )
    signal = CancelDuringCopy()
    with (
        request_execution(monotonic(), cancellation_signal=signal),
        pytest.raises(OperationExecutionCancelledError),
    ):
        koszul.module_koszul_top_homology(value)
    assert signal.checks == 2


def test_native_snapshot_bounds_aggregate_scalar_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.koszul import module_operations
    from jacobian.math.koszul.module_models import ModuleDifferential

    algebra = koszul.FiniteCommutativeAlgebra(
        basis=("1",), multiplication=(((q(1),),),), unit=(q(1),)
    )
    module = koszul.BasedFiniteModule(
        algebra=algebra, basis=("m",), action=(((q(1),),),)
    )
    coefficient = q(10**4000)
    value = koszul.ModuleKoszulComplex.model_construct(
        algebra=algebra,
        module=module,
        sequence=((q(0),),),
        basis_sizes=(64, 64),
        differentials=(
            ModuleDifferential.model_construct(
                row_count=64,
                column_count=64,
                entries=tuple(
                    (i, j, coefficient) for i in range(64) for j in range(64)
                ),
            ),
        ),
        square_zero=True,
    )

    def forbidden_kernel(*args: object, **kwargs: object) -> None:
        pytest.fail("large scalar payload reached kernel validation before admission")

    monkeypatch.setattr(module_operations, "_admit_top_homology", forbidden_kernel)
    with pytest.raises(
        OperationResourceAdmissionError, match="retained input envelope"
    ):
        koszul.module_koszul_top_homology(value)
