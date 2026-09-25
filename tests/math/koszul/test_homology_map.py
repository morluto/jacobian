"""Functorial maps on exact finite-module Koszul homology."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.homology_map import koszul_homology_map
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleChainMapMatrix,
    ModuleDifferential,
    ModuleKoszulChainMap,
    ModuleKoszulComplex,
    ModuleKoszulHomologyMapRequest,
    ModuleKoszulMapRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_homology,
    module_koszul_map,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _dual_numbers() -> FiniteCommutativeAlgebra:
    return FiniteCommutativeAlgebra(
        basis=("1", "e"),
        multiplication=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(1)), (q(0), q(0))),
        ),
        unit=(q(1), q(0)),
    )


def test_induced_homology_maps_use_quotient_class_coordinates() -> None:
    algebra = _dual_numbers()
    source = BasedFiniteModule(
        algebra=algebra,
        basis=("1_mod_e",),
        action=(((q(1),),), ((q(0),),)),
    )
    target = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=source,
            target=target,
            sequence=((q(0), q(1)),),
            # 1 mod e -> e is an exact A-module map.
            map_matrix=((q(0),), (q(1),)),
        )
    )

    result = koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=chain_map))
    assert result.source_homology.dimensions == (1, 1)
    assert result.target_homology.dimensions == (1, 1)
    # In degree zero e is a boundary; in degree one e tensor e is a nonzero
    # class. This independently checks quotient-coordinate handling.
    assert result.degree_maps == (((q(0),),), ((q(1),),))
    assert ModuleKoszulHomologyMapRequest(
        chain_map=ModuleKoszulChainMap.model_validate_json(
            encode_strict_json(chain_map.model_dump(mode="json")), strict=True
        )
    )


def test_forged_typed_request_is_revalidated_before_chain_map_access() -> None:
    request = ModuleKoszulHomologyMapRequest.model_construct(chain_map=None)
    with pytest.raises(OperationDomainValidationError) as error:
        koszul_homology_map(request)
    assert error.value.errors()[0]["type"] == "koszul.module.homology_map_request"


def test_consumer_rechecks_serialized_chain_map_relations() -> None:
    algebra = _dual_numbers()
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=module,
            target=module,
            sequence=((q(0), q(1)),),
            map_matrix=((q(1), q(0)), (q(0), q(1))),
        )
    )
    corrupted = chain_map.model_copy(
        update={"module_map": ((q(1), q(0)), (q(0), q(0)))}
    )
    with pytest.raises(OperationDomainValidationError):
        koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=corrupted))


def test_forged_typed_chain_bases_are_revalidated_before_admission() -> None:
    algebra = _dual_numbers()
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=module,
            target=module,
            sequence=((q(0), q(1)), (q(0), q(1))),
            map_matrix=((q(1), q(0)), (q(0), q(1))),
        )
    )
    # model_construct bypasses the axis validator, so the operation itself must
    # reject the forged zero-sized chain bases rather than trusting them.
    forged_complexes = tuple(
        ModuleKoszulComplex.model_construct(
            algebra=algebra,
            module=module,
            sequence=chain_map.sequence,
            basis_sizes=(0, 0, 0),
            differentials=(
                ModuleDifferential(row_count=0, column_count=0, entries=()),
                ModuleDifferential(row_count=0, column_count=0, entries=()),
            ),
            square_zero=True,
        )
        for _ in ("source", "target")
    )
    forged = ModuleKoszulChainMap.model_construct(
        algebra=algebra,
        source=module,
        target=module,
        sequence=chain_map.sequence,
        module_map=chain_map.module_map,
        source_complex=forged_complexes[0],
        target_complex=forged_complexes[1],
        degree_maps=tuple(
            ModuleChainMapMatrix(row_count=0, column_count=0, entries=())
            for _ in range(3)
        ),
    )

    request = ModuleKoszulHomologyMapRequest.model_construct(chain_map=forged)
    with pytest.raises(OperationDomainValidationError) as error:
        koszul_homology_map(request)
    assert error.value.errors()[0]["type"] == "koszul.module.homology_map_request"


def test_induced_map_envelope_is_preflighted_before_reconstruction(monkeypatch) -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1",),
        multiplication=(((q(1),),),),
        unit=(q(1),),
    )
    source = BasedFiniteModule(
        algebra=algebra,
        basis=tuple(f"m{index}" for index in range(8)),
        action=(
            tuple(
                tuple(q(int(row == column)) for column in range(8)) for row in range(8)
            ),
        ),
    )
    target = BasedFiniteModule(
        algebra=algebra,
        basis=("m",),
        action=(((q(1),),),),
    )
    # The pair has 8 + 1 = 9 chain cells, one above the induced-map cap, while
    # the standalone module map is still a cheap accepted request.
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=source,
            target=target,
            sequence=(),
            map_matrix=((q(1), q(0), q(0), q(0), q(0), q(0), q(0), q(0)),),
        )
    )

    def reconstruction_must_not_run(_request):
        raise AssertionError("reconstruction ran before envelope preflight")

    monkeypatch.setattr(
        "jacobian.math.koszul.homology_map.module_koszul_map",
        reconstruction_must_not_run,
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=chain_map))
    assert error.value.errors()[0]["type"] == (
        "koszul.module.homology_map_basis_budget"
    )


def test_induced_map_coefficient_is_preflighted_before_reconstruction(
    monkeypatch,
) -> None:
    algebra = _dual_numbers()
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    huge = q(10**9)
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=module,
            target=module,
            sequence=((q(0), q(1)),),
            map_matrix=((huge, q(0)), (q(0), huge)),
        )
    )

    def reconstruction_must_not_run(_request):
        raise AssertionError("reconstruction ran before coefficient preflight")

    monkeypatch.setattr(
        "jacobian.math.koszul.homology_map.module_koszul_map",
        reconstruction_must_not_run,
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=chain_map))
    assert error.value.errors()[0]["type"] == (
        "koszul.module.homology_map_coefficient_budget"
    )


def test_induced_map_preflights_echoed_algebra_unit() -> None:
    algebra = FiniteCommutativeAlgebra(
        basis=("1", "e"),
        multiplication=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(1)), (q(0), q(0))),
        ),
        unit=(q(10**9), q(0)),
    )
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=module,
            target=module,
            sequence=((q(0), q(1)),),
            map_matrix=((q(1), q(0)), (q(0), q(1))),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=chain_map))
    assert error.value.errors()[0]["type"] == (
        "koszul.module.homology_map_coefficient_budget"
    )


def test_induced_map_reuses_admitted_complexes(monkeypatch) -> None:
    algebra = _dual_numbers()
    module = BasedFiniteModule(
        algebra=algebra,
        basis=("1", "e"),
        action=(
            ((q(1), q(0)), (q(0), q(1))),
            ((q(0), q(0)), (q(1), q(0))),
        ),
    )
    chain_map = module_koszul_map(
        ModuleKoszulMapRequest(
            algebra=algebra,
            source=module,
            target=module,
            sequence=((q(0), q(1)),),
            map_matrix=((q(1), q(0)), (q(0), q(1))),
        )
    )
    expected_source = module_koszul_homology(chain_map.source_complex)
    expected_target = module_koszul_homology(chain_map.target_complex)

    def admission_must_not_run(*_args, **_kwargs):
        raise AssertionError("freshly reconstructed complex was revalidated")

    monkeypatch.setattr(
        "jacobian.math.koszul.module_operations._admit_complex",
        admission_must_not_run,
    )
    monkeypatch.setattr(
        "jacobian.math.koszul.module_operations._require_square_zero",
        admission_must_not_run,
    )

    result = koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=chain_map))
    # The private kernel must reproduce the public revalidating homology path.
    assert result.source_homology == expected_source
    assert result.target_homology == expected_target


def test_catalog_example_runs() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "homological.koszul.homology_map.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.degree_maps == (((q(1),),),)
