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
    ModuleKoszulChainMap,
    ModuleKoszulHomologyMapRequest,
    ModuleKoszulMapRequest,
)
from jacobian.math.koszul.module_operations import module_koszul_map


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


def test_forged_small_chain_bases_do_not_bypass_admission() -> None:
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
    payload = chain_map.model_dump(mode="json")
    for side in ("source_complex", "target_complex"):
        payload[side]["basis_sizes"] = [0, 0, 0]
        payload[side]["differentials"] = [
            {"row_count": 0, "column_count": 0, "entries": []},
            {"row_count": 0, "column_count": 0, "entries": []},
        ]
    payload["degree_maps"] = [
        {"row_count": 0, "column_count": 0, "entries": []} for _ in range(3)
    ]
    forged = ModuleKoszulChainMap.model_validate_json(
        encode_strict_json(payload), strict=True
    )

    # Each true complex has eight total basis vectors, so the pair exceeds the
    # induced-map cap even though the serialized claims say both are zero.
    with pytest.raises(OperationResourceAdmissionError) as error:
        koszul_homology_map(ModuleKoszulHomologyMapRequest(chain_map=forged))
    assert error.value.errors()[0]["type"] == "koszul.module.homology_map_basis_budget"


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
