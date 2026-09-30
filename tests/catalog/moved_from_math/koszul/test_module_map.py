"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/koszul/test_module_map.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
)


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _dual_numbers() -> tuple[FiniteCommutativeAlgebra, BasedFiniteModule]:
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


def test_catalog_declares_module_map_transport() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "homological.koszul.module_map.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.source == result.target
    assert result.degree_maps[0].entries == ((0, 0, q(1)),)
