"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_simplicial_set_subset.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology.simplicial_sets import (
    SimplicialSubsetPrefix,
    standard_simplex,
)
from jacobian.math.topology.simplicial_sets.subset_models import SimplicialSubsetRequest
from jacobian.math.topology.simplicial_sets.subset_tools import TOOLS


def _delta_one_prefix():
    return standard_simplex(1, 2)


def test_subset_operation_catalog_example_and_json_request() -> None:
    operation_id = "topology.simplicial_set.subset.from_degree_families.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = SimplicialSubsetRequest.model_validate(tool.examples[0].input)
    assert tool.run(request).inclusion.source.sets[0] == ("(0)",)

    tool = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    result = tool.run(
        tool.request_type.model_validate(json.loads(request.model_dump_json()))
    )
    result = result.model_dump(mode="json")
    assert result["inclusion"]["maps"] == [[0], [0], [0]]
    assert (
        SimplicialSubsetPrefix.model_validate(result).inclusion.source
        == tool.run(request).inclusion.source
    )
