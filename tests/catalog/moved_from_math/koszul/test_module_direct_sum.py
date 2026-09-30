"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/koszul/test_module_direct_sum.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_catalog_declares_the_reusable_direct_sum_postcondition() -> None:
    assert "homological.koszul.module_direct_sum.compute" in {
        tool.operation_id for tool in BUILTIN_TOOLS
    }
