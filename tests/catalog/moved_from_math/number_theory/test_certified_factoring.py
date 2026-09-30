"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/test_certified_factoring.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations


def test_operations_are_discoverable_via_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {t.operation_id for t in BUILTIN_TOOLS}
    assert "integer.factor.certified_compute" in ids
    assert "integer.primality.certificate.compute" in ids
