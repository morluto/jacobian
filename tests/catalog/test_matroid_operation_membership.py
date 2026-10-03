"""Catalog membership check: no duplicate matroid rank operation."""

from __future__ import annotations


class TestCatalogAdmission:
    def test_duplicate_rank_operation_not_registered(self) -> None:
        """prime_field.matrix.rank covers full-ground-set rank; no duplicate."""
        from jacobian.catalog.builtins import BUILTIN_TOOLS

        ids = [t.operation_id for t in BUILTIN_TOOLS]
        assert "matroid.rank.compute" not in ids
        assert "matroid.closure.compute" in ids
