"""Tests for the pseudomanifold boundary operation (#1821)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import SimplicialComplexRequest
from jacobian.math.topology._structural import (
    BoundaryRequest,
    BoundaryResult,
    compute_boundary,
)
from jacobian.math.topology._tools import compute_boundary_entry


def _request(vertices: list[str], facets: list[list[str]]) -> BoundaryRequest:
    return BoundaryRequest(
        complex=SimplicialComplexRequest.model_validate(
            {"vertices": vertices, "facets": facets}
        )
    )


class TestKnownAnswer:
    def test_interval_boundary_is_two_endpoints(self) -> None:
        result = compute_boundary(_request(["a", "b", "c"], [["a", "b"], ["b", "c"]]))
        assert isinstance(result, BoundaryResult)
        assert result.dimension == 1
        assert result.boundary_ridges == (("a",), ("c",))
        assert result.component_count == 2

    def test_disk_boundary_is_its_rim(self) -> None:
        result = compute_boundary(_request(["a", "b", "c"], [["a", "b", "c"]]))
        assert result.boundary_ridges == (("a", "b"), ("a", "c"), ("b", "c"))
        assert result.component_count == 1
        assert result.boundary_complex.dimension == 1


class TestBoundaryDegenerate:
    def test_single_edge_boundary(self) -> None:
        result = compute_boundary(_request(["a", "b"], [["a", "b"]]))
        assert result.boundary_ridges == (("a",), ("b",))
        assert result.component_count == 2


class TestAdversarial:
    def test_circle_has_no_boundary(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_boundary(
                _request(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
            )

    def test_non_pseudomanifold_rejected(self) -> None:
        # Three triangles sharing one edge: ridge incidence three.
        with pytest.raises(OperationDomainValidationError):
            compute_boundary(
                _request(
                    ["a", "b", "c", "d", "e"],
                    [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"]],
                )
            )

    def test_nonpure_complex_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_boundary(_request(["a", "b", "c"], [["a", "b"], ["c"]]))

    def test_catalog_entry_rejects_closed(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_boundary_entry(
                _request(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
            )


class TestDefiningInvariant:
    def test_every_boundary_ridge_has_incidence_one(self) -> None:
        request = _request(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])
        result = compute_boundary(request)
        facet_sets = [frozenset(f) for f in request.complex.facets]
        for ridge in result.boundary_ridges:
            ridge_set = frozenset(ridge)
            assert len(ridge) == result.dimension  # codimension-one face
            assert sum(ridge_set <= facet for facet in facet_sets) == 1

    def test_ridges_generate_closure(self) -> None:
        request = _request(["a", "b", "c"], [["a", "b", "c"]])
        result = compute_boundary(request)
        assert set(map(tuple, result.boundary_complex.maximal_simplices)) == set(
            result.boundary_ridges
        )


class TestNativeCatalogParity:
    def test_entry_matches_kernel(self) -> None:
        request = _request(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        assert compute_boundary_entry(request) == compute_boundary(request)
