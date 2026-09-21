"""Tests for exact permutation-valued lattice-gauge path holonomy."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.gauge import (
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    PermutationLabel,
)
from jacobian.math.gauge._models import HolonomyRequest
from jacobian.math.gauge._tools import TOOLS
from jacobian.math.gauge.operations import path_holonomy


def _perm(image: list[int]) -> PermutationLabel:
    return PermutationLabel(degree=len(image), image=tuple(image))


def _triangle_field() -> GaugeField:
    lattice = GaugeLattice(
        vertices=("a", "b", "c"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="ca", tail="c", head="a"),
        ),
    )
    cycle = _perm([1, 2, 0])
    return GaugeField(
        lattice=lattice,
        degree=3,
        edge_labels=(
            GaugeFieldEdgeLabel(edge_id="ab", label=cycle),
            GaugeFieldEdgeLabel(edge_id="bc", label=cycle),
            GaugeFieldEdgeLabel(edge_id="ca", label=cycle),
        ),
    )


def _path(*steps: tuple[str, bool]) -> OrientedGaugePath:
    return OrientedGaugePath(
        steps=tuple(GaugePathStep(edge_id=e, forward=f) for e, f in steps)
    )


class TestKnownAnswer:
    def test_triangle_loop_holonomy_is_identity(self) -> None:
        result = path_holonomy(
            _triangle_field(), _path(("ab", True), ("bc", True), ("ca", True))
        )
        assert tuple(result.holonomy.image) == (0, 1, 2)
        assert (result.start, result.end) == ("a", "a")
        assert [c.edge_id for c in result.contributions] == ["ab", "bc", "ca"]

    def test_single_edge_backward_is_inverse(self) -> None:
        field = _triangle_field()
        forward = path_holonomy(field, _path(("ab", True)))
        backward = path_holonomy(field, _path(("ab", False)))
        assert tuple(forward.holonomy.image) == (1, 2, 0)
        assert tuple(backward.holonomy.image) == (2, 0, 1)
        assert (backward.start, backward.end) == ("b", "a")

    def test_open_path_endpoints(self) -> None:
        result = path_holonomy(_triangle_field(), _path(("ab", True), ("bc", True)))
        # (0 1 2)^2 = (0 2 1).
        assert tuple(result.holonomy.image) == (2, 0, 1)
        assert (result.start, result.end) == ("a", "c")


class TestBoundary:
    def test_trusted_empty_path_is_rejected_before_kernel_construction(self) -> None:
        empty = OrientedGaugePath.model_construct(steps=())

        with pytest.raises(OperationDomainValidationError) as raised:
            path_holonomy(_triangle_field(), empty)

        assert raised.value.errors()[0]["type"] == "lattice_gauge.holonomy.empty_path"

    def test_trivial_two_edge_identity_field(self) -> None:
        lattice = GaugeLattice(
            vertices=("a", "b"),
            edges=(GaugeEdge(edge_id="e", tail="a", head="b"),),
        )
        field = GaugeField(
            lattice=lattice,
            degree=2,
            edge_labels=(GaugeFieldEdgeLabel(edge_id="e", label=_perm([0, 1])),),
        )
        result = path_holonomy(field, _path(("e", True), ("e", False)))
        assert tuple(result.holonomy.image) == (0, 1)
        assert (result.start, result.end) == ("a", "a")

    def test_noncommuting_labels_order_matters(self) -> None:
        lattice = GaugeLattice(
            vertices=("a", "b", "c"),
            edges=(
                GaugeEdge(edge_id="ab", tail="a", head="b"),
                GaugeEdge(edge_id="bc", tail="b", head="c"),
            ),
        )
        # (0 1) then (0 2): 0->1->1? compose left-to-right.
        field = GaugeField(
            lattice=lattice,
            degree=3,
            edge_labels=(
                GaugeFieldEdgeLabel(edge_id="ab", label=_perm([1, 0, 2])),
                GaugeFieldEdgeLabel(edge_id="bc", label=_perm([2, 1, 0])),
            ),
        )
        result = path_holonomy(field, _path(("ab", True), ("bc", True)))
        # 0 ->1 ->1; 1 ->0 ->2; 2 ->2 ->0  =>  [1,2,0].
        assert tuple(result.holonomy.image) == (1, 2, 0)


class TestAdversarial:
    def test_disconnected_path_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            path_holonomy(_triangle_field(), _path(("ab", True), ("ca", True)))

    def test_unknown_edge_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            path_holonomy(_triangle_field(), _path(("zz", True)))

    def test_native_rejects_non_field(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            path_holonomy("not-a-field", _path(("ab", True)))  # type: ignore[arg-type]


class TestDefiningInvariant:
    def test_reverse_traversal_gives_inverse(self) -> None:
        field = _triangle_field()
        steps = (("ab", True), ("bc", True), ("ca", True))
        forward = path_holonomy(field, _path(*steps))
        reversed_steps = tuple((e, not f) for e, f in reversed(steps))
        backward = path_holonomy(field, _path(*reversed_steps))
        assert (backward.start, backward.end) == (forward.end, forward.start)
        degree = forward.holonomy.degree
        identity = tuple(range(degree))
        composed = tuple(
            backward.holonomy.image[forward.holonomy.image[i]] for i in range(degree)
        )
        assert composed == identity

    def test_concatenation_splits_product(self) -> None:
        field = _triangle_field()
        whole = path_holonomy(field, _path(("ab", True), ("bc", True), ("ca", True)))
        first = path_holonomy(field, _path(("ab", True), ("bc", True)))
        second = path_holonomy(
            field,
            _path(
                ("ca", True),
            ),
        )
        degree = whole.holonomy.degree
        composed = tuple(
            second.holonomy.image[first.holonomy.image[i]] for i in range(degree)
        )
        assert composed == tuple(whole.holonomy.image)


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        request = HolonomyRequest(
            field=_triangle_field(),
            path=_path(("ab", True), ("bc", True), ("ca", True)),
        )
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "lattice_gauge.holonomy.compute"
        )
        assert tool.run(request) == path_holonomy(
            _triangle_field(), _path(("ab", True), ("bc", True), ("ca", True))
        )

    def test_operation_is_published(self) -> None:
        assert "lattice_gauge.holonomy.compute" in {tool.operation_id for tool in TOOLS}
