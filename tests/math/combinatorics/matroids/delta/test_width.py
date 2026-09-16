"""Tests for the declared delta-matroid width operation (#1954)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidWidthRequest,
    DeltaMatroidWidthResult,
)
from jacobian.math.combinatorics.matroids.delta._tools import _width
from jacobian.math.combinatorics.matroids.delta.operations import width
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def _matroid(
    ground: tuple[str, ...], feasible: tuple[tuple[int, ...], ...]
) -> FiniteDeltaMatroid:
    return FiniteDeltaMatroid.model_construct(ground=ground, feasible=feasible)


WIDE = _matroid(("a", "b"), ((), (0,), (0, 1), (1,)))
MATROID = _matroid(("a", "b"), (((0,), (1,))))


class TestKnownAnswer:
    def test_wide_family_width_two(self) -> None:
        assert width(WIDE) == 2

    def test_width_zero_family(self) -> None:
        assert width(MATROID) == 0

    def test_catalog_path_agrees(self) -> None:
        result = _width(DeltaMatroidWidthRequest(delta_matroid=WIDE))
        assert isinstance(result, DeltaMatroidWidthResult)
        assert result.width == 2
        assert result.delta_matroid == WIDE


class TestBoundaryDegenerate:
    def test_single_empty_feasible_set(self) -> None:
        single = _matroid(("a",), ((),))
        assert width(single) == 0

    def test_singleton_ground(self) -> None:
        single = _matroid(("a",), ((), (0,)))
        assert width(single) == 1


class TestAdversarial:
    def test_non_canonical_row_order_still_measures_correctly(self) -> None:
        # Row order is transport, not mathematics: sizes {1, 1} give width 0.
        forged = FiniteDeltaMatroid.model_construct(
            ground=("a", "b"),
            feasible=(((1,), (0,))),
        )
        assert width(forged) == 0

    def test_forged_out_of_range_row_rejected(self) -> None:
        forged = FiniteDeltaMatroid.model_construct(
            ground=("a",),
            feasible=(((), (5,))),
        )
        with pytest.raises(OperationDomainValidationError):
            width(forged)

    def test_catalog_path_rejects_forged_carrier(self) -> None:
        forged = FiniteDeltaMatroid.model_construct(ground=("a",), feasible=())
        with pytest.raises(OperationDomainValidationError):
            _width(DeltaMatroidWidthRequest.model_construct(delta_matroid=forged))


class TestDefiningInvariant:
    @pytest.mark.parametrize(
        ("matroid", "expected"),
        [
            (WIDE, 2),
            (MATROID, 0),
            (_matroid(("a",), ((),)), 0),
            (_matroid(("a",), ((), (0,))), 1),
        ],
    )
    def test_width_is_max_minus_min(
        self, matroid: FiniteDeltaMatroid, expected: int
    ) -> None:
        sizes = [len(row) for row in matroid.feasible]
        assert width(matroid) == max(sizes) - min(sizes) == expected


class TestNativeCatalogParity:
    @pytest.mark.parametrize("matroid", [WIDE, MATROID, _matroid(("a",), ((), (0,)))])
    def test_native_matches_catalog(self, matroid: FiniteDeltaMatroid) -> None:
        assert _width(DeltaMatroidWidthRequest(delta_matroid=matroid)).width == width(
            matroid
        )
