"""Minimum generalized exact-cover tests."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.exact_cover import (
    ExactCoverRow,
    GeneralizedExactCoverInstance,
    minimum_generalized_exact_cover,
)


def _instance(
    rows: tuple[tuple[str, tuple[str, ...]], ...],
) -> GeneralizedExactCoverInstance:
    return GeneralizedExactCoverInstance(
        primary_items=("p", "q"),
        secondary_items=(),
        rows=tuple(ExactCoverRow(row_id=name, items=items) for name, items in rows),
    )


def test_minimum_cover_beats_first_larger_cover() -> None:
    result = minimum_generalized_exact_cover(
        _instance(
            (
                ("a-p", ("p",)),
                ("b-q", ("q",)),
                ("z-both", ("p", "q")),
            )
        )
    )
    assert result.status == "EXACT"
    assert result.selected_row_ids == ("z-both",)
    assert result.lower_bound == result.upper_bound == 1


def test_infeasibility_requires_exhaustion() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=("p",), secondary_items=(), rows=()
    )
    assert minimum_generalized_exact_cover(instance).status == "INFEASIBLE"


def test_node_limit_without_incumbent_is_an_operational_failure() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        minimum_generalized_exact_cover(
            _instance((("a-p", ("p",)), ("b-q", ("q",)))),
            search_node_limit=1,
        )


def test_node_limit_with_incumbent_returns_witness_backed_bounds() -> None:
    result = minimum_generalized_exact_cover(
        _instance((("a-both", ("p", "q")), ("b-p", ("p",)), ("c-q", ("q",)))),
        search_node_limit=2,
    )
    assert result.status == "BOUNDED"
    assert result.selected_row_ids == ("a-both",)
    assert result.lower_bound == result.upper_bound == 1
