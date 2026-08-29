"""Tests for divisibility-incidence graph construction."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_graph import (
    compute_divisibility_incidence_graph,
)
from jacobian.math.number_theory._divisibility_graph_models import (
    DivisibilityIncidenceGraphRequest,
)
from jacobian.math.number_theory.operations import divisibility_incidence_graph


def test_basic() -> None:
    result = compute_divisibility_incidence_graph(
        DivisibilityIncidenceGraphRequest.model_validate(
            {"left_family": ["2", "3"], "right_family": ["6", "12", "5"]}
        )
    )
    edges = {tuple(edge) for edge in result.graph.edges}
    assert ("L0", "R0") in edges
    assert ("L0", "R1") in edges
    assert ("L1", "R0") in edges
    assert ("L1", "R1") in edges
    assert len(result.graph.edges) == 4
    assert result.left_family == ("2", "3")
    assert result.right_family == ("6", "12", "5")


def test_native_graph_accepts_canonical_values() -> None:
    result = divisibility_incidence_graph((2,), (4,))
    assert result.graph.edges == (("L0", "R0"),)


def test_request_families_are_immutable() -> None:
    request = DivisibilityIncidenceGraphRequest.model_validate(
        {"left_family": ["2"], "right_family": ["4"]}
    )

    assert request.left_family == ("2",)
    assert request.right_family == ("4",)
    with pytest.raises(AttributeError):
        cast(Any, request.left_family).append("3")


def test_no_edges() -> None:
    result = compute_divisibility_incidence_graph(
        DivisibilityIncidenceGraphRequest.model_validate(
            {"left_family": ["7"], "right_family": ["3"]}
        )
    )
    assert len(result.graph.edges) == 0


def test_bipartite() -> None:
    result = compute_divisibility_incidence_graph(
        DivisibilityIncidenceGraphRequest.model_validate(
            {"left_family": ["1", "2", "3"], "right_family": ["2", "3", "6"]}
        )
    )
    for e in result.graph.edges:
        assert e[0].startswith("L") and e[1].startswith("R")


@pytest.mark.parametrize("value", ["abc", "01", "9" * 257, 3])
def test_rejects_non_positive_canonical_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        DivisibilityIncidenceGraphRequest.model_validate(
            {"left_family": [value], "right_family": ["1"]}
        )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_native_operation_rejects_non_positive_integer(value: str) -> None:
    request = DivisibilityIncidenceGraphRequest(
        left_family=(value,), right_family=("1",)
    )
    with pytest.raises(OperationDomainValidationError, match="positive"):
        compute_divisibility_incidence_graph(request)


def test_rejects_combined_vertex_budget() -> None:
    request = DivisibilityIncidenceGraphRequest.model_validate(
        {
            "left_family": [str(value) for value in range(1, 257)],
            "right_family": ["1"],
        }
    )
    with pytest.raises(OperationDomainValidationError, match="total values"):
        compute_divisibility_incidence_graph(request)


@pytest.mark.parametrize(
    ("left_family", "right_family"),
    ((["2", "2"], ["4"]), (["2"], ["4", "4"])),
)
def test_rejects_duplicate_family_values(
    left_family: list[str], right_family: list[str]
) -> None:
    request = DivisibilityIncidenceGraphRequest.model_validate(
        {"left_family": left_family, "right_family": right_family}
    )
    with pytest.raises(OperationDomainValidationError, match="values must be unique"):
        compute_divisibility_incidence_graph(request)
