"""Regression tests binding independence-number results to their source graph."""

from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs import _independence_z3 as z3_backend
from jacobian.math.graphs import independence as independence_models
from jacobian.math.graphs._independence_z3 import solve_independence_number
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
    IndependenceNumberResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _path_graph() -> SimpleUndirectedGraph:
    return _graph(("a", "b", "c"), (("a", "b"), ("b", "c")))


def test_budget_exposes_only_enforced_limits() -> None:
    assert (
        "max_solver_calls"
        not in IndependenceNumberBudget.model_json_schema()["properties"]
    )
    with pytest.raises(ValidationError):
        IndependenceNumberBudget(max_solver_calls=1)


def test_producer_results_replay_against_source() -> None:
    empty = solve_independence_number(IndependenceNumberRequest(graph=_graph((), ())))
    assert empty.status == "EXACT"
    assert empty.optimum_value == 0
    assert "result_schema_version" not in empty.model_dump(mode="json")
    assert IndependenceNumberResult.model_validate(empty.model_dump()) == empty

    complete = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b"), (("a", "b"),)))
    )
    assert complete.status == "EXACT"
    assert complete.optimum_value == 1
    assert complete.witness_vertices in (("a",), ("b",))

    disconnected = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b", "c", "d"), (("a", "b"),)))
    )
    assert disconnected.optimum_value == 3

    nonunique = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b"), ()))
    )
    assert nonunique.optimum_value == 2
    assert nonunique.witness_vertices == ("a", "b")


def test_witness_membership_and_edge_freedom_are_enforced() -> None:
    request = IndependenceNumberRequest(graph=_path_graph())
    result = solve_independence_number(request)
    assert result.optimum_value == 2
    dumped = result.model_dump()

    nonexistent = copy.deepcopy(dumped)
    nonexistent["witness_vertices"] = ["a", "zz"]
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(nonexistent)

    added_edge = copy.deepcopy(dumped)
    added_edge["graph"]["edges"] = [
        ["a", "b"],
        ["b", "c"],
        ["a", "c"],
    ]
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(added_edge)

    foreign_graph = copy.deepcopy(dumped)
    foreign_graph["graph"] = {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "d"]],
    }
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(foreign_graph)


def test_field_mutations_are_rejected() -> None:
    request = IndependenceNumberRequest(graph=_path_graph())
    result = solve_independence_number(request)
    dumped = result.model_dump()

    forged_order = copy.deepcopy(dumped)
    forged_order["order"] = 7
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(forged_order)

    unsorted_witness = copy.deepcopy(dumped)
    if len(unsorted_witness["witness_vertices"]) > 1:
        unsorted_witness["witness_vertices"] = list(
            reversed(unsorted_witness["witness_vertices"])
        )
        with pytest.raises(ValidationError):
            IndependenceNumberResult.model_validate(unsorted_witness)

    cardinality_break = copy.deepcopy(dumped)
    cardinality_break["incumbent_value"] = result.incumbent_value + 1
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(cardinality_break)

    bound_break = copy.deepcopy(dumped)
    bound_break["lower_bound"] = result.lower_bound + 1
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(bound_break)

    false_optimum = copy.deepcopy(dumped)
    false_optimum["status"] = "UNKNOWN"
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(false_optimum)

    unknown_with_optimum = copy.deepcopy(dumped)
    unknown_with_optimum["status"] = "UNKNOWN"
    unknown_with_optimum["optimum_value"] = result.optimum_value
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(unknown_with_optimum)


def test_matching_via_edge_intersection_composition() -> None:
    """The Atlas move dual -> edge-intersection -> independence stays bound.

    The triangle hypergraphs of K4 and K5 have pairwise-intersecting
    hyperedges, so their edge-intersection graphs are complete and the
    matching numbers replay as one.
    """

    from itertools import combinations

    for order in (4, 5):
        triangles = [frozenset(c) for c in combinations(range(order), 3)]
        count = len(triangles)
        pairs = tuple(
            (f"t{i}", f"t{j}")
            for i in range(count)
            for j in range(i + 1, count)
            if triangles[i] & triangles[j]
        )
        graph = _graph(tuple(f"t{i}" for i in range(count)), pairs)
        result = solve_independence_number(IndependenceNumberRequest(graph=graph))
        assert result.status == "EXACT"
        assert result.optimum_value == 1


def test_forged_nonmaximum_optimum_is_rejected_by_source_replay() -> None:
    feasible = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b", "c"), ()))
    )
    dumped = feasible.model_dump()
    dumped["graph"]["edges"] = [["a", "b"], ["b", "c"]]
    dumped["witness_vertices"] = ["a"]
    for field in (
        "optimum_value",
        "incumbent_value",
        "lower_bound",
        "upper_bound",
    ):
        dumped[field] = 1
    dumped["termination_reason"] = "OPTIMUM_ESTABLISHED"
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(dumped)


def test_edge_removal_invalidating_the_claimed_optimum_is_rejected() -> None:
    triangle = solve_independence_number(
        IndependenceNumberRequest(
            graph=_graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
        )
    )
    assert triangle.optimum_value == 1
    dumped = triangle.model_dump()
    dumped["graph"]["edges"] = []
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(dumped)


def test_exhausted_exact_replay_budget_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IndependenceNumberRequest(
        graph=_graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d")))
    )
    result = solve_independence_number(request)
    assert result.status == "EXACT"
    monkeypatch.setattr(independence_models, "_EXACT_REPLAY_SEARCH_NODES", 1)
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(result.model_dump())


def _matching_graph(edges: int) -> SimpleUndirectedGraph:
    return _graph(
        tuple(f"m{index:02d}" for index in range(2 * edges)),
        tuple((f"m{2 * index:02d}", f"m{2 * index + 1:02d}") for index in range(edges)),
    )


def test_matching_produced_exact_result_revalidates() -> None:
    """The reported rejection case: 15 disjoint edges, order 30, optimum 15.

    The producing solve must emit an ``EXACT`` payload whose serialized
    output loads back as an equal ``IndependenceNumberResult``, so every
    produced result revalidates against the retained source graph.
    """

    graph = _matching_graph(15)
    result = solve_independence_number(IndependenceNumberRequest(graph=graph))
    assert result.status == "EXACT"
    assert result.optimum_value == 15
    assert result.order == 30
    assert result.termination_reason == "OPTIMUM_ESTABLISHED"
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_structured_disjoint_graphs_replay_within_tight_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Component decomposition keeps disjoint structures inside tiny budgets.

    Ten disjoint five-cycles have optimum 20, isolated vertices are forced
    without branching, and the whole replay stays deterministic and exact
    well below the default search-node envelope that the naive search
    exceeded on a 30-vertex matching.
    """

    cycle_vertices = []
    cycle_edges = []
    for component in range(10):
        ids = [f"c{component}_{index}" for index in range(5)]
        cycle_vertices += ids
        for index in range(5):
            left, right = sorted((ids[index], ids[(index + 1) % 5]))
            cycle_edges.append((left, right))
    cycles = _graph(tuple(cycle_vertices), tuple(cycle_edges))
    mixed = _graph(
        ("i0", "i1", "i2", "p0", "p1", "p2"),
        (("p0", "p1"), ("p1", "p2")),
    )

    produced = []
    for graph, expected in ((cycles, 20), (mixed, 5), (_matching_graph(15), 15)):
        result = solve_independence_number(IndependenceNumberRequest(graph=graph))
        assert result.status == "EXACT"
        assert result.optimum_value == expected
        produced.append(result)

    monkeypatch.setattr(independence_models, "_EXACT_REPLAY_SEARCH_NODES", 512)
    for result in produced:
        assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_unreplayable_solver_optimum_demotes_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solver optimum the bounded replay cannot certify claims no optimum."""

    request = IndependenceNumberRequest(
        graph=_graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d")))
    )
    monkeypatch.setattr(independence_models, "_EXACT_REPLAY_SEARCH_NODES", 1)
    result = solve_independence_number(request)
    assert result.status == "UNKNOWN"
    assert result.optimum_value is None
    assert result.termination_reason == "REPLAY_INCOMPLETE"
    assert result.lower_bound == result.incumbent_value
    assert result.upper_bound == 4
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result


class _StubBound:
    """A Z3 objective bound carrying a concrete integer estimate."""

    def __init__(self, value: int) -> None:
        self._value = value

    def as_long(self) -> int:
        return self._value


class _StubVariable:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubModel:
    def eval(self, variable: _StubVariable, model_completion: bool) -> bool:
        assert model_completion is True
        return variable.name.endswith("_0")


class _StubObjective:
    """Open Optimize bounds: proven incumbent 1, unproven estimate 2."""

    def lower(self) -> _StubBound:
        return _StubBound(1)

    def upper(self) -> _StubBound:
        return _StubBound(2)


class _StubOptimizer:
    def set(self, *args: object, **kwargs: object) -> None:
        return None

    def add(self, *args: object) -> None:
        return None

    def maximize(self, expression: object) -> _StubObjective:
        return _StubObjective()

    def check(self) -> int:
        return _StubZ3.sat

    def model(self) -> _StubModel:
        return _StubModel()


class _StubZ3:
    """Minimal Optimize surface: sat with one selected vertex, open bounds."""

    sat = 1
    unsat = -1

    def Bool(self, name: str) -> _StubVariable:  # noqa: N802 (z3 API mirror)
        return _StubVariable(name)

    @staticmethod
    def Or(*expressions: object) -> object:  # noqa: N802 (z3 API mirror)
        return expressions

    @staticmethod
    def Not(expression: object) -> object:  # noqa: N802 (z3 API mirror)
        return expression

    @staticmethod
    def Sum(expressions: list[object]) -> object:  # noqa: N802 (z3 API mirror)
        return expressions

    @staticmethod
    def If(  # noqa: N802 (z3 API mirror)
        condition: object, then: object, otherwise: object
    ) -> object:
        assert condition is not None and otherwise is not None
        return then

    @staticmethod
    def is_true(value: object) -> bool:
        return value is True

    @staticmethod
    def is_int_value(value: object) -> bool:
        return isinstance(value, _StubBound)

    Optimize = _StubOptimizer


def test_sat_with_open_objective_bounds_reports_order_as_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete ``sat`` optimize still validates against the order bound.

    Z3 can return ``sat`` with a feasible incumbent while its objective
    bounds remain open, so the producing fallthrough must normalize the
    upper bound to the independently safe graph order instead of echoing
    the optimizer's intermediate estimate through ``model_construct``.
    """

    monkeypatch.setattr(z3_backend, "z3", _StubZ3())
    result = solve_independence_number(IndependenceNumberRequest(graph=_path_graph()))
    assert result.status == "UNKNOWN"
    assert result.optimum_value is None
    assert result.incumbent_value == 1
    assert result.lower_bound == 1
    assert result.upper_bound == result.order == 3
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_replay_deadline_is_charged_to_the_request_envelope() -> None:
    """An elapsed deadline rejects the claim instead of opening a fresh budget."""

    graph = _graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d")))
    with pytest.raises(ValueError):
        independence_models._replay_exact_optimum(
            graph, 2, deadline=time.monotonic() - 1.0
        )
    independence_models._replay_exact_optimum(graph, 2, deadline=time.monotonic() + 60)


def test_producer_replay_shares_the_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay work cannot extend the producing solve past its wall budget.

    The solver establishes the optimum inside the budget, but the shared
    deadline has elapsed by replay time, so the typed ``UNKNOWN`` demotion
    keeps the whole call inside the one admitted envelope and still
    revalidates against the retained source graph.
    """

    request = IndependenceNumberRequest(
        graph=_graph(("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d"))),
        resource_budget=IndependenceNumberBudget(wall_seconds=5),
    )
    monkeypatch.setattr(
        independence_models,
        "time",
        SimpleNamespace(monotonic=lambda: float("inf")),
    )
    result = solve_independence_number(request)
    assert result.status == "UNKNOWN"
    assert result.optimum_value is None
    assert result.termination_reason == "REPLAY_INCOMPLETE"
    assert result.upper_bound == 4
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_result_headroom_admission_rejects_oversized_echo() -> None:
    """A huge identifier is rejected before solving, not by dispatch after."""

    oversized = "v" * (6 * 1024 * 1024)
    with pytest.raises(ValidationError):
        IndependenceNumberRequest(graph=_graph((oversized,), ()))


def test_large_labels_near_boundary_stay_admitted_and_transportable() -> None:
    """Admission tracks the predicted serialized result, not a coarse cap.

    One hundred twenty-eight 30k-character identifiers predict a doubled
    echo of roughly 7.7 MB, which stays inside the canonical output limit:
    the solve runs, the exact payload serializes under the limit, and the
    round trip revalidates.
    """

    labels = tuple(f"n{index:03d}" + "x" * 30_000 for index in range(128))
    request = IndependenceNumberRequest(graph=_graph(labels, ()))
    result = solve_independence_number(request)
    assert result.status == "EXACT"
    assert result.optimum_value == 128
    assert result.witness_vertices == tuple(sorted(labels))
    encoded = len(encode_strict_json(result.model_dump(mode="json")))
    assert encoded <= CanonicalLimits().max_output_bytes
    assert IndependenceNumberResult.model_validate(result.model_dump()) == result


def test_incomplete_tight_upper_bound_is_rejected() -> None:
    """An UNKNOWN upper bound must bind to an independently safe quantity.

    Forging an edgeless three-vertex graph down to witness ``("a",)`` with
    ``upper_bound = 1`` claims an incumbent gap that no source-bound check
    authenticates; validation now requires the graph order itself.
    """

    dumped = solve_independence_number(
        IndependenceNumberRequest(graph=_graph(("a", "b", "c"), ()))
    ).model_dump()
    dumped["status"] = "UNKNOWN"
    dumped["optimum_value"] = None
    dumped["witness_vertices"] = ["a"]
    dumped["incumbent_value"] = 1
    dumped["lower_bound"] = 1
    dumped["upper_bound"] = 1
    dumped["termination_reason"] = "SOLVER_UNKNOWN"
    with pytest.raises(ValidationError):
        IndependenceNumberResult.model_validate(dumped)
