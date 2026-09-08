"""Exact exhaustive feasibility oracles, complete ledgers and failure closure."""

import json
from itertools import combinations, combinations_with_replacement, islice, product
from time import monotonic

import pytest
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.discrepancy import FiniteSetSystem, compute_discrepancy
from jacobian.math.combinatorics.discrepancy.bounded_coloring import (
    BoundedColoringBudget,
    BoundedColoringResult,
    decide,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring._models import (
    BoundedColoringRequest,
    SatisfiableBoundedColoring,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring._process import (
    decode_reply,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring._z3 import solve
from jacobian.math.combinatorics.discrepancy.bounded_coloring.operations import (
    _admit,
    _outcome,
)


def oracle(source: FiniteSetSystem, bounds: tuple[int, ...]) -> bool:
    return any(
        all(
            abs(sum(color[index] for index in subset)) <= bound
            for subset, bound in zip(source.sets, bounds, strict=True)
        )
        for color in product((-1, 1), repeat=source.ground_set_size)
    )


def check_sat(result: BoundedColoringResult) -> None:
    assert isinstance(result.outcome, SatisfiableBoundedColoring)
    evaluation = compute_discrepancy(result.set_system, result.outcome.coloring)
    assert evaluation.signed_sums == result.outcome.signed_sums
    assert all(
        abs(total) <= bound
        for total, bound in zip(
            evaluation.signed_sums, result.absolute_bounds, strict=True
        )
    )
    negated = compute_discrepancy(
        result.set_system, tuple(-value for value in result.outcome.coloring)
    )
    assert negated.signed_sums == tuple(-value for value in result.outcome.signed_sums)
    assert BoundedColoringResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "n,sets,bounds,expected",
    [
        (0, (), (), "SATISFIABLE"),
        (0, ((),), (0,), "SATISFIABLE"),
        (4, (), (), "SATISFIABLE"),
        (1, ((0,),), (0,), "UNSATISFIABLE"),
        (1, ((0,),), (1,), "SATISFIABLE"),
        (2, ((0, 1), (0,)), (0, 1), "SATISFIABLE"),
        (2, ((0, 1), (0,)), (0, 0), "UNSATISFIABLE"),
        (3, ((0, 1), (1, 2), (0, 2)), (0, 0, 0), "UNSATISFIABLE"),
        (2, ((0, 1), (0, 1), ()), (2, 0, 0), "SATISFIABLE"),
    ],
)
def test_exact_native_fixtures(
    n: int, sets: tuple[tuple[int, ...], ...], bounds: tuple[int, ...], expected: str
) -> None:
    source = FiniteSetSystem(ground_set_size=n, sets=sets)
    result = decide(source, bounds)
    assert result.outcome.status == expected
    assert oracle(source, bounds) == (expected == "SATISFIABLE")
    if isinstance(result.outcome, SatisfiableBoundedColoring):
        check_sat(result)
    else:
        assert result.outcome.model_dump() == {"status": "UNSATISFIABLE"}


def test_all_two_row_systems_and_bound_vectors_through_four_elements() -> None:
    # Exhaustive assignment enumeration is only the independent oracle. The
    # production PB kernel is exercised directly to avoid thousands of worker
    # startups; public process execution is covered by the native/MCP fixtures.
    for n in range(5):
        subsets = tuple(
            subset for size in range(n + 1) for subset in combinations(range(n), size)
        )
        for rows in combinations_with_replacement(subsets, 2):
            source = FiniteSetSystem(ground_set_size=n, sets=rows)
            for bounds in product(*(range(len(row) + 1) for row in rows)):
                constraints = _admit(source, bounds)
                reply = solve(n, constraints, 1_000_000, monotonic() + 10)
                outcome = _outcome(source, bounds, reply)
                assert outcome.status in ("SATISFIABLE", "UNSATISFIABLE")
                assert (outcome.status == "SATISFIABLE") == oracle(source, bounds)
                if isinstance(outcome, SatisfiableBoundedColoring):
                    check_sat(
                        BoundedColoringResult(
                            set_system=source, absolute_bounds=bounds, outcome=outcome
                        )
                    )


def test_induced_edge_source_fixture() -> None:
    vertices = range(4)
    edges = tuple(combinations(vertices, 2))
    sets = tuple(
        tuple(i for i, edge in enumerate(edges) if set(edge) <= set(subset))
        for size in (3, 4)
        for subset in combinations(vertices, size)
    )
    source = FiniteSetSystem(ground_set_size=6, sets=sets)
    result = decide(source, tuple(len(row) // 3 for row in sets))
    check_sat(result)
    assert isinstance(result.outcome, SatisfiableBoundedColoring)
    for subset in sets:
        positive = sum(result.outcome.coloring[index] == 1 for index in subset)
        assert min(positive, len(subset) - positive) >= (len(subset) + 2) // 3


def test_full_carrier_incidence_and_ledger_boundary() -> None:
    source = FiniteSetSystem(ground_set_size=64, sets=(tuple(range(64)),) * 1000)
    result = decide(source, (0,) * 1000)
    check_sat(result)
    assert isinstance(result.outcome, SatisfiableBoundedColoring)
    assert result.outcome.signed_sums == (0,) * 1000


def test_thousand_distinct_dense_constraints_at_full_ground_axis() -> None:
    rows = tuple(
        tuple(v for pair in pairs for v in (2 * pair, 2 * pair + 1))
        for pairs in islice(combinations(range(32), 16), 1000)
    )
    source = FiniteSetSystem(ground_set_size=64, sets=rows)
    result = decide(source, (0,) * len(rows))
    check_sat(result)


@pytest.mark.parametrize("bounds", [(), (0, 0), (-1,), (2,), (True,)])
def test_native_invalid_bounds(bounds: tuple[int, ...]) -> None:
    with pytest.raises(OperationDomainValidationError):
        decide(FiniteSetSystem(ground_set_size=1, sets=((0,),)), bounds)


def test_empty_set_requires_zero_and_wire_axis_rejects() -> None:
    with pytest.raises(ValidationError):
        BoundedColoringRequest(
            set_system=FiniteSetSystem(ground_set_size=0, sets=((),)),
            absolute_bounds=(1,),
        )


def test_budget_counters_reject_coerced_strings() -> None:
    with pytest.raises(ValidationError):
        BoundedColoringBudget.model_validate({"wall_seconds": "15"})
    with pytest.raises(ValidationError):
        BoundedColoringBudget.model_validate({"solver_work_limit": "1000"})


def test_request_schema_states_per_set_upper_bound() -> None:
    schema = BoundedColoringRequest.model_json_schema()
    description = schema["properties"]["absolute_bounds"]["description"]
    assert "at most that set's cardinality" in description


def test_real_backend_work_exhaustion_and_expired_solver_are_claim_free() -> None:
    source = FiniteSetSystem(ground_set_size=6, sets=(tuple(range(6)),))
    result = decide(source, (0,), BoundedColoringBudget(solver_work_limit=1))
    assert result.outcome.model_dump() == {"status": "BUDGET_EXCEEDED"}
    reply = solve(6, _admit(source, (0,)), 1_000_000, monotonic() - 1)
    assert reply == {"status": "BUDGET_EXCEEDED", "coloring": None}


def test_shared_expired_context_is_an_operational_timeout() -> None:
    source = FiniteSetSystem(ground_set_size=0, sets=())
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            decide(source, ())


def test_wall_expiry_after_real_solver_discards_mathematical_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.discrepancy.bounded_coloring import operations

    now = monotonic()
    clock = iter((now, now, now + 100))
    monkeypatch.setattr(operations, "monotonic", lambda: next(clock))
    source = FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
    result = decide(source, (0,))
    assert result.outcome.model_dump() == {"status": "BUDGET_EXCEEDED"}


def test_invalid_bounds_reject_before_budget_short_circuit() -> None:
    source = FiniteSetSystem(ground_set_size=1, sets=((0,),))
    with request_execution(monotonic() - 100):
        with pytest.raises(OperationDomainValidationError, match="absolute bounds"):
            decide(source, (2,))


def test_elapsed_request_start_exhausts_own_wall_budget() -> None:
    source = FiniteSetSystem(ground_set_size=0, sets=())
    with request_execution(monotonic() - 100):
        result = decide(source, ())
    assert result.outcome.model_dump() == {"status": "BUDGET_EXCEEDED"}


def test_parsed_budget_result_still_rejects_invalid_bounds() -> None:
    source = FiniteSetSystem(ground_set_size=1, sets=((0,),))
    with pytest.raises(ValidationError, match="set size"):
        BoundedColoringResult.model_validate(
            {
                "set_system": source.model_dump(mode="json"),
                "absolute_bounds": [2],
                "outcome": {"status": "BUDGET_EXCEEDED"},
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON",
        b'{"status":"UNSATISFIABLE","coloring":[1]}',
        b'{"status":"SATISFIABLE","coloring":[0]}',
        b'{"status":"SATISFIABLE","coloring":[true]}',
        b'{"status":"SATISFIABLE","coloring":[]}',
        b'{"status":"unknown","coloring":null}',
        b'{"status":"UNSATISFIABLE","coloring":null,"extra":1}',
    ],
)
def test_malformed_worker_codec_cannot_publish_infeasibility(payload: bytes) -> None:
    assert decode_reply(payload, 1) == {"status": "EXECUTION_FAILED", "coloring": None}


def test_invalid_candidate_fails_exact_replay_without_claim() -> None:
    source = FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
    outcome = _outcome(source, (0,), {"status": "SATISFIABLE", "coloring": (1, 1)})
    assert outcome.model_dump() == {"status": "EXECUTION_FAILED"}


def test_closed_schema_excludes_witness_from_non_success() -> None:
    source = FiniteSetSystem(ground_set_size=1, sets=((0,),))
    result = decide(source, (0,)).model_dump(mode="json")
    for status in ("UNSATISFIABLE", "BUDGET_EXCEEDED", "EXECUTION_FAILED"):
        result["outcome"] = {"status": status, "coloring": [1], "signed_sums": [1]}
        with pytest.raises(SchemaValidationError):
            validate(result, BoundedColoringResult.model_json_schema())
        with pytest.raises(ValidationError):
            BoundedColoringResult.model_validate_json(json.dumps(result))


def test_real_failed_worker_execution_is_claim_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.discrepancy.bounded_coloring import _process

    # Invoke Python on an existing directory with no __main__: a real bounded
    # child process fails, rather than fabricating a solver's mathematical answer.
    monkeypatch.setattr(_process, "_WORKER", _process._WORKER.parent)
    source = FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
    result = decide(source, (0,))
    assert result.outcome.model_dump() == {"status": "EXECUTION_FAILED"}


def test_unavailable_backend_is_claim_free(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from typing import Any

    original_import = builtins.__import__

    def unavailable(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "z3":
            raise ImportError("controlled unavailable-backend test")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", unavailable)
        reply = solve(2, (((0, 1), 1, 1),), 1000, monotonic() + 10)
    assert reply == {"status": "EXECUTION_FAILED", "coloring": None}


def test_fano_plane_has_no_nonmonochromatic_two_coloring() -> None:
    source = FiniteSetSystem(
        ground_set_size=7,
        sets=(
            (0, 1, 2),
            (0, 3, 4),
            (0, 5, 6),
            (1, 3, 5),
            (1, 4, 6),
            (2, 3, 6),
            (2, 4, 5),
        ),
    )
    assert not oracle(source, (1,) * 7)
    result = decide(source, (1,) * 7)
    assert result.outcome.model_dump() == {"status": "UNSATISFIABLE"}
