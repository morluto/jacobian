"""Acceptance tests for bounded polynomial-map inverse synthesis."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from types import SimpleNamespace
from typing import Any

import pytest

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.polynomials import _support as polynomial_support


def _term(coefficient: int, exponents: list[int]) -> dict[str, Any]:
    return {
        "coefficient": {"num": str(coefficient), "den": "1"},
        "exponents": exponents,
    }


def _triangular_forward() -> dict[str, Any]:
    return {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x", "y"],
        "coordinates": [
            {"terms": [_term(1, [1, 0]), _term(1, [0, 2])]},
            {"terms": [_term(1, [0, 1])]},
        ],
    }


def _request(
    *,
    degree: int,
    timeout_ms: int = 10_000,
    max_unknowns: int = 64,
    explicit_support: list[list[list[int]]] | None = None,
) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id="polynomial.map.inverse.candidate_synthesize",
        input={
            "forward_map": _triangular_forward(),
            "source_variables": ["x", "y"],
            "target_variables": ["u", "v"],
            "inverse_degree_bound": degree,
            "support_mode": (
                "EXPLICIT" if explicit_support is not None else "FULL_TOTAL_DEGREE"
            ),
            "explicit_support": explicit_support,
            "solver": "sympy.solve",
            "limits": {
                "timeout_ms": timeout_ms,
                "max_inverse_degree": 4,
                "max_composition_degree": 32,
                "max_unknown_coefficients": max_unknowns,
                "max_coefficient_equations": 512,
                "max_residual_terms": 1024,
            },
        },
    )


def test_triangular_automorphism_is_found_and_verified(
    authorized_complete_runtime,
) -> None:
    result = authorized_complete_runtime.core.capabilities.invoke(_request(degree=2))

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "FOUND"
    assert result.output["candidate_inverse_map"]["coordinates"] == [
        {"terms": [_term(1, [1, 0]), _term(-1, [0, 2])]},
        {"terms": [_term(1, [0, 1])]},
    ]
    assert result.output["verification_output"]["inverse_verified"] is True
    assert result.output["verification_artifact_uri"] is not None
    assert result.output["noninvertibility_proved"] is False
    assert result.output["inverse_after_forward"] == [
        {"terms": []},
        {"terms": []},
    ]
    assert result.output["forward_after_inverse"] == [
        {"terms": []},
        {"terms": []},
    ]


def test_degree_below_required_returns_bounded_no_candidate(
    attached_complete_runtime,
) -> None:
    result = attached_complete_runtime.core.capabilities.invoke(_request(degree=1))

    assert result.output["status"] == "NO_CANDIDATE_WITHIN_ANSATZ"
    assert result.output["candidate_inverse_map"] is None
    assert result.output["noninvertibility_proved"] is False


def test_redundant_explicit_ansatz_is_underdetermined(
    attached_complete_runtime,
) -> None:
    identity = {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "coordinates": [{"terms": [_term(1, [1])]}],
    }
    request = CapabilityRequest(
        capability_id="polynomial.map.inverse.candidate_synthesize",
        input={
            "forward_map": identity,
            "source_variables": ["x"],
            "target_variables": ["u"],
            "inverse_degree_bound": 1,
            "support_mode": "EXPLICIT",
            "explicit_support": [[[1], [1]]],
            "solver": "sympy.solve",
            "limits": {
                "timeout_ms": 10_000,
                "max_inverse_degree": 1,
                "max_composition_degree": 8,
                "max_unknown_coefficients": 4,
                "max_coefficient_equations": 16,
                "max_residual_terms": 16,
            },
        },
    )

    result = attached_complete_runtime.core.capabilities.invoke(request)

    assert result.output["status"] == "UNDERDETERMINED"
    assert result.output["candidate_inverse_map"] is None
    assert "free parameters" in result.output["verification_failure"]


def test_zero_timeout_and_unknown_budget_are_explicit(
    attached_complete_runtime,
) -> None:

    timeout = attached_complete_runtime.core.capabilities.invoke(
        _request(degree=2, timeout_ms=0)
    )
    exhausted = attached_complete_runtime.core.capabilities.invoke(
        _request(degree=2, max_unknowns=1)
    )

    assert timeout.execution.status is ExecutionStatus.TIMEOUT
    assert timeout.output["status"] == "TIMEOUT"
    assert exhausted.output["status"] == "BUDGET_EXHAUSTED"


class _StubbornProcess:
    def __init__(self) -> None:
        self.alive = True
        self.events: list[str] = []
        self.join_timeouts: list[float] = []

    def start(self) -> None:
        self.events.append("start")

    def join(self, timeout: float | None = None) -> None:
        if timeout is None:
            raise AssertionError("polynomial solver join must remain bounded")
        self.events.append("join")
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        self.alive = False


def _install_stubborn_solver(
    monkeypatch: pytest.MonkeyPatch, process: _StubbornProcess
) -> None:
    def make_queue(*, maxsize: int) -> object:
        assert maxsize == 1
        return object()

    def make_process(*, target: Any, args: tuple[Any, ...]) -> _StubbornProcess:
        assert callable(target)
        assert args
        return process

    context = SimpleNamespace(Queue=make_queue, Process=make_process)

    def get_context(method: str) -> SimpleNamespace:
        assert method == "spawn"
        return context

    monkeypatch.setattr(polynomial_support.multiprocessing, "get_context", get_context)


def test_timeout_kills_stubborn_solver_without_unbounded_wait(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _StubbornProcess()
    _install_stubborn_solver(monkeypatch, process)

    result = attached_complete_runtime.core.capabilities.invoke(_request(degree=2))

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "TIMEOUT"
    assert result.output["candidate_inverse_map"] is None
    assert result.output["noninvertibility_proved"] is False
    assert process.events == [
        "start",
        "join",
        "terminate",
        "join",
        "kill",
        "join",
    ]
    assert 0 < process.join_timeouts[0] <= 10.0
    assert process.join_timeouts[1:] == [1.0, 1.0]
    assert all(timeout > 0 and isfinite(timeout) for timeout in process.join_timeouts)


def test_unknown_solver_is_unsupported_without_truth_claim(
    attached_complete_runtime,
) -> None:
    payload = deepcopy(_request(degree=2).input)
    payload["solver"] = "unknown.exact_solver"

    result = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.inverse.candidate_synthesize",
            input=payload,
        )
    )

    assert result.output["status"] == "UNSUPPORTED"
    assert result.output["candidate_inverse_map"] is None
    assert result.output["noninvertibility_proved"] is False


def test_full_support_and_coefficient_order_are_deterministic(
    attached_complete_runtime,
) -> None:
    first = attached_complete_runtime.core.capabilities.invoke(_request(degree=2))
    second = attached_complete_runtime.core.capabilities.invoke(_request(degree=2))

    assert first.output["ansatz"] == second.output["ansatz"]
    assert (
        first.output["coefficient_equations"] == second.output["coefficient_equations"]
    )
    supports = first.output["ansatz"]["coordinate_supports"]
    assert supports[0] == [
        [2, 0],
        [1, 1],
        [1, 0],
        [0, 2],
        [0, 1],
        [0, 0],
    ]
    assert first.output["ansatz"]["coefficient_symbols"][0] == [
        "c_0_0",
        "c_0_1",
        "c_0_2",
        "c_0_3",
        "c_0_4",
        "c_0_5",
    ]


@pytest.mark.parametrize(
    "mutation",
    ["variable_order", "coefficient_domain"],
)
def test_ring_mismatches_fail_closed(attached_complete_runtime, mutation: str) -> None:
    payload = deepcopy(_request(degree=2).input)
    if mutation == "variable_order":
        payload["source_variables"] = ["y", "x"]
    else:
        payload["forward_map"]["domain"] = "RR"

    result = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.inverse.candidate_synthesize",
            input=payload,
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()


def test_corrupted_found_candidate_does_not_verify(
    authorized_complete_runtime,
) -> None:
    synthesized = authorized_complete_runtime.core.capabilities.invoke(
        _request(degree=2)
    )
    corrupted = deepcopy(synthesized.output["candidate_inverse_map"])
    corrupted["coordinates"][0]["terms"][1]["coefficient"]["num"] = "-2"
    checked = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.map.inverse.verify",
            input={
                "forward_map": _triangular_forward(),
                "inverse_map": corrupted,
                "source_variables": ["x", "y"],
                "target_variables": ["u", "v"],
            },
        )
    )
    assert checked.output["inverse_verified"] is False
