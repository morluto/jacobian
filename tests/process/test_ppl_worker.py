"""Process-boundary behavior for exact PPL linear optimization."""

from collections.abc import Callable
from fractions import Fraction
from time import monotonic
from typing import cast

import pytest

from jacobian._execution import bind_request_deadline, request_execution
from jacobian.math.optimization import _ppl_process, _ppl_worker
from jacobian.math.optimization._ppl import ExactLinearOutcome


def _source() -> tuple[
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]:
    return (Fraction(1),), ((Fraction(1),),), (Fraction(1),)


def test_ppl_worker_returns_exact_optimal_witness() -> None:
    objective, coefficients, rhs = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        result = _ppl_process.solve_standard_form_process(
            objective, coefficients, rhs, maximum_result_digits=128
        )
    assert result.status == "OPTIMAL"
    assert result.point == (Fraction(1),)
    assert result.dual == (Fraction(1),)


def test_ppl_worker_batches_independent_exact_programs_in_one_process() -> None:
    objective, coefficients, rhs = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        results = _ppl_process.solve_standard_form_batch_process(
            (
                (objective, coefficients, rhs),
                ((Fraction(-1),), coefficients, rhs),
            ),
            maximum_result_digits=128,
        )
    assert tuple(result.status for result in results) == ("OPTIMAL", "OPTIMAL")
    assert results[0].point == results[1].point == (Fraction(1),)
    assert results[0].dual == (Fraction(1),)
    assert results[1].dual == (Fraction(-1),)


def test_ppl_worker_stops_before_later_program_after_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def first_is_infeasible(
        _objective: tuple[Fraction, ...],
        _coefficients: tuple[tuple[Fraction, ...], ...],
        _rhs: tuple[Fraction, ...],
    ) -> ExactLinearOutcome:
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail("the worker must not solve after an infeasibility witness")
        return ExactLinearOutcome(status="INFEASIBLE", witness=(Fraction(1),))

    monkeypatch.setattr(_ppl_worker, "solve_standard_form", first_is_infeasible)
    encoded = _ppl_worker._solve_program_batch((_source(), _source()))

    assert calls == 1
    assert [outcome["status"] for outcome in encoded] == ["INFEASIBLE"]


def test_ppl_worker_accepts_infeasible_batch_prefix() -> None:
    objective, coefficients, _rhs = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        results = _ppl_process.solve_standard_form_batch_process(
            (
                (objective, coefficients, (Fraction(-1),)),
                (objective, coefficients, (Fraction(1),)),
            ),
            maximum_result_digits=128,
        )

    assert len(results) == 1
    assert results[0].status == "INFEASIBLE"
    assert results[0].witness


def test_ppl_worker_rejects_incomplete_nonterminal_batch_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def incomplete(*args: object, **kwargs: object) -> object:
        decoder = cast(Callable[[object], object], kwargs["decode_result"])
        return decoder(
            {
                "protocol_version": 1,
                "outcomes": [
                    {
                        "status": "OPTIMAL",
                        "point": [["1", "1"]],
                        "dual": [["1", "1"]],
                        "witness": [],
                        "ray": [],
                    }
                ],
            }
        )

    monkeypatch.setattr(_ppl_process, "run_checked_worker_process", incomplete)
    source = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        with pytest.raises(ValueError, match="invalid outcome batch prefix"):
            _ppl_process.solve_standard_form_batch_process(
                (source, source), maximum_result_digits=128
            )


def test_ppl_worker_rejects_malformed_protocol_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed(*args: object, **kwargs: object) -> object:
        decoder = cast(Callable[[object], object], kwargs["decode_result"])
        return decoder(
            {
                "protocol_version": 1,
                "outcomes": [
                    {
                        "status": "OPTIMAL",
                        "point": [["1", "1"]],
                        "dual": [],
                        "witness": [],
                        "ray": [],
                    }
                ],
            }
        )

    monkeypatch.setattr(_ppl_process, "run_checked_worker_process", malformed)
    objective, coefficients, rhs = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        with pytest.raises(ValueError, match="malformed vector"):
            _ppl_process.solve_standard_form_process(
                objective, coefficients, rhs, maximum_result_digits=128
            )


def test_ppl_worker_bounds_rationals_before_integer_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def oversized(*args: object, **kwargs: object) -> object:
        decoder = cast(Callable[[object], object], kwargs["decode_result"])
        return decoder(
            {
                "protocol_version": 1,
                "outcomes": [
                    {
                        "status": "OPTIMAL",
                        "point": [["1" * 129, "1"]],
                        "dual": [["1", "1"]],
                        "witness": [],
                        "ray": [],
                    }
                ],
            }
        )

    def unexpected_parse(value: str) -> int:
        pytest.fail("an oversized rational must be rejected before integer parsing")

    monkeypatch.setattr(_ppl_process, "run_checked_worker_process", oversized)
    monkeypatch.setattr(_ppl_process, "parse_canonical_integer", unexpected_parse)
    objective, coefficients, rhs = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        with pytest.raises(ValueError, match="malformed rational"):
            _ppl_process.solve_standard_form_process(
                objective, coefficients, rhs, maximum_result_digits=128
            )


def test_ppl_worker_recomputes_deadline_after_payload_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian._execution import OperationExecutionTimeoutError

    encoded = False

    def encode(value: object) -> bytes:
        nonlocal encoded
        encoded = True
        return b"{}"

    def unexpected_worker(*args: object, **kwargs: object) -> None:
        pytest.fail("an expired request must not launch PPL")

    objective, coefficients, rhs = _source()
    started = monotonic()
    deadline = started + 10
    with request_execution(started):
        bind_request_deadline(deadline)
        monkeypatch.setattr(_ppl_process, "encode_strict_json", encode)
        monkeypatch.setattr(_ppl_process, "monotonic", lambda: deadline + 1)
        monkeypatch.setattr(
            _ppl_process, "run_checked_worker_process", unexpected_worker
        )
        with pytest.raises(OperationExecutionTimeoutError, match="before PPL"):
            _ppl_process.solve_standard_form_process(
                objective, coefficients, rhs, maximum_result_digits=128
            )
    assert encoded
