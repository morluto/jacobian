"""Process-boundary behavior for exact PPL linear optimization."""

from collections.abc import Callable
from fractions import Fraction
from time import monotonic
from typing import cast

import pytest

from jacobian._execution import bind_request_deadline, request_execution
from jacobian.math.optimization import _ppl_process


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
        result = _ppl_process.solve_standard_form_process(objective, coefficients, rhs)
    assert result.status == "OPTIMAL"
    assert result.point == (Fraction(1),)
    assert result.dual == (Fraction(1),)


def test_ppl_worker_rejects_malformed_protocol_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed(*args: object, **kwargs: object) -> object:
        decoder = cast(Callable[[object], object], kwargs["decode_result"])
        return decoder(
            {
                "protocol_version": 1,
                "status": "OPTIMAL",
                "point": [["1", "1"]],
                "dual": [],
                "witness": [],
                "ray": [],
            }
        )

    monkeypatch.setattr(_ppl_process, "run_checked_worker_process", malformed)
    objective, coefficients, rhs = _source()
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() + 10)
        with pytest.raises(ValueError, match="malformed vector"):
            _ppl_process.solve_standard_form_process(objective, coefficients, rhs)
