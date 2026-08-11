from __future__ import annotations

import pytest

from jacobian.canonical import canonicalize_json
from jacobian.sat_smt.cvc5 import _parse_worker_output
from jacobian.sat_smt.cvc5_worker import CVC5_WORKER_PROTOCOL


def test_cvc5_worker_result_rejects_non_string_solver_status() -> None:
    stdout = canonicalize_json(
        {
            "protocol": CVC5_WORKER_PROTOCOL,
            "solver_status": [],
            "proof_written": False,
            "alethe_hole_count": 0,
        }
    )

    with pytest.raises(ValueError, match="invalid cvc5 worker result"):
        _parse_worker_output(stdout)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "protocol": CVC5_WORKER_PROTOCOL,
            "solver_status": "SATISFIABLE",
            "proof_written": True,
            "alethe_hole_count": 0,
        },
        {
            "protocol": CVC5_WORKER_PROTOCOL,
            "solver_status": "UNSATISFIABLE",
            "proof_written": False,
            "alethe_hole_count": None,
        },
        {
            "protocol": CVC5_WORKER_PROTOCOL,
            "solver_status": "UNKNOWN",
            "proof_written": False,
            "alethe_hole_count": 0,
        },
    ),
)
def test_cvc5_worker_result_rejects_inconsistent_proof_shape(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid cvc5 worker result"):
        _parse_worker_output(canonicalize_json(payload))
