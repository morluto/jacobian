from __future__ import annotations

import random
from typing import cast

import pytest

from jacobian.math.logic._cnf import CnfCanonicalizeRequest, canonicalize_cnf
from jacobian.math.logic._sat import SatSolveRequest, solve_sat


def _threshold_3sat_request(*, timeout_ms: int) -> SatSolveRequest:
    """Return one deterministic admitted UNSAT instance near the 3-SAT threshold."""

    variable_count = 250
    clause_count = round(4.27 * variable_count)
    random_source = random.Random(3566)
    clauses: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    while len(clauses) < clause_count:
        variable_indices = sorted(random_source.sample(range(variable_count), 3))
        clause = cast(
            tuple[int, int, int],
            tuple(
                -(index + 1) if random_source.choice((False, True)) else index + 1
                for index in variable_indices
            ),
        )
        if clause in seen:
            continue
        seen.add(clause)
        clauses.append(clause)
    canonical = canonicalize_cnf(
        CnfCanonicalizeRequest(
            variable_names=tuple(f"x{index:04d}" for index in range(variable_count)),
            clauses=tuple(clauses),
        )
    ).cnf
    return SatSolveRequest(cnf=canonical, timeout_ms=timeout_ms)


@pytest.mark.scale
def test_sat_solver_finishes_near_threshold_work_with_extended_wall_time() -> None:
    result = solve_sat(_threshold_3sat_request(timeout_ms=120_000))

    assert result.outcome == "UNSAT"
    assert result.assignment is None
    assert result.source.cnf == _threshold_3sat_request(timeout_ms=1_000).cnf
