"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_game_theory._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "game_theory.best_response.compute",
        AdmissionDecision.DROP,
        "misnamed pure maximin row calculation that is not a best response without an opponent strategy",
    ),
    OperationAdmission(
        "game_theory.nash_equilibrium.compute",
        AdmissionDecision.KEEP,
        "exact primal-dual linear programming returns a complete equilibrium witness for every bounded finite zero-sum game",
    ),
    OperationAdmission(
        "game.deterministic_terminal.solve",
        AdmissionDecision.KEEP,
        "exact threshold attractors return the complete all-position minimax profile and bound optimal stationary witnesses for a bounded owned arena",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
