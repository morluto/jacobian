"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

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
)
