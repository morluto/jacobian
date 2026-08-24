"""Admission decisions for exact multicommodity-flow operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.multicommodity_flow._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "network.multicommodity_flow.profile.compute",
        AdmissionDecision.KEEP,
        "complete exact source-bound conservation and capacity profile of a canonical sparse commodity-by-edge tensor; distinct from single-commodity flow optimization and reusable by future feasibility, congestion, and routing operations",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
