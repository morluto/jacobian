"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_topology._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "topology.beat_points.compute",
        AdmissionDecision.KEEP,
        "complete beat-point witness family for a bounded finite topology",
    ),
    OperationAdmission(
        "topology.connected_components.compute",
        AdmissionDecision.KEEP,
        "complete connected component partition of a bounded finite topology",
    ),
    OperationAdmission(
        "topology.is_continuous.compute",
        AdmissionDecision.KEEP,
        "exact continuity verdict between bounded finite topologies",
    ),
    OperationAdmission(
        "topology.specialization_preorder.compute",
        AdmissionDecision.KEEP,
        "complete specialization preorder of a bounded finite topology",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
