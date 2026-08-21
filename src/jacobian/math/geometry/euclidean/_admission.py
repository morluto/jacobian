"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.geometry.euclidean._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "geometry.euclidean.angle_equality.compute",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.euclidean.segment_ratio.compute",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
    OperationAdmission(
        "geometry.euclidean.triangle_similarity.compute",
        AdmissionDecision.DROP,
        "elementary exact formula without material leverage over direct Python",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
