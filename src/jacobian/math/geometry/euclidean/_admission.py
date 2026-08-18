"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

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
