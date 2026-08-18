"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "metric_space.ball.compute",
        AdmissionDecision.NATIVE_ONLY,
        "direct row filter on a caller-supplied finite distance matrix",
        native_symbol="jacobian.math.finite_metric_spaces.ball",
    ),
    OperationAdmission(
        "metric_space.gromov_hyperbolicity.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "metric_space.profile.compute",
        AdmissionDecision.KEEP,
        "one complete exact metric profile whose mutually bound fields form a reusable invariant family",
    ),
)
