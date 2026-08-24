"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.analysis._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "analysis.real_function.point_enclosure.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "interval.compute.enclosure",
        AdmissionDecision.KEEP,
        "distinct rigorous bounded enclosure for composable user-defined expressions",
    ),
    OperationAdmission(
        "interval.expression.box_enclosure.compute",
        AdmissionDecision.KEEP,
        "distinct source-bound uniform enclosure over a complete rational box",
    ),
    OperationAdmission(
        "interval.expression.second_jet_enclosure.compute",
        AdmissionDecision.KEEP,
        "distinct source-bound uniform value, gradient, and Hessian enclosure over a rational box",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
