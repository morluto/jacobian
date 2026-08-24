"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.probability._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "probability.finite_distribution.condition.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.finite_distribution.convolution.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "probability.finite_distribution.event_probability.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of a caller-supplied finite distribution",
    ),
    OperationAdmission(
        "probability.finite_distribution.pushforward.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.finite_distribution.raw_moment.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.gaussian_polynomial.moment.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.graph_reliability.all_terminal.compute",
        AdmissionDecision.KEEP,
        "the complete connected-spanning-subgraph profile and its uniform exact probability are distinct from two-terminal reliability and require bounded exhaustive computation",
    ),
    OperationAdmission(
        "probability.graph_reliability.connection_probability.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.joint.mutual_information.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
