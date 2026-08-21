"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.posets._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "poset.finite.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.linear_extensions.count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.mobius_function.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.width.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.lower_closure.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.upper_closure.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.dual.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.induced_subposet.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "poset.closure.compute",
        AdmissionDecision.KEEP,
        "exact lower or upper closure of a bounded poset subset",
    ),
    OperationAdmission(
        "poset.zeta_transform.compute",
        AdmissionDecision.KEEP,
        "exact incidence-algebra zeta transform on a bounded poset",
    ),
    OperationAdmission(
        "poset.incidence_convolution.compute",
        AdmissionDecision.KEEP,
        "exact bounded incidence-algebra convolution",
    ),
    OperationAdmission(
        "poset.antichain_profile.compute",
        AdmissionDecision.KEEP,
        "exact bounded antichain profile and maximum witnesses",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
