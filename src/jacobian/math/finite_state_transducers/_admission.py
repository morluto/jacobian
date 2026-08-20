"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_state_transducers._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "transducer.relation.path.replay.compute",
        AdmissionDecision.KEEP,
        "exact path replay over a rational transducer relation",
    ),
    OperationAdmission(
        "transducer.subsequential.compose.compute",
        AdmissionDecision.KEEP,
        "exact composition of bounded subsequential transducers",
    ),
    OperationAdmission(
        "transducer.subsequential.run.compute",
        AdmissionDecision.KEEP,
        "exact subsequential transducer run with complete output",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
