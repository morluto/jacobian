"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.groups.actions._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "group_action.element_cycles.compute",
        AdmissionDecision.KEEP,
        "exact cycle decomposition, cycle lengths, cycle type, and fixed "
        "points of one bounded permutation group element",
    ),
    OperationAdmission(
        "group_action.subset.canonicalize",
        AdmissionDecision.KEEP,
        "exact canonical subset image with a generated-group transporter and "
        "orbit-stabilizer sizes under one bounded finite permutation action",
    ),
    OperationAdmission(
        "group_action.cycle_index.compute",
        AdmissionDecision.KEEP,
        "exact cycle-index polynomial as cycle-type multiplicity data",
    ),
    OperationAdmission(
        "group_action.burnside_count.compute",
        AdmissionDecision.KEEP,
        "exact Burnside orbit count with per-element fixed-point contributions",
    ),
    OperationAdmission(
        "group_action.polya_inventory.compute",
        AdmissionDecision.KEEP,
        "exact Pólya enumeration inventory polynomial for bounded colours",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
