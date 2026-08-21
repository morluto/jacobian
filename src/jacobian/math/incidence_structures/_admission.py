"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.incidence_structures._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "incidence.matrix.compute",
        AdmissionDecision.KEEP,
        "exact labelled 0/1 incidence matrix with point rows and block columns",
    ),
    OperationAdmission(
        "incidence.degree_profile.compute",
        AdmissionDecision.KEEP,
        "per-point and per-block degree profiles with total incidence count",
    ),
    OperationAdmission(
        "incidence.containment_profiles.compute",
        AdmissionDecision.KEEP,
        "t-subset containment multiplicity profiles with histogram",
    ),
    OperationAdmission(
        "incidence.intersections.compute",
        AdmissionDecision.KEEP,
        "block pairwise intersection subsets and size histogram",
    ),
    OperationAdmission(
        "incidence.dual.compute",
        AdmissionDecision.KEEP,
        "dual incidence structure swapping points and block IDs",
    ),
    OperationAdmission(
        "incidence.complement.compute",
        AdmissionDecision.KEEP,
        "block complement incidence structure preserving block IDs",
    ),
    OperationAdmission(
        "incidence.restriction.compute",
        AdmissionDecision.KEEP,
        "point/block restriction preserving block IDs",
    ),
    OperationAdmission(
        "incidence.derived_residual.compute",
        AdmissionDecision.KEEP,
        "derived and residual incidence structures at a point",
    ),
    OperationAdmission(
        "incidence.levi_graph.compute",
        AdmissionDecision.KEEP,
        "labelled bipartite Levi incidence graph",
    ),
    OperationAdmission(
        "incidence.gram.compute",
        AdmissionDecision.KEEP,
        "Gram concordance matrix N N^T or N^T N",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
