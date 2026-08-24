"""Owner-local admission decisions for built-in polytope operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polytope._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polytope.rational.support.compute",
        AdmissionDecision.KEEP,
        "exact bounded support value and complete exposed vertex face over a "
        "canonical labelled rational V-polytope; a reusable convex-geometry "
        "primitive distinct from volume, separation, and H/V conversion",
    ),
    OperationAdmission(
        "polytope.facets.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded source-bound value: complete canonical facet "
        "incidences are reusable by face, duality, and local-polytope operations "
        "and cannot be recovered from a volume or one-point separation result",
    ),
    OperationAdmission(
        "polytope.volume.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value with material "
        "computational and reliability leverage: exact rational volume "
        "is a composable building block for Ehrhart positivity and "
        "lattice point counting",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
