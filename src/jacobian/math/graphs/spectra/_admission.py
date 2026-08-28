"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.spectra._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.spectrum.adjacency.characteristic_polynomial.compute",
        AdmissionDecision.KEEP,
        "exact monic adjacency characteristic polynomial over QQ, composable with graph spectral and chromatic-spectral comparisons",
    ),
    OperationAdmission(
        "graph.spectrum.laplacian.characteristic_polynomial.compute",
        AdmissionDecision.KEEP,
        "exact monic Laplacian characteristic polynomial over QQ, composable with graph spectral invariants",
    ),
    OperationAdmission(
        "graph.spectrum.adjacency.compute",
        AdmissionDecision.KEEP,
        "exact adjacency eigenvalue/multiplicity multiset bound to its "
        "retained source graph; independently supplied claims have an exact "
        "bounded owner verifier",
    ),
    OperationAdmission(
        "graph.spectrum.laplacian.compute",
        AdmissionDecision.KEEP,
        "exact Laplacian eigenvalue/multiplicity multiset bound to its "
        "retained source graph; independently supplied claims have an exact "
        "bounded owner verifier",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
