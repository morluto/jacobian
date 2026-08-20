"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.topology._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "topology.simplicial_complex.canonicalize",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "topology.simplicial_complex.chain_complex.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "topology.simplicial_homology.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "topology.simplicial_homology.integral.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "topology.simplicial_complex.f_vector.compute",
        AdmissionDecision.KEEP,
        "exact f-vector, h-vector, and Euler characteristic of a simplicial complex",
    ),
    OperationAdmission(
        "topology.simplicial_complex.link.compute",
        AdmissionDecision.KEEP,
        "exact link of a simplex with maximal facets of the link complex",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
