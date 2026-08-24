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
    OperationAdmission(
        "topology.simplicial_complex.star.compute",
        AdmissionDecision.KEEP,
        "exact closed star of a simplex: all facets containing the simplex",
    ),
    OperationAdmission(
        "topology.simplicial_complex.deletion.compute",
        AdmissionDecision.KEEP,
        "exact vertex deletion: the induced subcomplex on the undeclared vertices",
    ),
    OperationAdmission(
        "topology.simplicial_complex.skeleton.compute",
        AdmissionDecision.KEEP,
        "exact k-skeleton subcomplex of a simplicial complex",
    ),
    OperationAdmission(
        "topology.simplicial_complex.join.compute",
        AdmissionDecision.KEEP,
        "exact join of two simplicial complexes on disjoint vertex sets",
    ),
    OperationAdmission(
        "topology.simplicial_complex.barycentric_subdivision.compute",
        AdmissionDecision.KEEP,
        "exact barycentric subdivision (order complex) of a simplicial complex",
    ),
    OperationAdmission(
        "topology.simplicial_complex.pseudomanifold.decide",
        AdmissionDecision.KEEP,
        "exact pseudomanifold decision with closed/boundary distinction",
    ),
    OperationAdmission(
        "topology.simplicial_complex.shelling.check",
        AdmissionDecision.KEEP,
        "exact shelling order checker for a submitted facet permutation",
    ),
    OperationAdmission(
        "topology.simplicial_complex.elementary_collapse.check",
        AdmissionDecision.KEEP,
        "exact elementary collapse: verify a free face and remove it with its coface",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
