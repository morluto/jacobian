"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.additive_combinatorics._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "additive.ordered_difference_profile.compute",
        AdmissionDecision.KEEP,
        "one complete exact ordered-difference profile of a bounded integer-vector set retaining every source pair",
    ),
    OperationAdmission(
        "additive.direct_sum_predicate.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "additive.energy.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of additive.representation_profile.compute",
    ),
    OperationAdmission(
        "additive.representation_profile.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "additive.sumset_cardinality.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of additive.representation_profile.compute",
    ),
    OperationAdmission(
        "additive.subset_sum.residue_profile.compute",
        AdmissionDecision.KEEP,
        "one complete indexed-subset multiplicity profile in a finite cyclic group; unlike an integer-sum profile, its work and output depend on the modulus rather than the integer sum span",
    ),
    OperationAdmission(
        "additive.subset_sum.profile.compute",
        AdmissionDecision.KEEP,
        "one complete source-bound profile of sums and multiplicities over indexed at-most-once selections, which pair-sum composition cannot retain",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
