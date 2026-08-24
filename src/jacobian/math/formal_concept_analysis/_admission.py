"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.formal_concept_analysis._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "formal_context.objects.derivation.compute",
        AdmissionDecision.KEEP,
        "exact object derivation A' = {m : every g in A has m} with standard FCA empty-set convention",
    ),
    OperationAdmission(
        "formal_context.attributes.derivation.compute",
        AdmissionDecision.KEEP,
        "exact attribute derivation B' = {g : every m in B is possessed by g} with standard FCA empty-set convention",
    ),
    OperationAdmission(
        "formal_context.objects.closure.compute",
        AdmissionDecision.KEEP,
        "exact object closure A'' = (A')' with added objects and closed status",
    ),
    OperationAdmission(
        "formal_context.concept.from_objects.compute",
        AdmissionDecision.KEEP,
        "exact concept construction (A'', A') from an object subset",
    ),
    OperationAdmission(
        "formal_context.concept.from_attributes.compute",
        AdmissionDecision.KEEP,
        "exact concept construction (B', B'') from an attribute subset",
    ),
    OperationAdmission(
        "formal_context.concepts.enumerate.compute",
        AdmissionDecision.KEEP,
        "exact complete concept enumeration via NextClosure over attribute intents",
    ),
    OperationAdmission(
        "formal_context.concept_lattice.compute",
        AdmissionDecision.KEEP,
        "exact concept lattice with partial order, cover relation, top, and bottom",
    ),
    OperationAdmission(
        "implication_system.closure.compute",
        AdmissionDecision.KEEP,
        "exact least fixed point and replayable lineage under a bounded finite implication presentation",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
