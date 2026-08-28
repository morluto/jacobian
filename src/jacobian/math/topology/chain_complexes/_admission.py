"""Owner-local admission decisions for chain complex operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.topology.chain_complexes._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "chain_complex.construct.compute",
        AdmissionDecision.KEEP,
        "finite based chain complex from differential matrices over QQ or GF(p)",
    ),
    OperationAdmission(
        "chain_complex.verify_differential.compute",
        AdmissionDecision.KEEP,
        "exact verification that d^2 = 0",
    ),
    OperationAdmission(
        "chain_complex.verify_chain_map.compute",
        AdmissionDecision.KEEP,
        "exact verification that a chain map commutes with differentials",
    ),
    OperationAdmission(
        "chain_complex.homology.compute",
        AdmissionDecision.KEEP,
        "exact homology groups of a finite chain complex",
    ),
    OperationAdmission(
        "chain_complex.mapping_cone.compute",
        AdmissionDecision.KEEP,
        "exact mapping cone of a chain map",
    ),
    OperationAdmission(
        "chain_complex.tensor_product.compute",
        AdmissionDecision.KEEP,
        "exact tensor product of two chain complexes",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
