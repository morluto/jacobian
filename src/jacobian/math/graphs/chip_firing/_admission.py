"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.chip_firing._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.chip_firing.laplacian.compute",
        AdmissionDecision.KEEP,
        "exact graph Laplacian with degree vector and labelled axes",
    ),
    OperationAdmission(
        "graph.chip_firing.reduced_laplacian.compute",
        AdmissionDecision.KEEP,
        "exact reduced Laplacian with sink row/column deleted",
    ),
    OperationAdmission(
        "graph.chip_firing.fire_vertex.compute",
        AdmissionDecision.KEEP,
        "exact chip-firing action with vertex degree transfer",
    ),
    OperationAdmission(
        "graph.chip_firing.fire_vector.compute",
        AdmissionDecision.KEEP,
        "exact integer firing-vector action with degree preservation",
    ),
    OperationAdmission(
        "graph.chip_firing.stabilize.compute",
        AdmissionDecision.KEEP,
        "exact stabilization with odometer via least-action algorithm",
    ),
    OperationAdmission(
        "graph.chip_firing.parallel_step.compute",
        AdmissionDecision.KEEP,
        "one simultaneous legal-firing state transform",
    ),
    OperationAdmission(
        "graph.chip_firing.q_reduced.compute",
        AdmissionDecision.KEEP,
        "q-reduced canonical normal form with exact firing vector",
    ),
    OperationAdmission(
        "graph.chip_firing.canonical_divisor.compute",
        AdmissionDecision.KEEP,
        "exact graph canonical divisor K(v) = deg(v) - 2",
    ),
    OperationAdmission(
        "graph.chip_firing.critical_group.compute",
        AdmissionDecision.KEEP,
        "critical group invariant factors via SNF of reduced Laplacian",
    ),
    OperationAdmission(
        "graph.chip_firing.abel_jacobi.compute",
        AdmissionDecision.KEEP,
        "Abel-Jacobi coordinates in the cokernel of the reduced Laplacian",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
