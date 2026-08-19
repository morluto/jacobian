"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "combinatorial_map.faces.compute",
        AdmissionDecision.KEEP,
        "exact face-orbit family with replayable dart-successor permutation",
    ),
    OperationAdmission(
        "combinatorial_map.euler_characteristic.compute",
        AdmissionDecision.KEEP,
        "exact per-component and total Euler characteristic under the disconnected-surface convention",
    ),
    OperationAdmission(
        "combinatorial_map.orientable_genus.compute",
        AdmissionDecision.KEEP,
        "exact nonnegative per-component and total orientable genus of the supplied embedding",
    ),
    OperationAdmission(
        "combinatorial_map.orientation_reverse.compute",
        AdmissionDecision.KEEP,
        "exact orientation reversal with an induced face bijection",
    ),
    OperationAdmission(
        "combinatorial_map.connected_components.compute",
        AdmissionDecision.KEEP,
        "exact vertex, dart, and face component partition",
    ),
    OperationAdmission(
        "combinatorial_map.dual.compute",
        AdmissionDecision.KEEP,
        "exact embedded dual that preserves bridges as dual loops and parallel edges distinctly",
    ),
    OperationAdmission(
        "combinatorial_map.vertex_face_incidence.compute",
        AdmissionDecision.KEEP,
        "exact finite vertex-face incidence structure with multiplicity",
    ),
)
