"""Tool argument contract for phase-free Pauli addition."""

from jacobian._models import StrictModel
from jacobian.math.quantum._models import PhaseFreeQubitPauli


class PhaseFreePauliAdditionRequest(StrictModel):
    """Two phase-free Paulis on one ordered register."""

    left: PhaseFreeQubitPauli
    right: PhaseFreeQubitPauli
