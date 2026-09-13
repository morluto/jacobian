"""Finite exact sequences and recurrence solving.

Canonical public sequence values and operations live in
``jacobian.math.number_theory.sequences.core``.
"""

from jacobian.math.number_theory.sequences.core import (
    AutocorrelationCell,
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
    aperiodic_autocorrelation,
    cyclic_autocorrelation,
)

__all__ = [
    "AutocorrelationCell",
    "AutocorrelationResult",
    "FiniteIntegerSequence",
    "FiniteRationalSequence",
    "aperiodic_autocorrelation",
    "cyclic_autocorrelation",
]
