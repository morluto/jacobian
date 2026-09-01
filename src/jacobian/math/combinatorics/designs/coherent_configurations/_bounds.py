"""Request-scoped admission bounds for coherent-configuration analysis."""

from __future__ import annotations

from jacobian.math.combinatorics.designs.coherent_configurations.values import (
    MAX_ANALYSIS_WORK,
    CoherentConfigurationInput,
)


class CoherentConfigurationAdmissionError(ValueError):
    """A coherent-configuration analysis exceeds its execution envelope."""


def require_analysis_admission(source: CoherentConfigurationInput) -> None:
    """Reject predicted analysis work before cubic expansion."""
    point_count = len(source.points)
    relation_count = len(source.relation_ids)
    work = 4 * relation_count**2 * point_count**3
    if work > MAX_ANALYSIS_WORK:
        raise CoherentConfigurationAdmissionError(
            "coherent-configuration analysis exceeds the work budget"
        )


__all__ = [
    "CoherentConfigurationAdmissionError",
    "require_analysis_admission",
]
