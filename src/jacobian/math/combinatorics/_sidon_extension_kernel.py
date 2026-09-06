"""Exact Sidon extension-profile kernel."""

from __future__ import annotations

from types import MappingProxyType

from jacobian.math.combinatorics._sidon_extension_models import (
    SidonExtensionCandidateResult,
    SidonExtensionObstruction,
    SidonExtensionProfileResult,
    _candidate_obstruction,
    _require_extension_work_budget,
    _SidonExtensionAdmissionPlan,
    _validate_source_is_sidon,
)


def _sidon_extension_admission_plan(
    source_elements: tuple[int, ...],
    candidate_elements: tuple[int, ...],
) -> _SidonExtensionAdmissionPlan:
    """Admit one profile request and prepare reusable kernel data."""

    _require_extension_work_budget(len(source_elements), len(candidate_elements))
    source_differences = _validate_source_is_sidon(source_elements)
    return _SidonExtensionAdmissionPlan(
        source_differences=MappingProxyType(source_differences),
    )


def compute_sidon_extension_profile(
    source_elements: tuple[int, ...],
    candidate_elements: tuple[int, ...],
) -> SidonExtensionProfileResult:
    """Partition candidates into admissible and rejected.

    For each candidate x, check whether A plus x is Sidon by computing
    canonical positive differences and checking for collisions. If a collision
    is found, record the repeated difference and the two source pairs.
    """
    candidates = candidate_elements

    admission_plan = _sidon_extension_admission_plan(
        source_elements,
        candidate_elements,
    )
    source_diffs = admission_plan.source_differences

    admissible: list[int] = []
    rejected: list[SidonExtensionCandidateResult] = []

    for x in candidates:
        obstruction_data = _candidate_obstruction(
            source_elements,
            source_diffs,
            x,
        )
        if obstruction_data is not None:
            difference, pair_a, pair_b = obstruction_data
            rejected.append(
                SidonExtensionCandidateResult(
                    candidate=x,
                    is_admissible=False,
                    obstruction=SidonExtensionObstruction(
                        candidate=x,
                        repeated_difference=difference,
                        pair_a=pair_a,
                        pair_b=pair_b,
                    ),
                )
            )
        else:
            admissible.append(x)

    return SidonExtensionProfileResult._from_kernel(
        source_elements=source_elements,
        candidate_elements=candidate_elements,
        admissible=tuple(admissible),
        rejected=tuple(rejected),
    )


__all__ = ["compute_sidon_extension_profile"]
