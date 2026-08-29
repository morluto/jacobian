"""Exact Sidon extension-profile kernel."""

from __future__ import annotations

from types import MappingProxyType

from jacobian.math.combinatorics._difference_set_models import (
    _difference_set_validation_error,
)
from jacobian.math.combinatorics._sidon_extension_models import (
    MAX_EXTENSION_RESULT_BYTES,
    SidonExtensionCandidateResult,
    SidonExtensionObstruction,
    SidonExtensionProfileResult,
    _candidate_obstruction,
    _CandidateObstruction,
    _maximum_result_bytes,
    _require_extension_work_budget,
    _require_source_profile_memory_budget,
    _SidonExtensionAdmissionPlan,
    _validate_source_is_sidon,
)


def _sidon_extension_admission_plan(
    source_elements: tuple[str, ...],
    candidate_elements: tuple[str, ...],
) -> _SidonExtensionAdmissionPlan:
    """Admit one profile request and prepare reusable kernel data."""

    _require_extension_work_budget(len(source_elements), len(candidate_elements))
    _require_source_profile_memory_budget(source_elements)
    source_differences = _validate_source_is_sidon(source_elements)
    result_bytes = _maximum_result_bytes(source_elements, candidate_elements)
    candidate_obstructions: tuple[_CandidateObstruction | None, ...] | None = None
    if result_bytes > MAX_EXTENSION_RESULT_BYTES:
        candidate_obstructions = tuple(
            _candidate_obstruction(source_elements, source_differences, candidate)
            for candidate in candidate_elements
        )
        result_bytes = _maximum_result_bytes(
            source_elements, candidate_elements, candidate_obstructions
        )
    if result_bytes > MAX_EXTENSION_RESULT_BYTES:
        raise _difference_set_validation_error(
            "combinatorics.sidon_extension_result_budget",
            "Sidon extension result exceeds the canonical output budget",
        )
    return _SidonExtensionAdmissionPlan(
        source_differences=MappingProxyType(source_differences),
        candidate_obstructions=candidate_obstructions,
    )


def compute_sidon_extension_profile(
    source_elements: tuple[str, ...],
    candidate_elements: tuple[str, ...],
) -> SidonExtensionProfileResult:
    """Partition candidates into admissible and rejected.

    For each candidate x, check whether A plus x is Sidon by computing
    all ordered differences and checking for collisions. If a collision
    is found, record the repeated difference and the two source pairs.
    """
    candidates = candidate_elements

    admission_plan = _sidon_extension_admission_plan(
        source_elements,
        candidate_elements,
    )
    source_diffs = admission_plan.source_differences
    candidate_obstructions = admission_plan.candidate_obstructions

    admissible: list[str] = []
    rejected: list[SidonExtensionCandidateResult] = []

    for index, x in enumerate(candidates):
        if candidate_obstructions is None:
            obstruction_data = _candidate_obstruction(
                source_elements,
                source_diffs,
                x,
            )
        else:
            obstruction_data = candidate_obstructions[index]
        if obstruction_data is not None:
            difference, pair_a, pair_b = obstruction_data
            rejected.append(
                SidonExtensionCandidateResult(
                    candidate=x,
                    is_admissible=False,
                    obstruction=SidonExtensionObstruction(
                        candidate=x,
                        repeated_difference=str(difference),
                        pair_a=(str(pair_a[0]), str(pair_a[1])),
                        pair_b=(str(pair_b[0]), str(pair_b[1])),
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
