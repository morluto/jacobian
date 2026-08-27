"""Exact Sidon extension-profile kernel."""

from __future__ import annotations

from jacobian.math.combinatorics._sidon_extension_models import (
    SidonExtensionCandidateResult,
    SidonExtensionObstruction,
    SidonExtensionProfileRequest,
    SidonExtensionProfileResult,
    _candidate_obstruction,
    _ordered_difference_pairs,
)


def compute_sidon_extension_profile(
    request: SidonExtensionProfileRequest,
) -> SidonExtensionProfileResult:
    """Partition candidates into admissible and rejected.

    For each candidate x, check whether A plus x is Sidon by computing
    all ordered differences and checking for collisions. If a collision
    is found, record the repeated difference and the two source pairs.
    """
    candidates = request.candidate_elements

    source_diffs = _ordered_difference_pairs(request.source_elements)

    admissible: list[str] = []
    rejected: list[SidonExtensionCandidateResult] = []

    for x in candidates:
        obstruction_data = _candidate_obstruction(
            request.source_elements,
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
                        repeated_difference=str(difference),
                        pair_a=(str(pair_a[0]), str(pair_a[1])),
                        pair_b=(str(pair_b[0]), str(pair_b[1])),
                    ),
                )
            )
        else:
            admissible.append(x)

    return SidonExtensionProfileResult(
        source_elements=request.source_elements,
        candidate_elements=request.candidate_elements,
        admissible=tuple(admissible),
        rejected=tuple(rejected),
    )


__all__ = ["compute_sidon_extension_profile"]
