"""Exact Sidon extension-profile kernel."""

from __future__ import annotations

from collections import Counter

from jacobian.math.combinatorics._sidon_extension_models import (
    SidonExtensionCandidateResult,
    SidonExtensionObstruction,
    SidonExtensionProfileRequest,
    SidonExtensionProfileResult,
)


def compute_sidon_extension_profile(
    request: SidonExtensionProfileRequest,
) -> SidonExtensionProfileResult:
    """Partition candidates into admissible and rejected.

    For each candidate x, check whether A ∪ {x} is Sidon by computing
    all ordered differences and checking for collisions. If a collision
    is found, record the repeated difference and the two source pairs.
    """
    source = [int(e) for e in request.source_elements]
    candidates = [int(e) for e in request.candidate_elements]

    # Pre-compute the source ordered-difference profile
    source_diffs: dict[int, list[tuple[int, int]]] = {}
    for i, a in enumerate(source):
        for j, b in enumerate(source):
            if i != j:
                diff = a - b
                if diff not in source_diffs:
                    source_diffs[diff] = []
                source_diffs[diff].append((a, b))

    admissible: list[str] = []
    rejected: list[SidonExtensionCandidateResult] = []

    for x in candidates:
        x_str = str(x)
        # Build the ordered-difference profile of A ∪ {x}
        # We need to check all ordered differences of A ∪ {x}
        # The union has elements source + [x]
        union = source + [x]

        diff_map: dict[int, list[tuple[int, int]]] = {}
        found_obstruction = False
        obstruction: SidonExtensionObstruction | None = None

        for i, a in enumerate(union):
            for j, b in enumerate(union):
                if i == j:
                    continue
                diff = a - b
                if diff not in diff_map:
                    diff_map[diff] = []
                diff_map[diff].append((a, b))

        # Check for repeated differences
        for diff, pairs in diff_map.items():
            if len(pairs) > 1:
                # Found a collision
                pair_a = pairs[0]
                pair_b = pairs[1]
                obstruction = SidonExtensionObstruction(
                    candidate=x_str,
                    repeated_difference=str(diff),
                    pair_a=(str(pair_a[0]), str(pair_a[1])),
                    pair_b=(str(pair_b[0]), str(pair_b[1])),
                )
                found_obstruction = True
                break

        if found_obstruction:
            rejected.append(
                SidonExtensionCandidateResult(
                    candidate=x_str,
                    is_admissible=False,
                    obstruction=obstruction,
                )
            )
        else:
            admissible.append(x_str)

    return SidonExtensionProfileResult(
        source_elements=request.source_elements,
        candidate_elements=request.candidate_elements,
        admissible=admissible,
        rejected=rejected,
    )


__all__ = ["compute_sidon_extension_profile"]
