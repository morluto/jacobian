"""Typed contracts for the Sidon extension-profile operation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from pydantic import Field, StrictBool, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics._difference_set_models import (
    AdditiveDifferenceInteger,
    AdditiveInteger,
    _difference_set_validation_error,
)

MAX_EXTENSION_WORK = 4_000_000

_DifferencePair = tuple[int, int]
_CandidateObstruction = tuple[int, _DifferencePair, _DifferencePair]


@dataclass(frozen=True)
class _SidonExtensionAdmissionPlan:
    """Request-scoped admission data reused by the owner-local kernel."""

    source_differences: Mapping[int, _DifferencePair]


def _ordered_difference_pairs(
    elements: tuple[AdditiveInteger, ...],
) -> dict[int, _DifferencePair]:
    integers = tuple(map(int, elements))
    return {
        left - right: (left, right)
        for left in integers
        for right in integers
        if left != right
    }


def _extension_work_units(source_count: int, candidate_count: int) -> int:
    """Price source profiling and candidate-local difference checks."""

    source_pairs = source_count * (source_count - 1)
    per_candidate = 4 * source_count + 16
    return source_pairs + candidate_count * per_candidate


def _validate_source_and_candidates(
    source_elements: tuple[AdditiveInteger, ...],
    candidate_elements: tuple[AdditiveInteger, ...],
) -> None:
    if len(set(source_elements)) != len(source_elements):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "source elements must be unique",
        )
    if len(set(candidate_elements)) != len(candidate_elements):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "candidate elements must be unique",
        )
    if set(source_elements) & set(candidate_elements):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "source and candidate sets must be disjoint",
        )


def _require_extension_work_budget(
    source_count: int,
    candidate_count: int,
) -> None:
    if _extension_work_units(source_count, candidate_count) > MAX_EXTENSION_WORK:
        raise _difference_set_validation_error(
            "combinatorics.sidon_extension_work_budget",
            "Sidon extension search exceeds the bounded work budget",
        )


def _validate_source_is_sidon(
    source_elements: tuple[AdditiveInteger, ...],
) -> dict[int, _DifferencePair]:
    source_pairs = _ordered_difference_pairs(source_elements)
    if len(source_pairs) != len(source_elements) * (len(source_elements) - 1):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "source elements must form a Sidon set",
        )
    return source_pairs


def _validate_obstruction_witness(
    obstruction: SidonExtensionObstruction,
) -> None:
    pair_a = tuple(int(value) for value in obstruction.pair_a)
    pair_b = tuple(int(value) for value in obstruction.pair_b)
    difference = int(obstruction.repeated_difference)
    if (
        pair_a[0] == pair_a[1]
        or pair_b[0] == pair_b[1]
        or pair_a == pair_b
        or pair_a[0] - pair_a[1] != difference
        or pair_b[0] - pair_b[1] != difference
    ):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "obstruction pairs must witness the repeated difference",
        )


class SidonExtensionProfileRequest(StrictModel):
    """Source Sidon set A and candidate set C disjoint from A."""

    source_elements: tuple[AdditiveInteger, ...]
    candidate_elements: tuple[AdditiveInteger, ...] = Field(
        description=(
            "Unique candidates disjoint from the source. Candidate count is "
            "admitted from the exact work and result-size budgets."
        )
    )

    @model_validator(mode="after")
    def require_unique_and_disjoint(self) -> Self:
        _validate_source_and_candidates(
            self.source_elements,
            self.candidate_elements,
        )
        return self


class SidonExtensionObstruction(StrictModel):
    """One replayable repeated-difference obstruction for a rejected candidate."""

    candidate: AdditiveInteger
    repeated_difference: AdditiveDifferenceInteger
    pair_a: tuple[AdditiveInteger, AdditiveInteger]
    pair_b: tuple[AdditiveInteger, AdditiveInteger]

    @model_validator(mode="after")
    def require_repeated_difference_witness(self) -> Self:
        pair_a = tuple(int(value) for value in self.pair_a)
        pair_b = tuple(int(value) for value in self.pair_b)
        difference = int(self.repeated_difference)
        if (
            pair_a[0] == pair_a[1]
            or pair_b[0] == pair_b[1]
            or pair_a == pair_b
            or pair_a[0] - pair_a[1] != difference
            or pair_b[0] - pair_b[1] != difference
        ):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "obstruction pairs must witness the repeated difference",
            )
        return self


class SidonExtensionCandidateResult(StrictModel):
    """One candidate's admissibility or obstruction."""

    candidate: AdditiveInteger
    is_admissible: StrictBool
    obstruction: SidonExtensionObstruction | None = None

    @model_validator(mode="after")
    def require_matching_obstruction(self) -> Self:
        if self.is_admissible and self.obstruction is not None:
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "admissible candidates cannot carry an obstruction",
            )
        if not self.is_admissible and self.obstruction is None:
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "rejected candidates require an obstruction",
            )
        if (
            self.obstruction is not None
            and self.obstruction.candidate != self.candidate
        ):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "obstruction candidate must match its result row",
            )
        return self


class SidonExtensionProfileResult(StrictModel):
    """Complete partition of candidates into admissible and rejected."""

    source_elements: tuple[AdditiveInteger, ...]
    candidate_elements: tuple[AdditiveInteger, ...]
    admissible: tuple[AdditiveInteger, ...]
    rejected: tuple[SidonExtensionCandidateResult, ...]

    @model_validator(mode="after")
    def require_complete_partition(self) -> Self:
        _validate_profile_structure(
            self.source_elements,
            self.candidate_elements,
            self.admissible,
            self.rejected,
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_elements: tuple[AdditiveInteger, ...],
        candidate_elements: tuple[AdditiveInteger, ...],
        admissible: tuple[AdditiveInteger, ...],
        rejected: tuple[SidonExtensionCandidateResult, ...],
    ) -> Self:
        """Construct output already established by the owner-local kernel."""

        return cls.model_construct(
            source_elements=source_elements,
            candidate_elements=candidate_elements,
            admissible=admissible,
            rejected=rejected,
        )


def _validate_profile_structure(
    source_elements: tuple[AdditiveInteger, ...],
    candidate_elements: tuple[AdditiveInteger, ...],
    admissible: tuple[AdditiveInteger, ...],
    rejected: tuple[SidonExtensionCandidateResult, ...],
) -> None:
    """Validate cheap source binding, partition, and obstruction structure."""

    _validate_source_and_candidates(source_elements, candidate_elements)
    candidates = tuple(candidate_elements)
    rejected_candidates = tuple(item.candidate for item in rejected)
    partition = (*admissible, *rejected_candidates)
    if len(partition) != len(candidates) or len(set(partition)) != len(partition):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "result must partition every candidate exactly once",
        )
    if set(partition) != set(candidates):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "result candidates must match the request",
        )

    source_values = {int(value) for value in source_elements}
    for item in rejected:
        obstruction = item.obstruction
        if item.is_admissible or obstruction is None:
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "rejected candidates require an inadmissible obstruction",
            )
        if obstruction.candidate != item.candidate:
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "obstruction candidate must match its result row",
            )
        _validate_obstruction_witness(obstruction)
        extended = source_values | {int(item.candidate)}
        if not all(
            int(left) in extended and int(right) in extended
            for left, right in (obstruction.pair_a, obstruction.pair_b)
        ):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "obstruction pairs must belong to the source extension",
            )


def _candidate_obstruction(
    source_elements: tuple[AdditiveInteger, ...],
    source_pairs: Mapping[int, _DifferencePair],
    candidate: AdditiveInteger,
) -> _CandidateObstruction | None:
    """Find one repeated difference after adding a candidate."""

    candidate_value = int(candidate)
    candidate_pairs: dict[int, tuple[int, int]] = {}
    for source_element in source_elements:
        source_value = int(source_element)
        for pair in (
            (candidate_value, source_value),
            (source_value, candidate_value),
        ):
            difference = pair[0] - pair[1]
            previous = source_pairs.get(difference)
            if previous is None:
                previous = candidate_pairs.get(difference)
            if previous is not None:
                return difference, previous, pair
            candidate_pairs[difference] = pair
    return None


__all__ = [
    "MAX_EXTENSION_WORK",
    "SidonExtensionCandidateResult",
    "SidonExtensionObstruction",
    "SidonExtensionProfileRequest",
    "SidonExtensionProfileResult",
]
