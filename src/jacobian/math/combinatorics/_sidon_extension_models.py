"""Typed contracts for the Sidon extension-profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictBool, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.combinatorics._difference_set_models import (
    MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH,
    AdditiveDifferenceInteger,
    AdditiveInteger,
    _difference_set_validation_error,
)

MAX_EXTENSION_SOURCE_SIZE = 32
MAX_EXTENSION_WORK = 4_000_000
MAX_EXTENSION_RESULT_BYTES = 8 * 1024 * 1024
_EXTENSION_RESULT_ENVELOPE_BYTES = 4_096


def _ordered_difference_pairs(
    elements: tuple[AdditiveInteger, ...],
) -> dict[int, tuple[int, int]]:
    return {
        int(left) - int(right): (int(left), int(right))
        for left in elements
        for right in elements
        if left != right
    }


def _extension_work_units(source_count: int, candidate_count: int) -> int:
    """Price source profiling, candidate checks, and result replay."""

    source_pairs = source_count * (source_count - 1)
    per_candidate = 4 * source_count + 16
    return source_pairs + candidate_count * per_candidate


def _maximum_result_bytes(
    source_elements: tuple[AdditiveInteger, ...],
    candidate_elements: tuple[AdditiveInteger, ...],
) -> int:
    """Bound the result by materializing its largest certificate-shaped rows."""

    widest_element = max((*source_elements, *candidate_elements), key=len, default="0")
    widest_difference = "-" + "9" * (MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH - 1)
    rejected_entry = {
        "candidate": widest_element,
        "is_admissible": False,
        "obstruction": {
            "candidate": widest_element,
            "repeated_difference": widest_difference,
            "pair_a": [widest_element, widest_element],
            "pair_b": [widest_element, widest_element],
        },
    }
    result = {
        "source_elements": list(source_elements),
        "candidate_elements": list(candidate_elements),
        "admissible": [],
        "rejected": [rejected_entry] * len(candidate_elements),
    }
    return len(encode_strict_json(result)) + _EXTENSION_RESULT_ENVELOPE_BYTES


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
    source_pairs = _ordered_difference_pairs(source_elements)
    if len(source_pairs) != len(source_elements) * (len(source_elements) - 1):
        raise _difference_set_validation_error(
            "combinatorics.sidon_invariant",
            "source elements must form a Sidon set",
        )


class SidonExtensionProfileRequest(StrictModel):
    """Source Sidon set A and candidate set C disjoint from A."""

    source_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_EXTENSION_SOURCE_SIZE
    )
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
        work_units = _extension_work_units(
            len(self.source_elements),
            len(self.candidate_elements),
        )
        if work_units > MAX_EXTENSION_WORK:
            raise _difference_set_validation_error(
                "combinatorics.sidon_extension_work_budget",
                "Sidon extension search exceeds the bounded work budget",
            )
        result_bytes = _maximum_result_bytes(
            self.source_elements,
            self.candidate_elements,
        )
        if result_bytes > MAX_EXTENSION_RESULT_BYTES:
            raise _difference_set_validation_error(
                "combinatorics.sidon_extension_result_budget",
                "Sidon extension result exceeds the canonical output budget",
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

    source_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_EXTENSION_SOURCE_SIZE
    )
    candidate_elements: tuple[AdditiveInteger, ...]
    admissible: tuple[AdditiveInteger, ...]
    rejected: tuple[SidonExtensionCandidateResult, ...]

    @model_validator(mode="after")
    def require_complete_replayable_partition(self) -> Self:
        _validate_source_and_candidates(
            self.source_elements,
            self.candidate_elements,
        )
        result_bytes = _maximum_result_bytes(
            self.source_elements,
            self.candidate_elements,
        )
        if result_bytes > MAX_EXTENSION_RESULT_BYTES:
            raise _difference_set_validation_error(
                "combinatorics.sidon_extension_result_budget",
                "Sidon extension result exceeds the canonical output budget",
            )

        candidates = tuple(self.candidate_elements)
        admissible = tuple(self.admissible)
        rejected = tuple(item.candidate for item in self.rejected)
        partition = (*admissible, *rejected)
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
        source_values = {int(value) for value in self.source_elements}
        source_pairs = _ordered_difference_pairs(self.source_elements)
        for candidate in admissible:
            if (
                _candidate_obstruction(
                    self.source_elements,
                    source_pairs,
                    candidate,
                )
                is not None
            ):
                raise _difference_set_validation_error(
                    "combinatorics.sidon_invariant",
                    "admissible candidates must preserve the Sidon property",
                )
        for item in self.rejected:
            obstruction = item.obstruction
            if obstruction is None:
                raise _difference_set_validation_error(
                    "combinatorics.sidon_invariant",
                    "rejected candidates require an obstruction",
                )
            extended = source_values | {int(item.candidate)}
            if not all(
                int(left) in extended and int(right) in extended
                for left, right in (obstruction.pair_a, obstruction.pair_b)
            ):
                raise _difference_set_validation_error(
                    "combinatorics.sidon_invariant",
                    "obstruction pairs must belong to the source extension",
                )
        return self


def _candidate_obstruction(
    source_elements: tuple[AdditiveInteger, ...],
    source_pairs: dict[int, tuple[int, int]],
    candidate: AdditiveInteger,
) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    """Find one repeated difference after adding a candidate."""

    candidate_value = int(candidate)
    seen = dict(source_pairs)
    for source_element in source_elements:
        source_value = int(source_element)
        for pair in (
            (candidate_value, source_value),
            (source_value, candidate_value),
        ):
            difference = pair[0] - pair[1]
            previous = seen.get(difference)
            if previous is not None:
                return difference, previous, pair
            seen[difference] = pair
    return None


__all__ = [
    "MAX_EXTENSION_RESULT_BYTES",
    "MAX_EXTENSION_SOURCE_SIZE",
    "MAX_EXTENSION_WORK",
    "SidonExtensionCandidateResult",
    "SidonExtensionObstruction",
    "SidonExtensionProfileRequest",
    "SidonExtensionProfileResult",
]
