"""Typed contracts for the Sidon extension-profile operation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from pydantic import Field, PrivateAttr, StrictBool, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics._difference_set_models import (
    MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH,
    AdditiveDifferenceInteger,
    AdditiveInteger,
    _difference_set_validation_error,
)

MAX_EXTENSION_WORK = 4_000_000
MAX_EXTENSION_RESULT_BYTES = 8 * 1024 * 1024
MAX_EXTENSION_INTERMEDIATE_BYTES = 256 * 1024 * 1024
_EXTENSION_RESULT_ENVELOPE_BYTES = 4_096
_PYTHON_DICT_SLOT_BYTES = 64
_PYTHON_TUPLE_BYTES = 56

_DifferencePair = tuple[int, int]
_CandidateObstruction = tuple[int, _DifferencePair, _DifferencePair]


@dataclass(frozen=True)
class _SidonExtensionAdmissionPlan:
    """Request-scoped admission data reused by the owner-local kernel."""

    source_differences: Mapping[int, _DifferencePair]
    candidate_obstructions: tuple[_CandidateObstruction | None, ...] | None = None


def _python_int_storage_bytes(decimal_digits: int) -> int:
    """Conservatively bound one CPython integer's storage."""

    # A Python integer uses 30-bit limbs and has a 28-byte object header. Four
    # bits per decimal digit overstates the limb count for every accepted
    # integer, including the 130-character ordered differences.
    limbs = (decimal_digits * 4 + 29) // 30
    return 28 + 4 * limbs


def _source_profile_storage_bytes(
    source_elements: tuple[AdditiveInteger, ...],
) -> int:
    """Bound one materialized source-difference profile's peak storage."""

    pair_count = len(source_elements) * (len(source_elements) - 1)
    if pair_count == 0:
        return 0
    widest_element_length = max(len(value) for value in source_elements)
    element_bytes = _python_int_storage_bytes(widest_element_length)
    difference_bytes = _python_int_storage_bytes(widest_element_length + 2)
    entry_bytes = (
        _PYTHON_DICT_SLOT_BYTES
        + _PYTHON_TUPLE_BYTES
        + difference_bytes
        + 2 * element_bytes
    )
    return pair_count * entry_bytes


def _require_source_profile_memory_budget(
    source_elements: tuple[AdditiveInteger, ...],
) -> None:
    estimated_bytes = _source_profile_storage_bytes(source_elements)
    if estimated_bytes > MAX_EXTENSION_INTERMEDIATE_BYTES:
        raise _difference_set_validation_error(
            "combinatorics.sidon_extension_intermediate_budget",
            "Sidon source-difference profiling exceeds the bounded intermediate-storage budget",
        )


def _ordered_difference_pairs(
    elements: tuple[AdditiveInteger, ...],
) -> dict[int, _DifferencePair]:
    return {
        int(left) - int(right): (int(left), int(right))
        for left in elements
        for right in elements
        if left != right
    }


def _extension_work_units(source_count: int, candidate_count: int) -> int:
    """Price source profiling and candidate-local difference checks."""

    source_pairs = source_count * (source_count - 1)
    per_candidate = 4 * source_count + 16
    return source_pairs + candidate_count * per_candidate


def _maximum_result_bytes(
    source_elements: tuple[AdditiveInteger, ...],
    candidate_elements: tuple[AdditiveInteger, ...],
    candidate_obstructions: tuple[_CandidateObstruction | None, ...] | None = None,
) -> int:
    """Bound canonical result bytes without constructing a hypothetical profile.

    All accepted integer strings contain only ASCII digits and an optional
    minus sign, so their canonical JSON string size is their character length
    plus two quote bytes. Without candidate outcomes, every candidate is
    charged for the larger of its scalar admissible row and its
    certificate-shaped rejected row. An admitted request-scoped profile may
    provide the actual candidate outcomes to price the attainable partition.
    """

    def string_bytes(character_count: int) -> int:
        return character_count + 2

    def array_bytes(values: tuple[AdditiveInteger, ...]) -> int:
        if not values:
            return 2
        return 1 + sum(len(value) + 3 for value in values)

    def pair_array_bytes(character_count: int) -> int:
        return 1 + 2 * (character_count + 3)

    def object_bytes(fields: tuple[tuple[str, int], ...]) -> int:
        return 2 + sum(len(key) + 3 + value for key, value in fields) + len(fields) - 1

    source_bytes = array_bytes(source_elements)
    candidate_bytes = array_bytes(candidate_elements)
    widest_element_length = max(
        max((len(value) for value in source_elements), default=0),
        max((len(value) for value in candidate_elements), default=0),
    )

    def obstruction_bytes(
        obstruction: _CandidateObstruction | None,
        candidate_length: int,
    ) -> int:
        if obstruction is None:
            difference_length = MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH
            pair_length = widest_element_length
        else:
            difference, pair_a, pair_b = obstruction
            pair_length = max(
                len(str(pair_a[0])),
                len(str(pair_a[1])),
                len(str(pair_b[0])),
                len(str(pair_b[1])),
            )
            difference_length = len(str(difference))
        return object_bytes(
            (
                ("candidate", string_bytes(candidate_length)),
                ("repeated_difference", string_bytes(difference_length)),
                ("pair_a", pair_array_bytes(pair_length)),
                ("pair_b", pair_array_bytes(pair_length)),
            )
        )

    partition_content_bytes = 0
    all_candidates_admissible = len(source_elements) <= 1
    for index, candidate in enumerate(candidate_elements):
        admissible_bytes = string_bytes(len(candidate))
        if candidate_obstructions is not None:
            outcome = candidate_obstructions[index]
            row_bytes = (
                admissible_bytes
                if outcome is None
                else object_bytes(
                    (
                        ("candidate", admissible_bytes),
                        ("is_admissible", len("false")),
                        (
                            "obstruction",
                            obstruction_bytes(outcome, len(candidate)),
                        ),
                    )
                )
            )
            partition_content_bytes += row_bytes
        elif all_candidates_admissible:
            partition_content_bytes += admissible_bytes
        else:
            rejected_bytes = object_bytes(
                (
                    ("candidate", admissible_bytes),
                    ("is_admissible", len("false")),
                    ("obstruction", obstruction_bytes(None, len(candidate))),
                )
            )
            partition_content_bytes += max(admissible_bytes, rejected_bytes)
        # The two partition arrays share exactly the candidate count, but this
        # check must stop before a large rejected-profile estimate accumulates.
        if partition_content_bytes > MAX_EXTENSION_RESULT_BYTES:
            return MAX_EXTENSION_RESULT_BYTES + 1

    # Two arrays are always present.  Their combined separators and brackets
    # are at most four bytes plus one separator per partitioned candidate.
    partition_bytes = 4 + len(candidate_elements) + partition_content_bytes
    result_bytes = (
        object_bytes(
            (
                ("source_elements", source_bytes),
                ("candidate_elements", candidate_bytes),
                ("admissible", 0),
                ("rejected", 0),
            )
        )
        + partition_bytes
        + _EXTENSION_RESULT_ENVELOPE_BYTES
    )
    return result_bytes


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
    _admission_plan: _SidonExtensionAdmissionPlan | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def require_unique_and_disjoint(self) -> Self:
        _validate_source_and_candidates(
            self.source_elements,
            self.candidate_elements,
        )
        _require_extension_work_budget(
            len(self.source_elements),
            len(self.candidate_elements),
        )
        _require_source_profile_memory_budget(self.source_elements)
        source_differences = _validate_source_is_sidon(self.source_elements)
        result_bytes = _maximum_result_bytes(
            self.source_elements,
            self.candidate_elements,
        )
        candidate_obstructions: tuple[_CandidateObstruction | None, ...] | None = None
        if result_bytes > MAX_EXTENSION_RESULT_BYTES:
            candidate_obstructions = tuple(
                _candidate_obstruction(
                    self.source_elements,
                    source_differences,
                    candidate,
                )
                for candidate in self.candidate_elements
            )
            result_bytes = _maximum_result_bytes(
                self.source_elements,
                self.candidate_elements,
                candidate_obstructions,
            )
        if result_bytes > MAX_EXTENSION_RESULT_BYTES:
            raise _difference_set_validation_error(
                "combinatorics.sidon_extension_result_budget",
                "Sidon extension result exceeds the canonical output budget",
            )
        self._admission_plan = _SidonExtensionAdmissionPlan(
            source_differences=MappingProxyType(source_differences),
            candidate_obstructions=candidate_obstructions,
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
    def require_complete_replayable_partition(self) -> Self:
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
    _require_extension_work_budget(len(source_elements), len(candidate_elements))
    _require_source_profile_memory_budget(source_elements)
    _validate_source_is_sidon(source_elements)
    result_bytes = _maximum_result_bytes(source_elements, candidate_elements)
    if result_bytes > MAX_EXTENSION_RESULT_BYTES:
        raise _difference_set_validation_error(
            "combinatorics.sidon_extension_result_budget",
            "Sidon extension result exceeds the canonical output budget",
        )

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


def verify_sidon_extension_profile_result(
    result: SidonExtensionProfileResult,
) -> bool:
    """Replay an independently supplied profile claim within its bound."""

    try:
        _validate_profile_structure(
            result.source_elements,
            result.candidate_elements,
            result.admissible,
            result.rejected,
        )
    except (TypeError, ValueError):
        return False

    source_pairs = _ordered_difference_pairs(result.source_elements)
    for candidate in result.admissible:
        if (
            _candidate_obstruction(result.source_elements, source_pairs, candidate)
            is not None
        ):
            return False
    for item in result.rejected:
        if (
            _candidate_obstruction(
                result.source_elements,
                source_pairs,
                item.candidate,
            )
            is None
        ):
            return False
    return True


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
    "MAX_EXTENSION_INTERMEDIATE_BYTES",
    "MAX_EXTENSION_RESULT_BYTES",
    "MAX_EXTENSION_WORK",
    "SidonExtensionCandidateResult",
    "SidonExtensionObstruction",
    "SidonExtensionProfileRequest",
    "SidonExtensionProfileResult",
    "verify_sidon_extension_profile_result",
]
