"""Typed contracts and bounded replay for finite difference-set operations."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterator
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_SIDON_SET_SIZE = 32
MAX_CYCLIC_DIFFERENCE_SET_MODULUS = 4_096
MAX_DIFFERENCE_SET_EXTENSION_CANDIDATES = 50_000
MAX_DIFFERENCE_SET_ADDITIONAL_ELEMENTS = 3
# An ``AdditiveInteger`` canonical string is at most
# ``MAX_ADDITIVE_INTEGER_LENGTH`` characters: a positive value may use every
# character for digits, while a negative value spends one character on the
# leading ``-``. The widest ordered difference ``minuend - subtrahend`` pairs
# the largest accepted positive value with the most-negative accepted value
# (or vice versa), so its magnitude reaches ``(10**L - 1) + (10**(L - 1) - 1)``,
# which carries one extra digit; the negative sign then adds one more
# character. The result bound is therefore the input bound plus two.
MAX_ADDITIVE_INTEGER_LENGTH = 128
MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH = MAX_ADDITIVE_INTEGER_LENGTH + 2

AdditiveInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        max_length=MAX_ADDITIVE_INTEGER_LENGTH,
        strict=True,
    ),
]
AdditiveDifferenceInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        max_length=MAX_ADDITIVE_DIFFERENCE_INTEGER_LENGTH,
        strict=True,
    ),
]


def _difference_set_validation_error(code: str, message: str) -> PydanticCustomError:
    """Return one explicit stable error owned by this operation family."""

    return PydanticCustomError(code, message, {})


class IntegerSidonRequest(StrictModel):
    """One bounded finite integer set for ordered-difference replay."""

    elements: tuple[AdditiveInteger, ...] = Field(max_length=MAX_SIDON_SET_SIZE)

    @model_validator(mode="after")
    def require_unique_elements(self) -> Self:
        if len(set(self.elements)) != len(self.elements):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant", "Sidon input elements must be unique"
            )
        return self


class OrderedIntegerDifference(StrictModel):
    minuend: AdditiveInteger
    subtrahend: AdditiveInteger
    difference: AdditiveDifferenceInteger


class IntegerSidonResult(StrictModel):
    """Complete ordered-difference profile and exact Sidon decision."""

    normalized_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_SIDON_SET_SIZE
    )
    ordered_differences: tuple[OrderedIntegerDifference, ...] = Field(
        max_length=MAX_SIDON_SET_SIZE * (MAX_SIDON_SET_SIZE - 1)
    )
    is_sidon: StrictBool
    exactness: Literal["EXACT_INTEGER"] = "EXACT_INTEGER"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_complete_ordered_difference_profile(self) -> Self:
        values = tuple(int(value) for value in self.normalized_elements)
        if values != tuple(sorted(set(values))):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "normalized Sidon elements must be sorted and unique",
            )
        expected = tuple(
            (left, right, left - right)
            for left in values
            for right in values
            if left != right
        )
        actual = tuple(
            (
                int(record.minuend),
                int(record.subtrahend),
                int(record.difference),
            )
            for record in self.ordered_differences
        )
        if actual != expected:
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "ordered differences must cover every distinct ordered pair canonically",
            )
        if self.is_sidon != (len({item[2] for item in expected}) == len(expected)):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "Sidon decision must match the ordered differences",
            )
        return self


class CyclicPerfectDifferenceSetRequest(StrictModel):
    """One canonical residue set and modulus for exact PDS decision."""

    modulus: StrictInt = Field(ge=2, le=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    residues: tuple[StrictInt, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_canonical_residue_set(self) -> Self:
        if len(set(self.residues)) != len(self.residues):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant", "PDS residues must be unique"
            )
        if any(residue < 0 or residue >= self.modulus for residue in self.residues):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "PDS residues must be canonical modulo the modulus",
            )
        return self


def _cyclic_difference_multiplicities(
    residues: tuple[int, ...],
    modulus: int,
) -> dict[int, int]:
    """Recompute nonzero cyclic difference multiplicities from the residue set.

    The authoritative result-model validators use this clean-room recompute to
    reject forged but internally self-consistent profiles: a producer regression
    that materializes an incorrect ``COMPUTED`` result cannot pass the boundary
    even when the submitted multiplicity fields agree with the decision flag.
    """

    counts: Counter[int] = Counter(
        (left - right) % modulus
        for left in residues
        for right in residues
        if left != right
    )
    return {residue: counts.get(residue, 0) for residue in range(1, modulus)}


def _is_perfect_difference_set(
    residues: tuple[int, ...],
    modulus: int,
) -> bool:
    """Decide the perfect-difference-set property from the residue set."""

    if modulus != len(residues) * (len(residues) - 1) + 1:
        return False
    return all(
        multiplicity == 1
        for multiplicity in _cyclic_difference_multiplicities(
            residues, modulus
        ).values()
    )


class CyclicDifferenceMultiplicity(StrictModel):
    residue: StrictInt = Field(ge=1, lt=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    multiplicity: StrictInt = Field(ge=0, le=4_096)


class CyclicPerfectDifferenceSetResult(StrictModel):
    """Complete nonzero cyclic difference profile and exact PDS decision."""

    modulus: StrictInt = Field(ge=2, le=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    normalized_residues: tuple[StrictInt, ...] = Field(min_length=1, max_length=64)
    order: StrictInt = Field(ge=1, le=64)
    expected_modulus: StrictInt = Field(ge=1, le=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    difference_multiplicities: tuple[CyclicDifferenceMultiplicity, ...] = Field(
        min_length=1,
        max_length=MAX_CYCLIC_DIFFERENCE_SET_MODULUS - 1,
    )
    missing_residues: tuple[StrictInt, ...] = Field(
        max_length=MAX_CYCLIC_DIFFERENCE_SET_MODULUS - 1
    )
    repeated_residues: tuple[StrictInt, ...] = Field(
        max_length=MAX_CYCLIC_DIFFERENCE_SET_MODULUS - 1
    )
    is_perfect: StrictBool
    exactness: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_complete_cyclic_profile(self) -> Self:
        residues = self.normalized_residues
        if residues != tuple(sorted(set(residues))):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "normalized PDS residues must be sorted and unique",
            )
        if any(residue < 0 or residue >= self.modulus for residue in residues):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "normalized PDS residues must lie in the modulus",
            )
        if self.order != len(residues):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "PDS order must equal the residue-set cardinality",
            )
        if self.expected_modulus != self.order * (self.order - 1) + 1:
            raise _difference_set_validation_error(
                "combinatorics.invariant", "expected_modulus must equal k(k-1)+1"
            )
        profile = self.difference_multiplicities
        if tuple(item.residue for item in profile) != tuple(range(1, self.modulus)):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "cyclic difference profile must cover every nonzero residue",
            )
        recomputed = _cyclic_difference_multiplicities(residues, self.modulus)
        if any(item.multiplicity != recomputed[item.residue] for item in profile):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "cyclic difference multiplicities must be derived from the residues",
            )
        missing = tuple(item.residue for item in profile if item.multiplicity == 0)
        repeated = tuple(item.residue for item in profile if item.multiplicity > 1)
        if self.missing_residues != missing or self.repeated_residues != repeated:
            raise _difference_set_validation_error(
                "combinatorics.invariant",
                "missing and repeated residues must match the profile",
            )
        expected_perfect = (
            self.modulus == self.expected_modulus and not missing and not repeated
        )
        if self.is_perfect != expected_perfect:
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "PDS decision must match the complete residue profile",
            )
        return self


class CyclicDifferenceSetExtensionRequest(StrictModel):
    """A fixed-order direct-containment question in the derived cyclic group."""

    base_elements: tuple[AdditiveInteger, ...] = Field(min_length=1, max_length=64)
    target_order: StrictInt = Field(ge=2, le=64)

    @model_validator(mode="after")
    def require_bounded_complete_candidate_space(self) -> Self:
        if len(set(self.base_elements)) != len(self.base_elements):
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension base elements must be unique",
            )
        modulus = self.target_order * (self.target_order - 1) + 1
        if modulus > MAX_CYCLIC_DIFFERENCE_SET_MODULUS:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "derived extension modulus exceeds the supported bound",
            )
        base_residues = {int(value) % modulus for value in self.base_elements}
        additional = self.target_order - len(base_residues)
        if additional < 0:
            raise _difference_set_validation_error(
                "combinatorics.invariant",
                "target_order is smaller than the reduced base set",
            )
        if additional > MAX_DIFFERENCE_SET_ADDITIONAL_ELEMENTS:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension request requires too many added elements",
            )
        candidates = math.comb(modulus - len(base_residues), additional)
        if candidates > MAX_DIFFERENCE_SET_EXTENSION_CANDIDATES:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension candidate space exceeds the complete-search bound",
            )
        return self


def _extension_result_candidate_count(
    *,
    target_order: int,
    modulus: int,
    base_residues: tuple[int, ...],
) -> int:
    expected_modulus = target_order * (target_order - 1) + 1
    if modulus != expected_modulus:
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant", "extension modulus must equal k(k-1)+1"
        )
    if base_residues != tuple(sorted(set(base_residues))):
        raise _difference_set_validation_error(
            "combinatorics.invariant", "base residues must be sorted and unique"
        )
    if any(residue < 0 or residue >= modulus for residue in base_residues):
        raise _difference_set_validation_error(
            "combinatorics.invariant", "base residues must lie in the derived modulus"
        )
    additional = target_order - len(base_residues)
    if additional < 0 or additional > MAX_DIFFERENCE_SET_ADDITIONAL_ELEMENTS:
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant",
            "extension result lies outside the supported added-element bound",
        )
    return math.comb(modulus - len(base_residues), additional)


def _require_positive_extension_shape(
    *,
    target_order: int,
    modulus: int,
    base_residues: tuple[int, ...],
    extension: tuple[int, ...],
    coverage: str,
) -> None:
    if coverage != "WITNESS":
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant",
            "positive extension decisions require witness coverage",
        )
    if extension != tuple(sorted(set(extension))):
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant",
            "extension witness must be sorted and unique",
        )
    if len(extension) != target_order:
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant",
            "extension witness must have target_order residues",
        )
    if any(residue < 0 or residue >= modulus for residue in extension):
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant",
            "extension witness residues must lie in the derived modulus",
        )
    if not set(base_residues) <= set(extension):
        raise _difference_set_validation_error(
            "combinatorics.extension_invariant",
            "extension witness must contain the reduced base set",
        )
    if not _is_perfect_difference_set(extension, modulus):
        raise _difference_set_validation_error(
            "combinatorics.difference_set_invariant",
            "extension witness must be a perfect difference set of the derived modulus",
        )


def _enumerate_extension_candidates(
    base_residues: tuple[int, ...],
    target_order: int,
    modulus: int,
) -> Iterator[tuple[int, ...]]:
    """Yield every target_order residue superset of the reduced base set."""

    base_set = set(base_residues)
    available = tuple(residue for residue in range(modulus) if residue not in base_set)
    additional = target_order - len(base_residues)
    for combination in itertools.combinations(available, additional):
        yield tuple(sorted((*base_residues, *combination)))


def _find_extension_witness(
    base_residues: tuple[int, ...],
    target_order: int,
    modulus: int,
) -> tuple[int, ...] | None:
    """Return one perfect extension witness, or ``None`` if none exists."""

    for candidate in _enumerate_extension_candidates(
        base_residues, target_order, modulus
    ):
        if _is_perfect_difference_set(candidate, modulus):
            return candidate
    return None


class CyclicDifferenceSetExtensionResult(StrictModel):
    """A witness or complete negative decision for one fixed PDS order."""

    target_order: StrictInt = Field(ge=2, le=64)
    modulus: StrictInt = Field(ge=2, le=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    base_residues: tuple[StrictInt, ...] = Field(min_length=1, max_length=64)
    candidate_space_size: StrictInt = Field(
        ge=1, le=MAX_DIFFERENCE_SET_EXTENSION_CANDIDATES
    )
    decision: Literal["EXTENDS", "DOES_NOT_EXTEND"]
    extension: tuple[StrictInt, ...] = Field(max_length=64)
    coverage: Literal["WITNESS", "ALL_CANDIDATES"]
    exactness: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_fixed_order_scope_and_decision_shape(self) -> Self:
        expected_candidates = _extension_result_candidate_count(
            target_order=self.target_order,
            modulus=self.modulus,
            base_residues=self.base_residues,
        )
        if self.candidate_space_size != expected_candidates:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "candidate_space_size must cover the exact combination space",
            )
        if self.decision == "EXTENDS":
            _require_positive_extension_shape(
                target_order=self.target_order,
                modulus=self.modulus,
                base_residues=self.base_residues,
                extension=self.extension,
                coverage=self.coverage,
            )
        elif self.extension or self.coverage != "ALL_CANDIDATES":
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "negative extension decisions require empty witness and full coverage",
            )
        elif (
            _find_extension_witness(self.base_residues, self.target_order, self.modulus)
            is not None
        ):
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "negative extension decision must match the exhaustive search",
            )
        return self
