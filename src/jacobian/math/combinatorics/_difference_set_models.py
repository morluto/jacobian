"""Typed contracts for finite difference-set operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

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

# Empty arrays plus the longer ``false`` Sidon decision occupy 68 bytes
# before contents are inserted. Each ordered-difference object contributes
# 46 bytes of field spelling around the three integer wires.
_SIDON_RESULT_ENVELOPE_BYTES = 68
_SIDON_DIFFERENCE_ROW_OVERHEAD_BYTES = 46
MAX_SIDON_RESULT_BYTES = CanonicalLimits().max_output_bytes


@dataclass(frozen=True, slots=True)
class _IntegerSidonAdmissionPlan:
    """Request-scoped Sidon wires reused by trusted result construction."""

    normalized_wires: tuple[str, ...]
    difference_wires: tuple[tuple[str, str, str], ...]
    result_bytes: int
    is_sidon: bool


def _integer_sidon_profile(elements: tuple[int, ...]) -> _IntegerSidonAdmissionPlan:
    """Traverse one ordered-difference ledger and retain its canonical wires."""

    normalized_wires = tuple(str(value) for value in elements)
    normalized_bytes = sum(len(wire) + 2 for wire in normalized_wires) + max(
        len(normalized_wires) - 1, 0
    )
    difference_wires: list[tuple[str, str, str]] = []
    seen_differences: set[int] = set()
    difference_bytes = 0
    for left, left_wire in zip(elements, normalized_wires, strict=True):
        for right, right_wire in zip(elements, normalized_wires, strict=True):
            if left == right:
                continue
            difference = left - right
            difference_wire = str(difference)
            difference_wires.append((left_wire, right_wire, difference_wire))
            difference_bytes += (
                _SIDON_DIFFERENCE_ROW_OVERHEAD_BYTES
                + len(left_wire)
                + len(right_wire)
                + len(difference_wire)
            )
            seen_differences.add(difference)
    pair_count = len(difference_wires)
    difference_bytes += max(pair_count - 1, 0)
    return _IntegerSidonAdmissionPlan(
        normalized_wires=normalized_wires,
        difference_wires=tuple(difference_wires),
        result_bytes=_SIDON_RESULT_ENVELOPE_BYTES + normalized_bytes + difference_bytes,
        is_sidon=len(seen_differences) == pair_count,
    )


def _integer_sidon_canonical_result_bytes(elements: tuple[int, ...]) -> int:
    """Return the compact JSON size of one complete ordered-difference ledger.

    The estimate is exact for ``is_sidon=false`` and overcounts a true
    decision by one byte, so admission remains conservative.
    """

    return _integer_sidon_profile(elements).result_bytes


def _minimum_payload_sidon_start(cardinality: int) -> int:
    """Return the start of a consecutive n-set with minimum source-wire length.

    Consecutive integers minimize ordered-difference magnitudes. Shifting that
    interval toward the shortest AdditiveInteger spellings then minimizes the
    source wires that appear in every row; several starts can tie.
    """

    if cardinality <= 0:
        return 0
    current = sum(len(str(value)) for value in range(cardinality))
    best = current
    best_start = 0
    start = 0
    while start > -cardinality:
        start -= 1
        current += len(str(start)) - len(str(start + cardinality))
        if current < best:
            best = current
            best_start = start
        elif current > best:
            break
    return best_start


def _minimum_payload_sidon_elements(cardinality: int) -> tuple[int, ...]:
    """Return one unique AdditiveInteger n-set with the lightest ledger."""

    if cardinality <= 0:
        return ()
    start = _minimum_payload_sidon_start(cardinality)
    return tuple(range(start, start + cardinality))


def _minimum_integer_sidon_result_bytes(cardinality: int) -> int:
    """Compact JSON size of the lightest unique AdditiveInteger n-set ledger."""

    if cardinality <= 0:
        return _SIDON_RESULT_ENVELOPE_BYTES
    start = _minimum_payload_sidon_start(cardinality)
    length_sum = sum(len(str(value)) for value in range(start, start + cardinality))
    pair_count = cardinality * (cardinality - 1)
    normalized_bytes = length_sum + 2 * cardinality + (cardinality - 1)
    difference_value_bytes = 0
    for gap in range(1, cardinality):
        multiplicity = cardinality - gap
        positive_length = len(str(gap))
        difference_value_bytes += multiplicity * (2 * positive_length + 1)
    difference_bytes = (
        pair_count * _SIDON_DIFFERENCE_ROW_OVERHEAD_BYTES
        + 2 * (cardinality - 1) * length_sum
        + difference_value_bytes
        + max(pair_count - 1, 0)
    )
    return _SIDON_RESULT_ENVELOPE_BYTES + normalized_bytes + difference_bytes


def _max_sidon_set_size_for_output_budget(budget: int) -> int:
    """Largest ``n`` whose lightest unique AdditiveInteger n-set fits ``budget``.

    Pair work is ``n(n-1)`` ordered rows. The parser ceiling is the largest
    cardinality whose minimum payload can still fit; actual source and
    difference widths are reserved later by result-sensitive admission.
    """

    low = 0
    high = math.isqrt(max(budget, 0) // _SIDON_DIFFERENCE_ROW_OVERHEAD_BYTES) + 2
    while low < high:
        mid = (low + high + 1) // 2
        if _minimum_integer_sidon_result_bytes(mid) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


# Cheap parser bound derived from the canonical output budget. Result-sensitive
# admission still reserves the actual payload from source and difference widths.
MAX_SIDON_SET_SIZE = _max_sidon_set_size_for_output_budget(MAX_SIDON_RESULT_BYTES)
MAX_SIDON_ORDERED_DIFFERENCES = MAX_SIDON_SET_SIZE * max(MAX_SIDON_SET_SIZE - 1, 0)


def _difference_set_validation_error(code: str, message: str) -> PydanticCustomError:
    """Return one explicit stable error owned by this operation family."""

    return PydanticCustomError(code, message, {})


class IntegerSidonRequest(StrictModel):
    """One bounded finite integer set for an ordered-difference profile."""

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

    @classmethod
    def _from_kernel(cls, minuend: str, subtrahend: str, difference: str) -> Self:
        return cls.model_construct(
            minuend=minuend,
            subtrahend=subtrahend,
            difference=difference,
        )


class IntegerSidonResult(StrictModel):
    """Complete ordered-difference profile and exact Sidon decision."""

    normalized_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_SIDON_SET_SIZE
    )
    ordered_differences: tuple[OrderedIntegerDifference, ...] = Field(
        max_length=MAX_SIDON_ORDERED_DIFFERENCES
    )
    is_sidon: StrictBool

    @model_validator(mode="after")
    def require_canonical_profile_shape(self) -> Self:
        values = tuple(int(value) for value in self.normalized_elements)
        if values != tuple(sorted(set(values))):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "normalized Sidon elements must be sorted and unique",
            )
        if len(self.ordered_differences) != len(values) * (len(values) - 1):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "ordered-difference profile has the wrong cardinality",
            )
        seen_differences: set[int] = set()
        for record, (left, right) in zip(
            self.ordered_differences,
            ((left, right) for left in values for right in values if left != right),
            strict=True,
        ):
            difference = left - right
            if (
                int(record.minuend) != left
                or int(record.subtrahend) != right
                or int(record.difference) != difference
            ):
                raise _difference_set_validation_error(
                    "combinatorics.sidon_invariant",
                    "ordered-difference rows must be the canonical source pairs",
                )
            seen_differences.add(difference)
        if self.is_sidon != (len(seen_differences) == len(self.ordered_differences)):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "is_sidon must match the ordered-difference multiplicities",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        normalized_elements: tuple[AdditiveInteger, ...],
        ordered_differences: tuple[OrderedIntegerDifference, ...],
        is_sidon: bool,
    ) -> Self:
        return cls.model_construct(
            normalized_elements=normalized_elements,
            ordered_differences=ordered_differences,
            is_sidon=is_sidon,
        )


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


class CyclicDifferenceMultiplicity(StrictModel):
    residue: StrictInt = Field(ge=1, lt=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    multiplicity: StrictInt = Field(ge=0, le=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)


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

    @model_validator(mode="after")
    def require_canonical_profile_shape(self) -> Self:
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
        expected_missing = tuple(
            item.residue for item in profile if item.multiplicity == 0
        )
        expected_repeated = tuple(
            item.residue for item in profile if item.multiplicity > 1
        )
        if (
            self.missing_residues != expected_missing
            or self.repeated_residues != expected_repeated
        ):
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "cyclic difference exceptions must match the complete profile",
            )
        expected_perfect = (
            self.modulus == self.expected_modulus
            and not expected_missing
            and not expected_repeated
        )
        if self.is_perfect != expected_perfect:
            raise _difference_set_validation_error(
                "combinatorics.difference_set_invariant",
                "perfect-difference-set decision must match the complete profile",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        normalized_residues: tuple[int, ...],
        order: int,
        expected_modulus: int,
        difference_multiplicities: tuple[CyclicDifferenceMultiplicity, ...],
        missing_residues: tuple[int, ...],
        repeated_residues: tuple[int, ...],
        is_perfect: bool,
    ) -> Self:
        return cls.model_construct(
            modulus=modulus,
            normalized_residues=normalized_residues,
            order=order,
            expected_modulus=expected_modulus,
            difference_multiplicities=difference_multiplicities,
            missing_residues=missing_residues,
            repeated_residues=repeated_residues,
            is_perfect=is_perfect,
        )


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
        return self


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

    @model_validator(mode="after")
    def bind_extension_claim(self) -> Self:
        expected_modulus = self.target_order * (self.target_order - 1) + 1
        if self.modulus != expected_modulus:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension modulus must equal k(k-1)+1",
            )
        if self.base_residues != tuple(sorted(set(self.base_residues))) or any(
            residue < 0 or residue >= self.modulus for residue in self.base_residues
        ):
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension base residues must be canonical",
            )
        additional = self.target_order - len(self.base_residues)
        if additional < 0:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension target order must contain the retained base",
            )
        expected_candidates = math.comb(
            self.modulus - len(self.base_residues), additional
        )
        if self.candidate_space_size != expected_candidates:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "extension candidate-space size must match the retained source",
            )
        if self.decision == "EXTENDS":
            if (
                self.coverage != "WITNESS"
                or len(self.extension) != self.target_order
                or self.extension != tuple(sorted(set(self.extension)))
                or not set(self.base_residues) <= set(self.extension)
                or any(
                    residue < 0 or residue >= self.modulus for residue in self.extension
                )
            ):
                raise _difference_set_validation_error(
                    "combinatorics.extension_invariant",
                    "extension witness must be canonical and retain the base residues",
                )
        elif self.coverage != "ALL_CANDIDATES" or self.extension:
            raise _difference_set_validation_error(
                "combinatorics.extension_invariant",
                "negative extension decisions cannot carry a witness",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        target_order: int,
        modulus: int,
        base_residues: tuple[int, ...],
        candidate_space_size: int,
        extension: tuple[int, ...] | None,
    ) -> Self:
        return cls.model_construct(
            target_order=target_order,
            modulus=modulus,
            base_residues=base_residues,
            candidate_space_size=candidate_space_size,
            decision="EXTENDS" if extension is not None else "DOES_NOT_EXTEND",
            extension=extension or (),
            coverage="WITNESS" if extension is not None else "ALL_CANDIDATES",
        )
