"""Named Pydantic wire contracts for exact combinatorics capabilities."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.contracts.exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.contracts.results import ContractModel

_MAX_N = 1_000
MAX_BINOMIAL_N = 10_000
_MAX_PARTS = 256
_MAX_PARTITION_N = 30
_MAX_ENUMERATED_PARTITIONS = 10_000
MAX_LINEAR_RECURRENCE_ORDER = 16
MAX_LINEAR_RECURRENCE_INDEX = 512
MAX_LINEAR_RECURRENCE_REQUESTED_INDICES = 256
MAX_P_RECURSIVE_POLYNOMIAL_DEGREE = 16
MAX_RATIONAL_GENERATING_FUNCTION_DEGREE = 32
MAX_RATIONAL_SERIES_TRUNCATION_ORDER = 512
MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS = 64
MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS = 32_768
MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES = 10 * 1024 * 1024
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
_LOG10_2 = math.log10(2)

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


def _fraction_wire(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _require_bounded_fraction(
    value: Fraction,
    *,
    max_digits: int,
    label: str,
) -> None:
    if (
        len(format_canonical_integer(abs(value.numerator))) > max_digits
        or len(format_canonical_integer(value.denominator)) > max_digits
    ):
        raise ValueError(f"{label} exceeds the {max_digits}-digit bound")


def _lower_decimal_digits(value: int) -> int:
    if value == 0:
        return 1
    return math.floor((abs(value).bit_length() - 1) * _LOG10_2) + 1


def _minimum_fraction_wire_bytes(value: Fraction) -> int:
    return (
        _lower_decimal_digits(value.numerator)
        + _lower_decimal_digits(value.denominator)
        + 20
    )


def _validate_result_artifact_size(payload: dict[str, object]) -> None:
    try:
        canonicalize_json(
            payload,
            limits=CanonicalLimits(
                max_output_bytes=MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES,
                max_integer_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            ),
        )
    except ValueError as exc:
        raise ValueError(
            "the exact combinatorics result exceeds the durable artifact limit"
        ) from exc


def _recurrence_replay(
    coefficients: tuple[Fraction, ...],
    initial_values: tuple[Fraction, ...],
    end: int,
) -> list[Fraction]:
    replay = list(initial_values[: end + 1])
    while len(replay) <= end:
        replay.append(
            sum(
                (
                    coefficient * replay[len(replay) - offset]
                    for offset, coefficient in enumerate(coefficients, start=1)
                ),
                start=Fraction(),
            )
        )
    return replay


def _validate_recurrence_result_budget(
    *,
    coefficients: tuple[CanonicalRational, ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    scope: str,
    requested_indices: tuple[int, ...],
) -> None:
    replay = _recurrence_replay(
        tuple(value.as_fraction() for value in coefficients),
        tuple(value.as_fraction() for value in initial_values),
        requested_indices[-1],
    )
    minimum_size = sum(_minimum_fraction_wire_bytes(value) for value in replay)
    minimum_size += sum(
        _minimum_fraction_wire_bytes(replay[index]) for index in requested_indices
    )
    if minimum_size + 1_024 > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
        raise ValueError(
            "the exact combinatorics result exceeds the durable artifact limit"
        )
    for value in replay:
        if any(
            _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
            for component in (value.numerator, value.denominator)
        ):
            raise ValueError(
                "recurrence result exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            )
        _require_bounded_fraction(
            value,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="recurrence result",
        )
    _validate_result_artifact_size(
        {
            "backend": "sympy",
            "backend_version": "1.14.0",
            "coefficient_convention": coefficient_convention,
            "determinism": "DETERMINISTIC",
            "exactness": "EXACT_RATIONAL",
            "replay_prefix": [_fraction_wire(value) for value in replay],
            "replay_scope_end": requested_indices[-1],
            "scope": scope,
            "values": [
                {"index": index, "value": _fraction_wire(replay[index])}
                for index in requested_indices
            ],
            "verification": "UNVERIFIED",
        }
    )


def _validate_p_recursive_result_budget(
    *,
    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    polynomial_convention: str,
    scope: str,
    requested_indices: tuple[int, ...],
) -> None:
    polynomials = tuple(
        tuple(value.as_fraction() for value in polynomial)
        for polynomial in coefficient_polynomials
    )
    order = len(polynomials) - 1

    def polynomial_value(polynomial: tuple[Fraction, ...], index: int) -> Fraction:
        return sum(
            (
                coefficient * index**power
                for power, coefficient in enumerate(polynomial)
            ),
            start=Fraction(),
        )

    end = requested_indices[-1]
    replay = [value.as_fraction() for value in initial_values[: end + 1]]
    requested_index_set = set(requested_indices)
    minimum_size = 1_024 + sum(
        _minimum_fraction_wire_bytes(value) * (1 + (index in requested_index_set))
        for index, value in enumerate(replay)
    )
    residuals: list[tuple[int, Fraction]] = []
    while len(replay) <= end:
        index = len(replay)
        coefficients = tuple(
            polynomial_value(polynomial, index) for polynomial in polynomials
        )
        if coefficients[0] == 0:
            raise ValueError(
                f"leading coefficient polynomial vanishes at index {index}"
            )
        next_value = (
            -sum(
                (
                    coefficients[offset] * replay[index - offset]
                    for offset in range(1, order + 1)
                ),
                start=Fraction(),
            )
            / coefficients[0]
        )
        if any(
            _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
            for component in (next_value.numerator, next_value.denominator)
        ):
            raise ValueError(
                "polynomial-coefficient recurrence result exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            )
        _require_bounded_fraction(
            next_value,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="polynomial-coefficient recurrence result",
        )
        minimum_size += _minimum_fraction_wire_bytes(next_value) * (
            1 + (index in requested_index_set)
        )
        minimum_size += 32
        if minimum_size > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
            raise ValueError(
                "the exact combinatorics result exceeds the durable artifact limit"
            )
        replay.append(next_value)
        residuals.append(
            (
                index,
                sum(
                    (
                        coefficients[offset] * replay[index - offset]
                        for offset in range(order + 1)
                    ),
                    start=Fraction(),
                ),
            )
        )
    _validate_result_artifact_size(
        {
            "backend": "sympy",
            "backend_version": "1.14.0",
            "coefficient_convention": coefficient_convention,
            "polynomial_convention": polynomial_convention,
            "determinism": "DETERMINISTIC",
            "exactness": "EXACT_RATIONAL",
            "recurrence_order": order,
            "replay_prefix": [_fraction_wire(value) for value in replay],
            "residuals": [
                {"index": index, "value": _fraction_wire(value)}
                for index, value in residuals
            ],
            "replay_scope_end": end,
            "scope": scope,
            "values": [
                {"index": index, "value": _fraction_wire(replay[index])}
                for index in requested_indices
            ],
            "verification": "UNVERIFIED",
        }
    )


def _validate_series_result_budget(
    *,
    numerator: tuple[CanonicalRational, ...],
    denominator: tuple[CanonicalRational, ...],
    coefficient_convention: str,
    expansion_point: str,
    truncation_order: int,
) -> None:
    numerator_values = tuple(value.as_fraction() for value in numerator)
    denominator_values = tuple(value.as_fraction() for value in denominator)
    coefficients: list[Fraction] = []
    for degree in range(truncation_order):
        numerator_coefficient = (
            numerator_values[degree] if degree < len(numerator_values) else Fraction()
        )
        known = sum(
            (
                denominator_values[offset] * coefficients[degree - offset]
                for offset in range(
                    1,
                    min(degree, len(denominator_values) - 1) + 1,
                )
            ),
            start=Fraction(),
        )
        coefficient = (numerator_coefficient - known) / denominator_values[0]
        if any(
            _lower_decimal_digits(component) > MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS
            for component in (coefficient.numerator, coefficient.denominator)
        ):
            raise ValueError(
                "series coefficient exceeds the "
                f"{MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS}-digit bound"
            )
        _require_bounded_fraction(
            coefficient,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="series coefficient",
        )
        coefficients.append(coefficient)
    minimum_size = sum(_minimum_fraction_wire_bytes(value) for value in coefficients)
    minimum_size += truncation_order * _minimum_fraction_wire_bytes(Fraction())
    if minimum_size + 1_024 > MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES:
        raise ValueError(
            "the exact combinatorics result exceeds the durable artifact limit"
        )
    _validate_result_artifact_size(
        {
            "backend": "sympy",
            "backend_version": "1.14.0",
            "coefficient_convention": coefficient_convention,
            "coefficients": [_fraction_wire(value) for value in coefficients],
            "determinism": "DETERMINISTIC",
            "exactness": "EXACT_RATIONAL",
            "expansion_point": expansion_point,
            "residual_coefficients": [_fraction_wire(Fraction())] * truncation_order,
            "residual_congruence": (
                "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
            ),
            "truncation_order": truncation_order,
            "verification": "UNVERIFIED",
        }
    )


def _require_canonical_polynomial(
    coefficients: tuple[CanonicalRational, ...],
    *,
    label: str,
) -> None:
    for coefficient in coefficients:
        require_bounded_rational(
            coefficient,
            max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
            label=label,
        )
    if len(coefficients) > 1 and coefficients[-1].as_fraction() == 0:
        raise ValueError(f"{label} must omit trailing zero coefficients")


class NonnegativeIntegerRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=_MAX_N)


class NonnegativePairRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=_MAX_N)
    k: StrictInt = Field(ge=0, le=_MAX_N)


class BinomialRequest(ContractModel):
    """A wider safe bound for Python's efficient exact ``math.comb`` path."""

    n: StrictInt = Field(ge=0, le=MAX_BINOMIAL_N)
    k: StrictInt = Field(ge=0, le=MAX_BINOMIAL_N)


class IntegerSidonRequest(ContractModel):
    """One bounded finite integer set for ordered-difference replay."""

    elements: tuple[AdditiveInteger, ...] = Field(max_length=MAX_SIDON_SET_SIZE)

    @model_validator(mode="after")
    def require_unique_elements(self) -> Self:
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("Sidon input elements must be unique")
        return self


class OrderedIntegerDifference(ContractModel):
    minuend: AdditiveInteger
    subtrahend: AdditiveInteger
    difference: AdditiveDifferenceInteger


class IntegerSidonResult(ContractModel):
    """Complete ordered-difference profile and exact Sidon decision."""

    semantics_version: Literal["integer-sidon.ordered-differences.v1"]
    normalized_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_SIDON_SET_SIZE
    )
    ordered_differences: tuple[OrderedIntegerDifference, ...] = Field(
        max_length=MAX_SIDON_SET_SIZE * (MAX_SIDON_SET_SIZE - 1)
    )
    is_sidon: StrictBool
    exactness: Literal["EXACT_INTEGER"] = "EXACT_INTEGER"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["python-stdlib"] = "python-stdlib"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_complete_ordered_difference_profile(self) -> Self:
        values = tuple(int(value) for value in self.normalized_elements)
        if values != tuple(sorted(set(values))):
            raise ValueError("normalized Sidon elements must be sorted and unique")
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
            raise ValueError(
                "ordered differences must cover every distinct ordered pair canonically"
            )
        if self.is_sidon != (len({item[2] for item in expected}) == len(expected)):
            raise ValueError("Sidon decision must match the ordered differences")
        return self


class CyclicPerfectDifferenceSetRequest(ContractModel):
    """One canonical residue set and modulus for exact PDS decision."""

    modulus: StrictInt = Field(ge=2, le=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    residues: tuple[StrictInt, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_canonical_residue_set(self) -> Self:
        if len(set(self.residues)) != len(self.residues):
            raise ValueError("PDS residues must be unique")
        if any(residue < 0 or residue >= self.modulus for residue in self.residues):
            raise ValueError("PDS residues must be canonical modulo the modulus")
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


class CyclicDifferenceMultiplicity(ContractModel):
    residue: StrictInt = Field(ge=1, lt=MAX_CYCLIC_DIFFERENCE_SET_MODULUS)
    multiplicity: StrictInt = Field(ge=0, le=4_096)


class CyclicPerfectDifferenceSetResult(ContractModel):
    """Complete nonzero cyclic difference profile and exact PDS decision."""

    semantics_version: Literal["cyclic-perfect-difference-set.v1"]
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
    backend: Literal["python-stdlib"] = "python-stdlib"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_complete_cyclic_profile(self) -> Self:
        residues = self.normalized_residues
        if residues != tuple(sorted(set(residues))):
            raise ValueError("normalized PDS residues must be sorted and unique")
        if any(residue < 0 or residue >= self.modulus for residue in residues):
            raise ValueError("normalized PDS residues must lie in the modulus")
        if self.order != len(residues):
            raise ValueError("PDS order must equal the residue-set cardinality")
        if self.expected_modulus != self.order * (self.order - 1) + 1:
            raise ValueError("expected_modulus must equal k(k-1)+1")
        profile = self.difference_multiplicities
        if tuple(item.residue for item in profile) != tuple(range(1, self.modulus)):
            raise ValueError(
                "cyclic difference profile must cover every nonzero residue"
            )
        recomputed = _cyclic_difference_multiplicities(residues, self.modulus)
        if any(item.multiplicity != recomputed[item.residue] for item in profile):
            raise ValueError(
                "cyclic difference multiplicities must be derived from the residues"
            )
        missing = tuple(item.residue for item in profile if item.multiplicity == 0)
        repeated = tuple(item.residue for item in profile if item.multiplicity > 1)
        if self.missing_residues != missing or self.repeated_residues != repeated:
            raise ValueError("missing and repeated residues must match the profile")
        expected_perfect = (
            self.modulus == self.expected_modulus and not missing and not repeated
        )
        if self.is_perfect != expected_perfect:
            raise ValueError("PDS decision must match the complete residue profile")
        return self


class CyclicDifferenceSetExtensionRequest(ContractModel):
    """A fixed-order direct-containment question in the derived cyclic group."""

    base_elements: tuple[AdditiveInteger, ...] = Field(min_length=1, max_length=64)
    target_order: StrictInt = Field(ge=2, le=64)

    @model_validator(mode="after")
    def require_bounded_complete_candidate_space(self) -> Self:
        if len(set(self.base_elements)) != len(self.base_elements):
            raise ValueError("extension base elements must be unique")
        modulus = self.target_order * (self.target_order - 1) + 1
        if modulus > MAX_CYCLIC_DIFFERENCE_SET_MODULUS:
            raise ValueError("derived extension modulus exceeds the supported bound")
        base_residues = {int(value) % modulus for value in self.base_elements}
        additional = self.target_order - len(base_residues)
        if additional < 0:
            raise ValueError("target_order is smaller than the reduced base set")
        if additional > MAX_DIFFERENCE_SET_ADDITIONAL_ELEMENTS:
            raise ValueError("extension request requires too many added elements")
        candidates = math.comb(modulus - len(base_residues), additional)
        if candidates > MAX_DIFFERENCE_SET_EXTENSION_CANDIDATES:
            raise ValueError(
                "extension candidate space exceeds the complete-search bound"
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
        raise ValueError("extension modulus must equal k(k-1)+1")
    if base_residues != tuple(sorted(set(base_residues))):
        raise ValueError("base residues must be sorted and unique")
    if any(residue < 0 or residue >= modulus for residue in base_residues):
        raise ValueError("base residues must lie in the derived modulus")
    additional = target_order - len(base_residues)
    if additional < 0 or additional > MAX_DIFFERENCE_SET_ADDITIONAL_ELEMENTS:
        raise ValueError(
            "extension result lies outside the supported added-element bound"
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
        raise ValueError("positive extension decisions require witness coverage")
    if extension != tuple(sorted(set(extension))):
        raise ValueError("extension witness must be sorted and unique")
    if len(extension) != target_order:
        raise ValueError("extension witness must have target_order residues")
    if any(residue < 0 or residue >= modulus for residue in extension):
        raise ValueError("extension witness residues must lie in the derived modulus")
    if not set(base_residues) <= set(extension):
        raise ValueError("extension witness must contain the reduced base set")
    if not _is_perfect_difference_set(extension, modulus):
        raise ValueError(
            "extension witness must be a perfect difference set of the derived modulus"
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


class CyclicDifferenceSetExtensionResult(ContractModel):
    """A witness or complete negative decision for one fixed PDS order."""

    semantics_version: Literal["cyclic-pds-extension.fixed-order.v1"]
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
    backend: Literal["python-stdlib"] = "python-stdlib"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_fixed_order_scope_and_decision_shape(self) -> Self:
        expected_candidates = _extension_result_candidate_count(
            target_order=self.target_order,
            modulus=self.modulus,
            base_residues=self.base_residues,
        )
        if self.candidate_space_size != expected_candidates:
            raise ValueError(
                "candidate_space_size must cover the exact combination space"
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
            raise ValueError(
                "negative extension decisions require empty witness and full coverage"
            )
        else:
            if (
                _find_extension_witness(
                    self.base_residues, self.target_order, self.modulus
                )
                is not None
            ):
                raise ValueError(
                    "negative extension decision must match the exhaustive search"
                )
        return self


class IntegerListRequest(ContractModel):
    values: tuple[CanonicalInteger, ...] = Field(min_length=1, max_length=_MAX_PARTS)

    @model_validator(mode="after")
    def require_nonnegative_parts(self) -> Self:
        if any(parse_canonical_integer(value) < 0 for value in self.values):
            raise ValueError("integer list values must be nonnegative")
        return self


class IntegerResult(ContractModel):
    value: CanonicalInteger


class RationalResult(ContractModel):
    value: CanonicalRational


class FibonacciPairResult(ContractModel):
    """Two consecutive Fibonacci values forming one recurrence boundary."""

    n: StrictInt = Field(ge=0, le=10_000)
    f_n: CanonicalInteger
    f_n_plus_one: CanonicalInteger


class FibonacciPairRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=10_000)


class IntegerPartitionEnumerationRequest(ContractModel):
    """Enumerate every partition of n containing at most max_parts summands."""

    n: StrictInt = Field(ge=0, le=_MAX_PARTITION_N)
    max_parts: StrictInt = Field(ge=1, le=_MAX_PARTITION_N)


class IntegerPartitionEnumerationResult(ContractModel):
    """Complete canonical partition enumeration for one bounded request."""

    n: StrictInt = Field(ge=0, le=_MAX_PARTITION_N)
    max_parts: StrictInt = Field(ge=1, le=_MAX_PARTITION_N)
    partitions: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=_MAX_ENUMERATED_PARTITIONS
    )

    @model_validator(mode="after")
    def require_canonical_complete_items(self) -> Self:
        previous: tuple[int, ...] | None = None
        for partition in self.partitions:
            if len(partition) > self.max_parts:
                raise ValueError("partition exceeds max_parts")
            if any(part <= 0 for part in partition):
                raise ValueError("partition parts must be positive")
            if tuple(sorted(partition, reverse=True)) != partition:
                raise ValueError("partition parts must be nonincreasing")
            if sum(partition) != self.n:
                raise ValueError("partition parts must sum to n")
            if previous is not None and previous <= partition:
                raise ValueError(
                    "partitions must be unique in descending lexicographic order"
                )
            previous = tuple(partition)
        if self.n == 0 and self.partitions != ((),):
            raise ValueError("zero has exactly one empty partition")
        return self


class LinearRecurrenceEvaluationRequest(ContractModel):
    """Evaluate a bounded exact constant-coefficient recurrence.

    ``coefficients[j - 1]`` multiplies ``a[n - j]``. The initial vector is
    exactly ``a[0], ..., a[d - 1]`` for recurrence order ``d``.
    """

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_ORDER,
    )
    initial_values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_ORDER,
    )
    coefficient_convention: Literal["A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"]
    scope: Literal["PREFIX", "INDICES"]
    term_count: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    indices: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_LINEAR_RECURRENCE_REQUESTED_INDICES,
    )

    @model_validator(mode="after")
    def require_bounded_explicit_scope(self) -> Self:
        if len(self.initial_values) != len(self.coefficients):
            raise ValueError("initial_values length must equal the recurrence order")
        for label, values in (
            ("recurrence coefficient", self.coefficients),
            ("recurrence initial value", self.initial_values),
        ):
            for value in values:
                require_bounded_rational(
                    value,
                    max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
                    label=label,
                )
        if self.scope == "PREFIX":
            if self.term_count is None or self.indices:
                raise ValueError("PREFIX scope requires term_count and forbids indices")
        else:
            if self.term_count is not None or not self.indices:
                raise ValueError(
                    "INDICES scope requires indices and forbids term_count"
                )
            if any(
                index < 0 or index > MAX_LINEAR_RECURRENCE_INDEX
                for index in self.indices
            ):
                raise ValueError(
                    f"indices must lie between 0 and {MAX_LINEAR_RECURRENCE_INDEX}"
                )
            if any(left >= right for left, right in pairwise(self.indices)):
                raise ValueError("indices must be strictly increasing")
        requested_indices = (
            tuple(range(self.term_count))
            if self.scope == "PREFIX" and self.term_count is not None
            else self.indices
        )
        _validate_recurrence_result_budget(
            coefficients=self.coefficients,
            initial_values=self.initial_values,
            coefficient_convention=self.coefficient_convention,
            scope=self.scope,
            requested_indices=requested_indices,
        )
        return self


class IndexedRationalValue(ContractModel):
    index: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    value: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        require_bounded_rational(
            self.value,
            max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
            label="recurrence result",
        )
        return self


class LinearRecurrenceEvaluationResult(ContractModel):
    coefficient_convention: Literal["A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"]
    scope: Literal["PREFIX", "INDICES"]
    values: tuple[IndexedRationalValue, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    replay_prefix: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1,
    )
    replay_scope_end: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["sympy"] = "sympy"
    backend_version: Literal["1.14.0"] = "1.14.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def require_complete_replay_prefix(self) -> Self:
        if len(self.replay_prefix) != self.replay_scope_end + 1:
            raise ValueError(
                "replay_prefix must cover indices 0 through replay_scope_end"
            )
        for value in self.replay_prefix:
            require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                label="recurrence replay value",
            )
        indices = tuple(item.index for item in self.values)
        if any(left >= right for left, right in pairwise(indices)):
            raise ValueError("result indices must be strictly increasing")
        if indices[-1] != self.replay_scope_end:
            raise ValueError("the greatest requested index must bind replay_scope_end")
        if any(item.value != self.replay_prefix[item.index] for item in self.values):
            raise ValueError("indexed values must match the recurrence replay prefix")
        if self.scope == "PREFIX" and indices != tuple(
            range(self.replay_scope_end + 1)
        ):
            raise ValueError(
                "PREFIX results must contain consecutive indices from zero"
            )
        return self


class PolynomialCoefficientRecurrenceEvaluationRequest(ContractModel):
    """Evaluate a bounded exact polynomial-coefficient linear recurrence.

    ``coefficient_polynomials[j]`` is the ascending coefficient vector of
    ``p_j(n)`` in ``sum_{j=0}^d p_j(n) a[n-j] = 0``.  The initial vector is
    exactly ``a[0], ..., a[d-1]``.
    """

    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=2, max_length=MAX_LINEAR_RECURRENCE_ORDER + 1
    )
    initial_values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_ORDER
    )
    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ]
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"]
    scope: Literal["PREFIX", "INDICES"]
    term_count: StrictInt | None = Field(
        default=None, ge=1, le=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    indices: tuple[StrictInt, ...] = Field(
        default=(), max_length=MAX_LINEAR_RECURRENCE_REQUESTED_INDICES
    )

    @model_validator(mode="after")
    def require_bounded_regular_scope(self) -> Self:
        order = len(self.coefficient_polynomials) - 1
        if len(self.initial_values) != order:
            raise ValueError("initial_values length must equal the recurrence order")
        for polynomial in self.coefficient_polynomials:
            if (
                not polynomial
                or len(polynomial) > MAX_P_RECURSIVE_POLYNOMIAL_DEGREE + 1
            ):
                raise ValueError("coefficient polynomial degree is outside the bound")
            _require_canonical_polynomial(
                polynomial, label="recurrence polynomial coefficient"
            )
        for value in self.initial_values:
            require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_INPUT_RATIONAL_DIGITS,
                label="recurrence initial value",
            )
        if self.scope == "PREFIX":
            if self.term_count is None or self.indices:
                raise ValueError("PREFIX scope requires term_count and forbids indices")
            requested = tuple(range(self.term_count))
        else:
            if self.term_count is not None or not self.indices:
                raise ValueError(
                    "INDICES scope requires indices and forbids term_count"
                )
            if any(
                index < 0 or index > MAX_LINEAR_RECURRENCE_INDEX
                for index in self.indices
            ):
                raise ValueError("indices are outside the recurrence bound")
            if any(left >= right for left, right in pairwise(self.indices)):
                raise ValueError("indices must be strictly increasing")
            requested = self.indices
        _validate_p_recursive_result_budget(
            coefficient_polynomials=self.coefficient_polynomials,
            initial_values=self.initial_values,
            coefficient_convention=self.coefficient_convention,
            polynomial_convention=self.polynomial_convention,
            scope=self.scope,
            requested_indices=requested,
        )
        return self


class PolynomialCoefficientRecurrenceEvaluationResult(ContractModel):
    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ]
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"]
    scope: Literal["PREFIX", "INDICES"]
    recurrence_order: StrictInt = Field(ge=1, le=MAX_LINEAR_RECURRENCE_ORDER)
    values: tuple[IndexedRationalValue, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    replay_prefix: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    residuals: tuple[IndexedRationalValue, ...] = Field(
        max_length=MAX_LINEAR_RECURRENCE_INDEX + 1
    )
    replay_scope_end: StrictInt = Field(ge=0, le=MAX_LINEAR_RECURRENCE_INDEX)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["sympy"] = "sympy"
    backend_version: Literal["1.14.0"] = "1.14.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def require_complete_replay(self) -> Self:
        if len(self.replay_prefix) != self.replay_scope_end + 1:
            raise ValueError("replay_prefix must cover the complete bounded scope")
        indices = tuple(item.index for item in self.values)
        if any(left >= right for left, right in pairwise(indices)):
            raise ValueError("result indices must be strictly increasing")
        if indices[-1] != self.replay_scope_end:
            raise ValueError("the greatest requested index must bind replay_scope_end")
        if any(item.value != self.replay_prefix[item.index] for item in self.values):
            raise ValueError("indexed values must match the recurrence replay prefix")
        if self.scope == "PREFIX" and indices != tuple(range(len(indices))):
            raise ValueError(
                "PREFIX results must contain consecutive indices from zero"
            )
        for value in self.replay_prefix:
            require_bounded_rational(
                value,
                max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                label="polynomial-coefficient recurrence replay value",
            )
        residual_indices = tuple(item.index for item in self.residuals)
        if residual_indices != tuple(
            range(self.recurrence_order, self.replay_scope_end + 1)
        ):
            raise ValueError(
                "residuals must cover every recurrence step through replay_scope_end"
            )
        if any(item.value.as_fraction() != 0 for item in self.residuals):
            raise ValueError("every recurrence residual must be exactly zero")
        return self


class RationalGeneratingFunctionCoefficientsRequest(ContractModel):
    """Expand N(x)/D(x) at zero through one explicit finite order."""

    numerator: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_GENERATING_FUNCTION_DEGREE + 1,
    )
    denominator: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_GENERATING_FUNCTION_DEGREE + 1,
    )
    coefficient_convention: Literal["ASCENDING_POWERS_OF_X"]
    expansion_point: Literal["0"]
    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )

    @model_validator(mode="after")
    def require_regular_canonical_input(self) -> Self:
        _require_canonical_polynomial(self.numerator, label="numerator coefficient")
        _require_canonical_polynomial(
            self.denominator,
            label="denominator coefficient",
        )
        if self.denominator[0].as_fraction() == 0:
            raise ValueError("denominator constant coefficient must be nonzero")
        _validate_series_result_budget(
            numerator=self.numerator,
            denominator=self.denominator,
            coefficient_convention=self.coefficient_convention,
            expansion_point=self.expansion_point,
            truncation_order=self.truncation_order,
        )
        return self


class RationalGeneratingFunctionCoefficientsResult(ContractModel):
    coefficient_convention: Literal["ASCENDING_POWERS_OF_X"]
    expansion_point: Literal["0"]
    truncation_order: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )
    residual_congruence: Literal[
        "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
    ]
    residual_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_SERIES_TRUNCATION_ORDER,
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["sympy"] = "sympy"
    backend_version: Literal["1.14.0"] = "1.14.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def require_exact_finite_truncation(self) -> Self:
        if (
            len(self.coefficients) != self.truncation_order
            or len(self.residual_coefficients) != self.truncation_order
        ):
            raise ValueError(
                "coefficient and residual vectors must equal truncation_order"
            )
        for label, values in (
            ("series coefficient", self.coefficients),
            ("series residual", self.residual_coefficients),
        ):
            for value in values:
                require_bounded_rational(
                    value,
                    max_digits=MAX_COMBINATORICS_RESULT_RATIONAL_DIGITS,
                    label=label,
                )
        if any(value.as_fraction() != 0 for value in self.residual_coefficients):
            raise ValueError("residual coefficients must vanish through the truncation")
        return self
