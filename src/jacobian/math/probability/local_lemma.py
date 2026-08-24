"""Exact numerical witnesses for the finite asymmetric local lemma."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from jacobian.canonical import format_canonical_integer
from jacobian.math._rational_height import RationalHeight, sum_heights

# Conservative materialized-source ceilings.  The kernel and every result
# replay make one linear pass over these rows and directed incidences.
MAX_LOCAL_LEMMA_EVENTS = 1_024
MAX_LOCAL_LEMMA_INCIDENCES = 32_768
MAX_LOCAL_LEMMA_LABEL_LENGTH = 128
MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS = 256
# Per-component growth is derived before any product.  The aggregate ceiling
# leaves room in the 10 MiB canonical transport envelope for the retained
# source, labels, incidence indices, row metadata, and JSON punctuation.
MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS = 32_768
MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS = 2_000_000

_MAX_INPUT_RATIONAL_MAGNITUDE = 10**MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS


def _decimal_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _require_fraction(value: object, *, label: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"native {label} must be a Fraction")
    if (
        abs(value.numerator) >= _MAX_INPUT_RATIONAL_MAGNITUDE
        or value.denominator >= _MAX_INPUT_RATIONAL_MAGNITUDE
    ):
        raise ValueError(
            f"{label} exceeds the "
            f"{MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS}-digit input bound"
        )
    return value


def _rational_height(value: Fraction) -> RationalHeight:
    return RationalHeight(
        _decimal_digits(value.numerator),
        _decimal_digits(value.denominator),
    )


def _product_height(factors: Iterable[Fraction]) -> RationalHeight:
    height = RationalHeight(1, 1)
    for factor in factors:
        height = height.product(_rational_height(factor))
    return height


def _require_result_digit_budget(
    probability_upper_bounds: tuple[Fraction, ...],
    witness_parameters: tuple[Fraction, ...],
    neighborhoods: tuple[tuple[int, ...], ...],
) -> None:
    """Bound every exact intermediate and the complete serialized row ledger."""

    complements = tuple(Fraction(1) - value for value in witness_parameters)
    total_result_digits = 0
    for event_index, neighbors in enumerate(neighborhoods):
        product_height = _product_height(complements[index] for index in neighbors)
        witness = witness_parameters[event_index]
        right_hand_side_height = (
            RationalHeight(1, 1)
            if witness == 0
            else _rational_height(witness).product(product_height)
        )
        probability = probability_upper_bounds[event_index]
        probability_height = _rational_height(probability)
        slack_height = (
            probability_height
            if witness == 0
            else (
                right_hand_side_height
                if probability == 0
                else sum_heights((right_hand_side_height, probability_height))
            )
        )
        row_heights = (product_height, right_hand_side_height, slack_height)
        if any(
            height.exceeds(MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS)
            for height in row_heights
        ):
            raise ValueError(
                f"event {event_index} can exceed the "
                f"{MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS}-digit exact-result "
                "component bound"
            )
        total_result_digits += sum(
            height.numerator_digits + height.denominator_digits
            for height in row_heights
        )
        if total_result_digits > MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS:
            raise ValueError(
                "asymmetric local-lemma inequality rows can exceed the "
                f"{MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS}-digit aggregate "
                "exact-result bound"
            )

    witness_product_height = _product_height(complements)
    if witness_product_height.exceeds(MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS):
        raise ValueError(
            "the all-event witness product can exceed the "
            f"{MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS}-digit exact-result "
            "component bound"
        )
    total_result_digits += (
        witness_product_height.numerator_digits
        + witness_product_height.denominator_digits
    )
    if total_result_digits > MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS:
        raise ValueError(
            "the complete asymmetric local-lemma result can exceed the "
            f"{MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS}-digit aggregate "
            "exact-result bound"
        )


def _require_event_axis(event_labels: tuple[str, ...]) -> int:
    if type(event_labels) is not tuple:
        raise TypeError("native event_labels must be a tuple")
    event_count = len(event_labels)
    if event_count > MAX_LOCAL_LEMMA_EVENTS:
        raise ValueError(
            "asymmetric local-lemma source exceeds the "
            f"{MAX_LOCAL_LEMMA_EVENTS}-event bound"
        )
    if any(
        type(label) is not str or not 1 <= len(label) <= MAX_LOCAL_LEMMA_LABEL_LENGTH
        for label in event_labels
    ):
        raise ValueError(
            "event labels must be nonempty strings of at most "
            f"{MAX_LOCAL_LEMMA_LABEL_LENGTH} characters"
        )
    if any(not unicodedata.is_normalized("NFC", label) for label in event_labels):
        raise ValueError("event labels must use Unicode NFC")
    if len(set(event_labels)) != event_count:
        raise ValueError("event labels must be unique on the ordered axis")
    return event_count


def _require_aligned_tuples(
    event_count: int,
    probability_upper_bounds: tuple[Fraction, ...],
    witness_parameters: tuple[Fraction, ...],
    neighborhoods: tuple[tuple[int, ...], ...],
) -> None:
    if any(
        type(values) is not tuple
        for values in (
            probability_upper_bounds,
            witness_parameters,
            neighborhoods,
        )
    ):
        raise TypeError(
            "native probability, witness, and neighborhood axes must be tuples"
        )
    if (
        len(probability_upper_bounds) != event_count
        or len(witness_parameters) != event_count
        or len(neighborhoods) != event_count
    ):
        raise ValueError(
            "probability bounds, witness parameters, and neighborhoods must "
            "align with the event axis"
        )


def _require_probability_and_witness_domains(
    probability_upper_bounds: tuple[Fraction, ...],
    witness_parameters: tuple[Fraction, ...],
) -> None:
    for index, probability in enumerate(probability_upper_bounds):
        value = _require_fraction(
            probability,
            label=f"probability_upper_bounds[{index}]",
        )
        if not 0 <= value <= 1:
            raise ValueError("probability upper bounds must lie in [0, 1]")
    for index, witness in enumerate(witness_parameters):
        value = _require_fraction(
            witness,
            label=f"witness_parameters[{index}]",
        )
        if not 0 <= value < 1:
            raise ValueError("witness parameters must lie in [0, 1)")


def _require_neighborhoods(
    neighborhoods: tuple[tuple[int, ...], ...],
    event_count: int,
) -> None:
    incidence_count = 0
    for event_index, neighbors in enumerate(neighborhoods):
        if type(neighbors) is not tuple:
            raise TypeError("native neighborhoods must be tuples of indices")
        if any(type(index) is not int for index in neighbors):
            raise TypeError("native neighborhood indices must be integers")
        if any(index < 0 or index >= event_count for index in neighbors):
            raise ValueError(
                f"neighborhood {event_index} contains an index outside the event axis"
            )
        if any(left >= right for left, right in pairwise(neighbors)):
            raise ValueError(
                "each directed neighborhood must be a strictly increasing set of "
                "event indices"
            )
        incidence_count += len(neighbors)
        if incidence_count > MAX_LOCAL_LEMMA_INCIDENCES:
            raise ValueError(
                "asymmetric local-lemma source exceeds the "
                f"{MAX_LOCAL_LEMMA_INCIDENCES}-incidence work bound"
            )


@dataclass(frozen=True, slots=True)
class AsymmetricLocalLemmaWitness:
    """One materialized directed finite local-lemma numerical witness.

    The ordered labels define the event axis.  Probabilities, witnesses, and
    neighborhoods are aligned to that axis.  ``neighborhoods[i]`` is the
    strictly increasing directed index set Gamma(i).  A listed self-index is
    allowed and contributes its complement factor exactly once.
    """

    event_labels: tuple[str, ...]
    probability_upper_bounds: tuple[Fraction, ...]
    witness_parameters: tuple[Fraction, ...]
    neighborhoods: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        event_count = _require_event_axis(self.event_labels)
        _require_aligned_tuples(
            event_count,
            self.probability_upper_bounds,
            self.witness_parameters,
            self.neighborhoods,
        )
        _require_probability_and_witness_domains(
            self.probability_upper_bounds,
            self.witness_parameters,
        )
        _require_neighborhoods(self.neighborhoods, event_count)
        _require_result_digit_budget(
            self.probability_upper_bounds,
            self.witness_parameters,
            self.neighborhoods,
        )


@dataclass(frozen=True, slots=True)
class AsymmetricLocalLemmaInequality:
    """One exact reconstructed asymmetric local-lemma inequality."""

    event_index: int
    neighborhood_product: Fraction
    right_hand_side: Fraction
    slack: Fraction
    inequality_holds: bool

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("local-lemma inequality index must be nonnegative")
        for label, value in (
            ("neighborhood product", self.neighborhood_product),
            ("right-hand side", self.right_hand_side),
            ("slack", self.slack),
        ):
            if type(value) is not Fraction:
                raise TypeError(f"native local-lemma {label} must be a Fraction")
            if (
                _decimal_digits(value.numerator)
                > MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS
                or _decimal_digits(value.denominator)
                > MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS
            ):
                raise ValueError(
                    f"local-lemma {label} exceeds the exact-result digit bound"
                )
        if type(self.inequality_holds) is not bool:
            raise TypeError("native inequality_holds must be a bool")
        if self.inequality_holds != (self.slack >= 0):
            raise ValueError("inequality_holds must match the exact slack sign")


def _inequalities(
    source: AsymmetricLocalLemmaWitness,
) -> tuple[AsymmetricLocalLemmaInequality, ...]:
    complements = tuple(Fraction(1) - value for value in source.witness_parameters)
    rows: list[AsymmetricLocalLemmaInequality] = []
    for event_index, neighbors in enumerate(source.neighborhoods):
        neighborhood_product = Fraction(1)
        for neighbor_index in neighbors:
            neighborhood_product *= complements[neighbor_index]
        right_hand_side = source.witness_parameters[event_index] * neighborhood_product
        slack = right_hand_side - source.probability_upper_bounds[event_index]
        rows.append(
            AsymmetricLocalLemmaInequality(
                event_index=event_index,
                neighborhood_product=neighborhood_product,
                right_hand_side=right_hand_side,
                slack=slack,
                inequality_holds=slack >= 0,
            )
        )
    return tuple(rows)


def _witness_product(source: AsymmetricLocalLemmaWitness) -> Fraction:
    product = Fraction(1)
    for witness in source.witness_parameters:
        product *= 1 - witness
    return product


@dataclass(frozen=True, slots=True)
class AsymmetricLocalLemmaWitnessCheckResult:
    """A source-bound exact numerical-witness decision and complete row ledger."""

    source: AsymmetricLocalLemmaWitness
    inequalities: tuple[AsymmetricLocalLemmaInequality, ...]
    failed_event_indices: tuple[int, ...]
    valid: bool
    witness_product: Fraction

    def __post_init__(self) -> None:
        if type(self.source) is not AsymmetricLocalLemmaWitness:
            raise TypeError("local-lemma check source has the wrong native type")
        if type(self.inequalities) is not tuple or any(
            type(row) is not AsymmetricLocalLemmaInequality for row in self.inequalities
        ):
            raise TypeError("native local-lemma inequalities must be a tuple of rows")
        if type(self.failed_event_indices) is not tuple or any(
            type(index) is not int for index in self.failed_event_indices
        ):
            raise TypeError("native failed event indices must be a tuple of integers")
        expected_rows = _inequalities(self.source)
        if self.inequalities != expected_rows:
            raise ValueError(
                "local-lemma inequality ledger does not reconstruct from its source"
            )
        expected_failures = tuple(
            row.event_index for row in expected_rows if not row.inequality_holds
        )
        if self.failed_event_indices != expected_failures:
            raise ValueError("failed event indices do not match the inequality ledger")
        if type(self.valid) is not bool or self.valid != (not expected_failures):
            raise ValueError("local-lemma validity must match all exact slacks")
        expected_product = _witness_product(self.source)
        if type(self.witness_product) is not Fraction:
            raise TypeError("native witness product must be a Fraction")
        if self.witness_product != expected_product:
            raise ValueError("witness product does not reconstruct from the source")


def check_asymmetric_local_lemma_witness(
    source: AsymmetricLocalLemmaWitness,
) -> AsymmetricLocalLemmaWitnessCheckResult:
    """Check every exact finite asymmetric local-lemma witness inequality."""

    if type(source) is not AsymmetricLocalLemmaWitness:
        raise TypeError("source must be an AsymmetricLocalLemmaWitness")
    inequalities = _inequalities(source)
    failures = tuple(
        row.event_index for row in inequalities if not row.inequality_holds
    )
    return AsymmetricLocalLemmaWitnessCheckResult(
        source=source,
        inequalities=inequalities,
        failed_event_indices=failures,
        valid=not failures,
        witness_product=_witness_product(source),
    )


__all__ = [
    "AsymmetricLocalLemmaInequality",
    "AsymmetricLocalLemmaWitness",
    "AsymmetricLocalLemmaWitnessCheckResult",
    "check_asymmetric_local_lemma_witness",
]
