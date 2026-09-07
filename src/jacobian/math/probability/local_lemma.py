"""Canonical numerical witnesses for the finite asymmetric local lemma."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Any, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math._rational_height import RationalHeight, sum_heights

# Conservative materialized-source ceilings.  The kernel and every result
# replay make one linear pass over these rows and directed incidences.
MAX_LOCAL_LEMMA_EVENTS = 1_024
MAX_LOCAL_LEMMA_INCIDENCES = 32_768
MAX_LOCAL_LEMMA_LABEL_LENGTH = 128
MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS = 256
# Per-component growth is derived before any product. The aggregate digit
# ceiling bounds the retained exact rational components.
MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS = 32_768
MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS = 2_000_000

_MAX_INPUT_RATIONAL_MAGNITUDE = 10**MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS


def _decimal_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


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
) -> tuple[Fraction, ...]:
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
            raise OperationResourceAdmissionError(
                location=("source",),
                code="probability.local_lemma_result_bound",
                message=f"event {event_index} can exceed the "
                f"{MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS}-digit exact-result "
                "component bound",
            )
        total_result_digits += sum(
            height.numerator_digits + height.denominator_digits
            for height in row_heights
        )
        if total_result_digits > MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("source",),
                code="probability.local_lemma_result_bound",
                message="asymmetric local-lemma inequality rows can exceed the "
                f"{MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS}-digit aggregate "
                "exact-result bound",
            )

    witness_product_height = _product_height(complements)
    if witness_product_height.exceeds(MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS):
        raise OperationResourceAdmissionError(
            location=("source",),
            code="probability.local_lemma_result_bound",
            message="the all-event witness product can exceed the "
            f"{MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS}-digit exact-result "
            "component bound",
        )
    total_result_digits += (
        witness_product_height.numerator_digits
        + witness_product_height.denominator_digits
    )
    if total_result_digits > MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="probability.local_lemma_result_bound",
            message="the complete asymmetric local-lemma result can exceed the "
            f"{MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS}-digit aggregate "
            "exact-result bound",
        )

    return complements


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
    probability_upper_bounds: tuple[CanonicalRational, ...],
    witness_parameters: tuple[CanonicalRational, ...],
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
    probability_upper_bounds: tuple[CanonicalRational, ...],
    witness_parameters: tuple[CanonicalRational, ...],
) -> None:
    for values, witness in (
        (probability_upper_bounds, False),
        (witness_parameters, True),
    ):
        for value in values:
            if (
                abs(value.num) >= _MAX_INPUT_RATIONAL_MAGNITUDE
                or value.den >= _MAX_INPUT_RATIONAL_MAGNITUDE
            ):
                raise ValueError(
                    "probability or witness exceeds the 256-digit input bound"
                )
            if (
                value.num < 0
                or value.num > value.den
                or (witness and value.num == value.den)
            ):
                raise ValueError(
                    "witness parameters must lie in [0, 1)"
                    if witness
                    else "probability upper bounds must lie in [0, 1]"
                )


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


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.local_lemma_invariant", message)


LocalLemmaEventLabel = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_LOCAL_LEMMA_LABEL_LENGTH,
        strict=True,
    ),
]
LocalLemmaNeighborhood = Annotated[
    tuple[StrictInt, ...],
    Field(max_length=MAX_LOCAL_LEMMA_EVENTS),
]


def _bound_raw_rational(
    value: object,
    *,
    max_digits: int,
    label: str,
) -> int:
    if not isinstance(value, Mapping):
        return 0
    total = 0
    for component in ("num", "den"):
        raw_component = value.get(component)
        if isinstance(raw_component, str):
            sign_characters = int(raw_component.startswith("-"))
            digits = len(raw_component) - sign_characters
            if digits > max_digits:
                raise _validation_error(f"{label} exceeds the {max_digits}-digit bound")
            total += digits
    return total


def _bound_raw_source(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    sequence_bounds = {
        "event_labels": MAX_LOCAL_LEMMA_EVENTS,
        "probability_upper_bounds": MAX_LOCAL_LEMMA_EVENTS,
        "witness_parameters": MAX_LOCAL_LEMMA_EVENTS,
        "neighborhoods": MAX_LOCAL_LEMMA_EVENTS,
    }
    for field_name, maximum in sequence_bounds.items():
        raw = value.get(field_name)
        if isinstance(raw, (list, tuple)) and len(raw) > maximum:
            raise _validation_error(
                f"{field_name} exceeds the {MAX_LOCAL_LEMMA_EVENTS}-event bound"
            )
    for field_name in ("probability_upper_bounds", "witness_parameters"):
        raw_values = value.get(field_name)
        if isinstance(raw_values, (list, tuple)):
            for index, raw_value in enumerate(raw_values):
                _bound_raw_rational(
                    raw_value,
                    max_digits=MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS,
                    label=f"{field_name}[{index}]",
                )
    raw_neighborhoods = value.get("neighborhoods")
    if isinstance(raw_neighborhoods, (list, tuple)):
        incidence_count = 0
        for index, raw_neighborhood in enumerate(raw_neighborhoods):
            if not isinstance(raw_neighborhood, (list, tuple)):
                continue
            if len(raw_neighborhood) > MAX_LOCAL_LEMMA_EVENTS:
                raise _validation_error(
                    f"neighborhoods[{index}] exceeds the event-axis bound"
                )
            incidence_count += len(raw_neighborhood)
            if incidence_count > MAX_LOCAL_LEMMA_INCIDENCES:
                raise _validation_error(
                    "neighborhoods exceed the "
                    f"{MAX_LOCAL_LEMMA_INCIDENCES}-incidence work bound"
                )
    prepared = dict(value)
    for field_name in (
        "event_labels",
        "probability_upper_bounds",
        "witness_parameters",
    ):
        raw = prepared.get(field_name)
        if isinstance(raw, list):
            prepared[field_name] = tuple(raw)
    if isinstance(raw_neighborhoods, (list, tuple)):
        prepared["neighborhoods"] = tuple(
            tuple(row) if isinstance(row, list) else row for row in raw_neighborhoods
        )
    return prepared


class AsymmetricLocalLemmaWitness(StrictModel):
    """One canonical materialized finite asymmetric local-lemma witness.

    All four tuples share the ordered ``event_labels`` axis.  Neighborhoods
    are directed: ``neighborhoods[i]`` is Gamma(i), encoded as strictly
    increasing event indices.  A listed self-index is allowed and contributes
    ``1 - witness_parameters[i]`` exactly once.  No graph-independence claim is
    part of this value.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "One materialized finite asymmetric local-lemma numerical "
                "witness. All tuples share the ordered event_labels axis. "
                "Each neighborhoods[i] is the strictly increasing directed "
                "index set Gamma(i); listed self-indices are allowed and are "
                "multiplied once. Probability bounds lie in [0,1], witness "
                "parameters lie in [0,1), and this value makes no dependency-"
                "graph or independence claim."
            )
        }
    )

    event_labels: tuple[LocalLemmaEventLabel, ...] = Field(
        max_length=MAX_LOCAL_LEMMA_EVENTS,
        description=(
            "Unique Unicode-NFC labels forming the authoritative ordered event "
            "axis; the empty axis is allowed."
        ),
    )
    probability_upper_bounds: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_LOCAL_LEMMA_EVENTS,
        description=(
            "Exact p_i values in [0,1], aligned to event_labels; every rational "
            f"component has at most {MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS} "
            "decimal digits."
        ),
    )
    witness_parameters: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_LOCAL_LEMMA_EVENTS,
        description=(
            "Exact x_i values in [0,1), aligned to event_labels; every rational "
            f"component has at most {MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS} "
            "decimal digits."
        ),
    )
    neighborhoods: tuple[LocalLemmaNeighborhood, ...] = Field(
        max_length=MAX_LOCAL_LEMMA_EVENTS,
        description=(
            "Directed Gamma(i) sets aligned to event_labels. Each inner tuple is "
            "strictly increasing, contains only declared event indices, may "
            "include i itself, and contributes each listed complement once. The "
            f"complete family has at most {MAX_LOCAL_LEMMA_INCIDENCES} incidences."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source(cls, value: Any) -> Any:
        """Reject oversized materialized input before nested rational parsing."""

        value = canonicalize_json_containers(value)

        return _bound_raw_source(value)

    @model_validator(mode="after")
    def require_structure(self) -> Self:
        event_count = _require_event_axis(self.event_labels)
        _require_aligned_tuples(
            event_count,
            self.probability_upper_bounds,
            self.witness_parameters,
            self.neighborhoods,
        )
        _require_probability_and_witness_domains(
            self.probability_upper_bounds, self.witness_parameters
        )
        _require_neighborhoods(self.neighborhoods, event_count)
        return self


class AsymmetricLocalLemmaInequality(StrictModel):
    """One exact row ``p_i <= x_i * product_(j in Gamma(i)) (1-x_j)``."""

    event_index: StrictInt = Field(ge=0, lt=MAX_LOCAL_LEMMA_EVENTS)
    neighborhood_product: CanonicalRational
    right_hand_side: CanonicalRational
    slack: CanonicalRational
    inequality_holds: StrictBool


def _bound_raw_result(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    raw_rows = value.get("inequalities")
    if isinstance(raw_rows, (list, tuple)) and len(raw_rows) > MAX_LOCAL_LEMMA_EVENTS:
        raise _validation_error("inequality ledger exceeds the event-count bound")
    raw_failures = value.get("failed_event_indices")
    if (
        isinstance(raw_failures, (list, tuple))
        and len(raw_failures) > MAX_LOCAL_LEMMA_EVENTS
    ):
        raise _validation_error("failed event indices exceed the event-count bound")
    total_digits = _bound_raw_rational(
        value.get("witness_product"),
        max_digits=MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS,
        label="witness product",
    )
    if isinstance(raw_rows, (list, tuple)):
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                continue
            for field_name in (
                "neighborhood_product",
                "right_hand_side",
                "slack",
            ):
                total_digits += _bound_raw_rational(
                    raw_row.get(field_name),
                    max_digits=MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS,
                    label=f"inequalities[{index}].{field_name}",
                )
                if total_digits > MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS:
                    raise _validation_error(
                        "inequality ledger exceeds the aggregate exact-result "
                        "digit bound"
                    )
    prepared = dict(value)
    if isinstance(raw_rows, list):
        prepared["inequalities"] = tuple(raw_rows)
    if isinstance(raw_failures, list):
        prepared["failed_event_indices"] = tuple(raw_failures)
    return prepared


class AsymmetricLocalLemmaWitnessCheckResult(StrictModel):
    """Source-bound exact numerical result; dependency hypotheses are unchecked."""

    source: AsymmetricLocalLemmaWitness = Field(
        description=(
            "The complete canonical numerical witness whose exact inequalities "
            "were checked."
        )
    )
    inequalities: tuple[AsymmetricLocalLemmaInequality, ...] = Field(
        max_length=MAX_LOCAL_LEMMA_EVENTS,
        description=(
            "Complete event-axis-ordered ledger of exact neighborhood products, "
            "right-hand sides, and slacks."
        ),
    )
    failed_event_indices: tuple[StrictInt, ...] = Field(
        max_length=MAX_LOCAL_LEMMA_EVENTS,
        description="All and only event indices whose exact slack is negative.",
    )
    valid: StrictBool = Field(
        description="True exactly when every returned exact slack is nonnegative."
    )
    witness_product: CanonicalRational = Field(
        description=(
            "Exact product over all i of (1-x_i). For a valid numerical witness, "
            "this is the standard avoidance-probability lower bound only when "
            "the caller separately establishes the local-lemma dependency and "
            "measurability hypotheses."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: Any) -> Any:
        """Bound forged result collections and rationals before nested parsing."""

        value = canonicalize_json_containers(value)

        return _bound_raw_result(value)

    @model_validator(mode="after")
    def require_structural_ledger(self) -> Self:
        if tuple(row.event_index for row in self.inequalities) != tuple(
            range(len(self.source.event_labels))
        ):
            raise _validation_error("inequality rows must cover the ordered event axis")
        if self.failed_event_indices != tuple(
            sorted(set(self.failed_event_indices))
        ) or any(
            index < 0 or index >= len(self.source.event_labels)
            for index in self.failed_event_indices
        ):
            raise _validation_error(
                "failed event indices must be an ordered axis subset"
            )
        return self


def _inequalities(
    source: AsymmetricLocalLemmaWitness,
    probabilities: tuple[Fraction, ...],
    witnesses: tuple[Fraction, ...],
    complements: tuple[Fraction, ...],
) -> tuple[AsymmetricLocalLemmaInequality, ...]:
    rows: list[AsymmetricLocalLemmaInequality] = []
    for event_index, neighbors in enumerate(source.neighborhoods):
        neighborhood_product = Fraction(1)
        for neighbor_index in neighbors:
            neighborhood_product *= complements[neighbor_index]
        right_hand_side = witnesses[event_index] * neighborhood_product
        slack = right_hand_side - probabilities[event_index]
        rows.append(
            AsymmetricLocalLemmaInequality(
                event_index=event_index,
                neighborhood_product=CanonicalRational.from_fraction(
                    neighborhood_product
                ),
                right_hand_side=CanonicalRational.from_fraction(right_hand_side),
                slack=CanonicalRational.from_fraction(slack),
                inequality_holds=slack >= 0,
            )
        )
    return tuple(rows)


def _witness_product(complements: tuple[Fraction, ...]) -> Fraction:
    product = Fraction(1)
    for complement in complements:
        product *= complement
    return product


def check_asymmetric_local_lemma_witness(
    source: AsymmetricLocalLemmaWitness,
) -> AsymmetricLocalLemmaWitnessCheckResult:
    """Check every exact finite asymmetric local-lemma witness inequality."""

    if type(source) is not AsymmetricLocalLemmaWitness:
        raise TypeError("source must be an AsymmetricLocalLemmaWitness")
    probabilities = tuple(
        value.as_fraction() for value in source.probability_upper_bounds
    )
    witnesses = tuple(value.as_fraction() for value in source.witness_parameters)
    complements = _require_result_digit_budget(
        probabilities, witnesses, source.neighborhoods
    )
    inequalities = _inequalities(source, probabilities, witnesses, complements)
    failures = tuple(
        row.event_index for row in inequalities if not row.inequality_holds
    )
    return AsymmetricLocalLemmaWitnessCheckResult(
        source=source,
        inequalities=inequalities,
        failed_event_indices=failures,
        valid=not failures,
        witness_product=CanonicalRational.from_fraction(_witness_product(complements)),
    )


__all__ = [
    "AsymmetricLocalLemmaInequality",
    "AsymmetricLocalLemmaWitness",
    "AsymmetricLocalLemmaWitnessCheckResult",
    "check_asymmetric_local_lemma_witness",
]
