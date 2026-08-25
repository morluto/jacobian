"""Canonical wire contract for exact asymmetric local-lemma witnesses."""

from __future__ import annotations

from collections.abc import Mapping
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
from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.probability.local_lemma import (
    MAX_LOCAL_LEMMA_EVENTS,
    MAX_LOCAL_LEMMA_INCIDENCES,
    MAX_LOCAL_LEMMA_INPUT_RATIONAL_DIGITS,
    MAX_LOCAL_LEMMA_LABEL_LENGTH,
    MAX_LOCAL_LEMMA_RESULT_RATIONAL_DIGITS,
    MAX_LOCAL_LEMMA_TOTAL_RESULT_DIGITS,
    AsymmetricLocalLemmaInequality,
    AsymmetricLocalLemmaWitness,
    check_asymmetric_local_lemma_witness,
)
from jacobian.math.probability.local_lemma import (
    AsymmetricLocalLemmaWitnessCheckResult as NativeWitnessCheckResult,
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


class AsymmetricLocalLemmaWitnessRequest(StrictModel):
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

        return _bound_raw_source(value)

    @model_validator(mode="after")
    def require_admitted_source(self) -> Self:
        try:
            self.as_native()
        except ValueError as exc:
            raise _validation_error(str(exc)) from exc
        return self

    def as_native(self) -> AsymmetricLocalLemmaWitness:
        return AsymmetricLocalLemmaWitness(
            event_labels=self.event_labels,
            probability_upper_bounds=tuple(
                value.as_fraction() for value in self.probability_upper_bounds
            ),
            witness_parameters=tuple(
                value.as_fraction() for value in self.witness_parameters
            ),
            neighborhoods=tuple(tuple(row) for row in self.neighborhoods),
        )

    @classmethod
    def from_native(
        cls,
        source: AsymmetricLocalLemmaWitness,
    ) -> AsymmetricLocalLemmaWitnessRequest:
        return cls(
            event_labels=source.event_labels,
            probability_upper_bounds=tuple(
                CanonicalRational.from_fraction(value)
                for value in source.probability_upper_bounds
            ),
            witness_parameters=tuple(
                CanonicalRational.from_fraction(value)
                for value in source.witness_parameters
            ),
            neighborhoods=source.neighborhoods,
        )


class AsymmetricLocalLemmaInequalityResult(StrictModel):
    """One exact row ``p_i <= x_i * product_(j in Gamma(i)) (1-x_j)``."""

    event_index: StrictInt = Field(ge=0, lt=MAX_LOCAL_LEMMA_EVENTS)
    neighborhood_product: CanonicalRational
    right_hand_side: CanonicalRational
    slack: CanonicalRational
    inequality_holds: StrictBool

    def as_native(self) -> AsymmetricLocalLemmaInequality:
        return AsymmetricLocalLemmaInequality(
            event_index=self.event_index,
            neighborhood_product=self.neighborhood_product.as_fraction(),
            right_hand_side=self.right_hand_side.as_fraction(),
            slack=self.slack.as_fraction(),
            inequality_holds=self.inequality_holds,
        )

    @classmethod
    def from_native(
        cls,
        row: AsymmetricLocalLemmaInequality,
    ) -> AsymmetricLocalLemmaInequalityResult:
        return cls(
            event_index=row.event_index,
            neighborhood_product=CanonicalRational.from_fraction(
                row.neighborhood_product
            ),
            right_hand_side=CanonicalRational.from_fraction(row.right_hand_side),
            slack=CanonicalRational.from_fraction(row.slack),
            inequality_holds=row.inequality_holds,
        )


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

    source: AsymmetricLocalLemmaWitnessRequest = Field(
        description=(
            "The complete canonical numerical witness whose exact inequalities "
            "were checked."
        )
    )
    inequalities: tuple[AsymmetricLocalLemmaInequalityResult, ...] = Field(
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
        """Bound forged result collections and rationals before replay."""

        return _bound_raw_result(value)

    @model_validator(mode="after")
    def bind_result_to_source(self) -> Self:
        try:
            self.as_native()
        except ValueError as exc:
            raise _validation_error(str(exc)) from exc
        return self

    def as_native(self) -> NativeWitnessCheckResult:
        return NativeWitnessCheckResult(
            source=self.source.as_native(),
            inequalities=tuple(row.as_native() for row in self.inequalities),
            failed_event_indices=tuple(self.failed_event_indices),
            valid=self.valid,
            witness_product=self.witness_product.as_fraction(),
        )

    @classmethod
    def from_native(
        cls,
        result: NativeWitnessCheckResult,
    ) -> AsymmetricLocalLemmaWitnessCheckResult:
        return cls(
            source=AsymmetricLocalLemmaWitnessRequest.from_native(result.source),
            inequalities=tuple(
                AsymmetricLocalLemmaInequalityResult.from_native(row)
                for row in result.inequalities
            ),
            failed_event_indices=result.failed_event_indices,
            valid=result.valid,
            witness_product=CanonicalRational.from_fraction(result.witness_product),
        )


def compute_asymmetric_local_lemma_witness_check(
    request: AsymmetricLocalLemmaWitnessRequest,
) -> AsymmetricLocalLemmaWitnessCheckResult:
    return AsymmetricLocalLemmaWitnessCheckResult.from_native(
        check_asymmetric_local_lemma_witness(request.as_native())
    )


ASYMMETRIC_LOCAL_LEMMA_OPERATION = MathTool(
    operation_id="probability.local_lemma.asymmetric_witness.check",
    title="Check an exact asymmetric local-lemma numerical witness",
    description=(
        "Check every exact inequality p_i <= x_i * product over directed "
        "Gamma(i) of (1-x_j), returning the complete source-bound product, "
        "right-hand-side, and slack ledger. Listed self-neighbors contribute "
        "once. This checks only the numerical witness; it does not establish "
        "that the declared neighborhoods are a dependency graph or that any "
        "event-independence hypothesis holds."
    ),
    request_type=AsymmetricLocalLemmaWitnessRequest,
    result_type=AsymmetricLocalLemmaWitnessCheckResult,
    run=compute_asymmetric_local_lemma_witness_check,
    tags=(
        "probability",
        "Lovasz-local-lemma",
        "asymmetric-local-lemma",
        "numerical-witness",
        "exact-rational",
        "directed-neighborhoods",
        "bounded",
    ),
    examples=(
        example(
            "two_event_directed_witness",
            "Check two exact directed asymmetric local-lemma inequalities; all arrays share the ordered event axis, neighborhoods are strictly increasing index sets, and each witness lies in [0,1).",
            {
                "event_labels": ["A", "B"],
                "probability_upper_bounds": [
                    {"num": "1", "den": "4"},
                    {"num": "1", "den": "2"},
                ],
                "witness_parameters": [
                    {"num": "1", "den": "2"},
                    {"num": "1", "den": "2"},
                ],
                "neighborhoods": [[1], []],
            },
        ),
    ),
)


__all__ = [
    "ASYMMETRIC_LOCAL_LEMMA_OPERATION",
    "AsymmetricLocalLemmaInequalityResult",
    "AsymmetricLocalLemmaWitnessCheckResult",
    "AsymmetricLocalLemmaWitnessRequest",
]
