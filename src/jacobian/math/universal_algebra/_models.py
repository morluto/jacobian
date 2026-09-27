"""Typed bounded contracts for finite universal-algebra operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.universal_algebra.values import (
    MAX_ARITY,
    MAX_CARRIER_SIZE,
    MAX_SIGNATURE_SIZE,
    FiniteAlgebra,
    FiniteAlgebraCarrierMap,
    FiniteAlgebraHomomorphism,
    FlatTerm,
    UniversalAlgebraAdmissionError,
    require_term_for_algebra,
)

MAX_ENUMERATION_WORK = 1_000_000
CarrierBlock = Annotated[
    tuple[int, ...],
    Field(min_length=1, max_length=MAX_CARRIER_SIZE),
]


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"universal_algebra.{code}", message)


def _require_partition(
    algebra: FiniteAlgebra,
    partition: tuple[CarrierBlock, ...],
) -> None:
    expected = set(range(len(algebra.carrier)))
    seen: set[int] = set()
    for block in partition:
        for element in block:
            if element not in expected:
                raise _validation_error(
                    "partition_element_out_of_range",
                    "partition element out of carrier range",
                )
            if element in seen:
                raise _validation_error(
                    "partition_blocks_overlap", "partition blocks must be disjoint"
                )
            seen.add(element)
    if seen != expected:
        raise _validation_error(
            "partition_incomplete", "partition blocks must exactly cover the carrier"
        )


def _congruence_work(algebra: FiniteAlgebra) -> int:
    size = len(algebra.carrier)
    return sum(
        size**symbol.arity * max(1, symbol.arity) * size
        for symbol in algebra.operations
    )


class EvaluateRequest(StrictModel):
    algebra: FiniteAlgebra
    term: FlatTerm
    assignment: tuple[int, ...] = Field(default=(), max_length=256)


class EvaluateResult(StrictModel):
    algebra: FiniteAlgebra
    term: FlatTerm
    assignment: tuple[int, ...] = Field(max_length=256)
    value: int = Field(ge=0)


class EquationProfileRequest(StrictModel):
    algebra: FiniteAlgebra
    left: FlatTerm
    right: FlatTerm
    variable_count: int = Field(ge=1, le=8, strict=True)


class EquationCounterexample(StrictModel):
    assignment: tuple[int, ...] = Field(max_length=8)
    left_value: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)
    right_value: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)


class EquationProfileResult(StrictModel):
    algebra: FiniteAlgebra
    left: FlatTerm
    right: FlatTerm
    variable_count: int = Field(ge=1, le=8, strict=True)
    status: Literal["HOLDS", "FAILS"]
    satisfying_count: int = Field(ge=0, le=MAX_ENUMERATION_WORK)
    first_counterassignment: EquationCounterexample | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if (self.status == "FAILS") != (self.first_counterassignment is not None):
            raise _validation_error(
                "counterexample_status_mismatch",
                "FAILS must carry exactly one first counterassignment",
            )
        return self


def _require_countermodel_profile_intrinsics(
    profile: EquationProfileResult,
    carrier_size: int,
    profile_index: int,
) -> None:
    assignment_count = carrier_size**profile.variable_count
    if profile.satisfying_count > assignment_count:
        raise _validation_error(
            "countermodel_result_satisfying_count",
            f"profile {profile_index} satisfying_count cannot exceed its assignment space",
        )
    if profile.status == "HOLDS" and profile.satisfying_count != assignment_count:
        raise _validation_error(
            "countermodel_result_holds_count",
            f"profile {profile_index} HOLDS must cover its complete assignment space",
        )
    if profile.status == "FAILS" and profile.satisfying_count >= assignment_count:
        raise _validation_error(
            "countermodel_result_fails_count",
            f"profile {profile_index} FAILS must leave an assignment unsatisfied",
        )
    counterexample = profile.first_counterassignment
    if (
        counterexample is not None
        and counterexample.left_value == counterexample.right_value
    ):
        raise _validation_error(
            "countermodel_result_equal_values",
            f"profile {profile_index} counterexample values must differ",
        )


class MagmaEquation(StrictModel):
    """One equation over a source-bound binary magma term language."""

    left: FlatTerm
    right: FlatTerm


class ImplicationCountermodelCheckRequest(StrictModel):
    """Check an explicit finite magma against premises and one target."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "The algebra must have exactly one binary operation. Terms use "
                "the algebra's operation index and variable axis; every assignment "
                "to the variables occurring in an equation is exhausted."
            )
        }
    )

    algebra: FiniteAlgebra = Field(
        description="A finite algebra with exactly one binary operation (a magma)."
    )
    premises: tuple[MagmaEquation, ...] = Field(
        max_length=16,
        description="Premise equations checked universally over this magma.",
    )
    target: MagmaEquation = Field(
        description="The target equation whose failure is sought after premise checks."
    )


class ImplicationCountermodelCheckResult(StrictModel):
    """Complete premise and target profiles for one explicit finite magma."""

    algebra: FiniteAlgebra
    premises: tuple[EquationProfileResult, ...] = Field(max_length=16)
    target: EquationProfileResult
    is_countermodel: bool

    @model_validator(mode="after")
    def bind_countermodel_status(self) -> Self:
        expected = all(profile.status == "HOLDS" for profile in self.premises) and (
            self.target.status == "FAILS"
        )
        if self.is_countermodel != expected:
            raise _validation_error(
                "countermodel_status_mismatch",
                "is_countermodel must mean all premises hold and the target fails",
            )
        if any(profile.algebra != self.algebra for profile in self.premises) or (
            self.target.algebra != self.algebra
        ):
            raise _validation_error(
                "countermodel_source_mismatch",
                "all equation profiles must retain the checked magma",
            )
        if len(self.algebra.operations) != 1 or self.algebra.operations[0].arity != 2:
            raise _validation_error(
                "countermodel_magma_signature",
                "a finite-magma countermodel result must retain exactly one binary operation",
            )
        for profile_index, profile in enumerate((*self.premises, self.target)):
            for term_name, term in (("left", profile.left), ("right", profile.right)):
                try:
                    require_term_for_algebra(term, self.algebra)
                except UniversalAlgebraAdmissionError as exc:
                    raise _validation_error(
                        "countermodel_result_term_signature",
                        f"profile {profile_index} {term_name} term is not bound to the retained magma: {exc}",
                    ) from exc
            declared_variable_count = max(
                profile.left.variable_count, profile.right.variable_count
            )
            if profile.variable_count != declared_variable_count:
                raise _validation_error(
                    "countermodel_result_variable_count",
                    f"profile {profile_index} variable_count must match the declared variable axis",
                )
            _require_countermodel_profile_intrinsics(
                profile, len(self.algebra.carrier), profile_index
            )
            counterexample = profile.first_counterassignment
            if counterexample is not None:
                if len(counterexample.assignment) != profile.variable_count:
                    raise _validation_error(
                        "countermodel_result_assignment_axis",
                        f"profile {profile_index} counterassignment must cover its declared variable axis",
                    )
                if any(
                    value < 0 or value >= len(self.algebra.carrier)
                    for value in counterexample.assignment
                ):
                    raise _validation_error(
                        "countermodel_result_assignment_range",
                        f"profile {profile_index} counterassignment value is outside the retained magma carrier",
                    )
                if any(
                    value < 0 or value >= len(self.algebra.carrier)
                    for value in (
                        counterexample.left_value,
                        counterexample.right_value,
                    )
                ):
                    raise _validation_error(
                        "countermodel_result_value_range",
                        f"profile {profile_index} counterexample value is outside the retained magma carrier",
                    )
        return self


class SubalgebraRequest(StrictModel):
    algebra: FiniteAlgebra
    generators: tuple[int, ...] = Field(
        default=(),
        max_length=MAX_CARRIER_SIZE,
    )


class SubalgebraResult(StrictModel):
    algebra: FiniteAlgebra
    generators: tuple[int, ...]
    generated_carrier: tuple[int, ...]
    rounds: int = Field(ge=1)
    is_closed: bool


class CongruenceObstruction(StrictModel):
    """Typed operation witness for a failed congruence compatibility check."""

    kind: Literal["compatibility_violation"] = "compatibility_violation"
    operation: int = Field(ge=0, le=MAX_SIGNATURE_SIZE - 1)
    left_arguments: tuple[int, ...] = Field(max_length=MAX_ARITY)
    right_arguments: tuple[int, ...] = Field(max_length=MAX_ARITY)
    left_output: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)
    right_output: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)


class HomomorphismProfileRequest(StrictModel):
    """Check one total finite-algebra carrier map for operation preservation."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Check a complete carrier map between finite algebras with exactly "
                "matching ordered operation identifiers and arities. Every source "
                "operation-table cell is checked; the retained-map result is "
                "admitted by the complete table-cell work bound."
            )
        }
    )

    carrier_map: FiniteAlgebraCarrierMap = Field(
        description=(
            "Total source-to-target carrier map. Source and target signatures must "
            "match exactly; carrier sizes may differ."
        )
    )


class HomomorphismObstruction(StrictModel):
    """The first exact operation-preservation failure in canonical scan order."""

    operation: int = Field(ge=0, le=MAX_SIGNATURE_SIZE - 1)
    operation_id: str = Field(min_length=1, max_length=64)
    source_arguments: tuple[int, ...] = Field(max_length=MAX_ARITY)
    target_arguments: tuple[int, ...] = Field(max_length=MAX_ARITY)
    source_output: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)
    mapped_source_output: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)
    target_output: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)


class HomomorphismProfileResult(StrictModel):
    """A checked homomorphism or the first exact preservation obstruction.

    The positive branch carries a reusable :class:`FiniteAlgebraHomomorphism`;
    the negative branch retains the supplied carrier map. Parsing checks only
    branch shape and canonical container structure; the admitted producer
    establishes preservation, fibers, image, and the first obstruction.
    """

    status: Literal["HOMOMORPHISM", "NOT_A_HOMOMORPHISM"]
    homomorphism: FiniteAlgebraHomomorphism | None = None
    carrier_map: FiniteAlgebraCarrierMap | None = None
    kernel_partition: tuple[CarrierBlock, ...] = Field(
        default=(), max_length=MAX_CARRIER_SIZE
    )
    image: tuple[int, ...] = Field(default=(), max_length=MAX_CARRIER_SIZE)
    injective: bool | None = None
    surjective: bool | None = None
    isomorphism: bool | None = None
    obstruction: HomomorphismObstruction | None = None

    @model_validator(mode="after")
    def require_structural_profile(self) -> Self:
        if self.status == "HOMOMORPHISM":
            if self.homomorphism is None:
                raise _validation_error(
                    "positive_missing_homomorphism",
                    "HOMOMORPHISM must carry a checked homomorphism",
                )
            if self.carrier_map is not None or self.obstruction is not None:
                raise _validation_error(
                    "positive_failed_data",
                    "HOMOMORPHISM cannot carry a failed map or obstruction",
                )
            if None in (self.injective, self.surjective, self.isomorphism):
                raise _validation_error(
                    "positive_flags_missing",
                    "HOMOMORPHISM must carry all map-property flags",
                )
            _require_partition(self.homomorphism.source, self.kernel_partition)
            if (
                tuple(sorted(self.image)) != self.image
                or len(set(self.image)) != len(self.image)
                or any(
                    value < 0 or value >= len(self.homomorphism.target.carrier)
                    for value in self.image
                )
            ):
                raise _validation_error(
                    "image_not_canonical",
                    "image must be a sorted unique target-carrier sequence",
                )
            if self.isomorphism is not (self.injective and self.surjective):
                raise _validation_error(
                    "isomorphism_flags_mismatch",
                    "isomorphism must agree with injective and surjective",
                )
            return self

        if self.carrier_map is None or self.obstruction is None:
            raise _validation_error(
                "negative_data_missing",
                "NOT_A_HOMOMORPHISM must retain the carrier map and obstruction",
            )
        if self.homomorphism is not None:
            raise _validation_error(
                "negative_has_homomorphism",
                "NOT_A_HOMOMORPHISM cannot carry a homomorphism",
            )
        if (
            self.kernel_partition
            or self.image
            or any(
                flag is not None
                for flag in (self.injective, self.surjective, self.isomorphism)
            )
        ):
            raise _validation_error(
                "negative_positive_data",
                "NOT_A_HOMOMORPHISM cannot carry positive map-property data",
            )
        return self


class _PartitionRequest(StrictModel):
    algebra: FiniteAlgebra
    partition: tuple[CarrierBlock, ...] = Field(
        min_length=1,
        max_length=MAX_CARRIER_SIZE,
    )

    @model_validator(mode="after")
    def require_complete_partition(self) -> Self:
        _require_partition(self.algebra, self.partition)
        return self


class CongruenceRequest(_PartitionRequest):
    """Check one complete carrier partition for operation compatibility."""


class CongruenceResult(StrictModel):
    algebra: FiniteAlgebra
    partition: tuple[CarrierBlock, ...]
    is_congruence: bool
    obstruction: CongruenceObstruction | None = None

    @model_validator(mode="after")
    def bind_obstruction(self) -> Self:
        _require_partition(self.algebra, self.partition)
        if self.is_congruence:
            if self.obstruction is not None:
                raise _validation_error(
                    "congruence_has_obstruction",
                    "a congruence result cannot carry obstruction data",
                )
            return self
        if self.obstruction is None:
            raise _validation_error(
                "noncongruence_missing_obstruction",
                "a noncongruence result must identify its obstruction",
            )
        return self


class QuotientRequest(_PartitionRequest):
    """Construct ``A/theta`` for an admitted congruence partition."""


__all__ = [
    "MAX_COUNTERMODEL_ORDER",
    "MAX_COUNTERMODEL_TABLES",
    "CongruenceObstruction",
    "CongruenceRequest",
    "CongruenceResult",
    "CountermodelFindRequest",
    "CountermodelFindResult",
    "CountermodelFindStatus",
    "CountermodelFindStopReason",
    "EquationCounterexample",
    "EquationProfileRequest",
    "EquationProfileResult",
    "EvaluateRequest",
    "EvaluateResult",
    "HomomorphismObstruction",
    "HomomorphismProfileRequest",
    "HomomorphismProfileResult",
    "ImplicationCountermodelCheckRequest",
    "ImplicationCountermodelCheckResult",
    "MagmaEquation",
    "QuotientRequest",
    "SubalgebraRequest",
    "SubalgebraResult",
]


MAX_COUNTERMODEL_ORDER = 4
MAX_COUNTERMODEL_TABLES = 500_000
MAX_COUNTERMODEL_PREMISES = 16
MAX_COUNTERMODEL_SEARCH_WORK = 50_000_000

CountermodelFindStatus = Literal["FOUND", "EXHAUSTED_UP_TO_BOUND", "UNKNOWN"]

CountermodelFindStopReason = Literal["TABLE_BUDGET_EXHAUSTED"]


class CountermodelFindRequest(StrictModel):
    """Find a finite-magma countermodel by bounded table search.

    Carrier orders from ``min_order`` through ``max_order`` are enumerated
    in increasing order; within one order, tables enumerate row-major with
    cell values ascending, so the all-zero table comes first.  At most
    ``table_budget`` tables are checked before reporting UNKNOWN.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Bounded countermodel search over finite magma tables. "
                "Orders enumerate increasingly; tables enumerate row-major "
                "with ascending cell values. A negative conclusion follows "
                "only from completed search within the declared orders. "
                "Admission also limits table_budget times the complete "
                "assignment-evaluation work (sum over distinct premises and "
                "target of order^variables * term-node-count at each searched "
                "order) to 50,000,000; requests beyond this aggregate envelope "
                "are rejected."
            )
        }
    )

    premises: tuple[MagmaEquation, ...] = Field(
        default=(),
        max_length=MAX_COUNTERMODEL_PREMISES,
        description="Premise equations; all must hold in a countermodel.",
    )
    target: MagmaEquation = Field(
        description="The target equation a countermodel refutes.",
    )
    min_order: int = Field(
        ge=1,
        le=MAX_COUNTERMODEL_ORDER,
        description="Smallest carrier order searched.",
    )
    max_order: int = Field(
        ge=1,
        le=MAX_COUNTERMODEL_ORDER,
        description="Largest carrier order searched.",
    )
    table_budget: int = Field(
        ge=1,
        le=MAX_COUNTERMODEL_TABLES,
        description="Check at most this many tables before reporting UNKNOWN.",
    )

    @model_validator(mode="after")
    def require_order_range(self) -> Self:
        if self.min_order > self.max_order:
            raise _validation_error(
                "countermodel_find_order_range",
                "min_order must not exceed max_order",
            )
        return self


class CountermodelFindResult(StrictModel):
    """A bounded countermodel-search outcome with its exhaustion receipt.

    - ``FOUND``: ``order`` and ``certificate`` carry the first enumerated
      countermodel; ``orders_complete`` holds every smaller searched order
      fully examined, so the witness is minimal within the declared range.
    - ``EXHAUSTED_UP_TO_BOUND``: every rotation-free table of every
      declared order was examined; ``tables_examined == total_tables``.
      This is a finite bounded conclusion only, never a proof that the
      implication holds in all magmas.
    - ``UNKNOWN``: ``stop_reason`` records the spent table budget;
      ``current_order`` is the order under search when it ran out.

    ``total_tables`` is the exact admitted table count: ``n^(n^2)`` per
    order.
    """

    premises: tuple[MagmaEquation, ...] = Field(
        default=(), max_length=MAX_COUNTERMODEL_PREMISES
    )
    target: MagmaEquation
    min_order: int = Field(ge=1, le=MAX_COUNTERMODEL_ORDER)
    max_order: int = Field(ge=1, le=MAX_COUNTERMODEL_ORDER)
    table_budget: int = Field(ge=1, le=MAX_COUNTERMODEL_TABLES)
    status: CountermodelFindStatus
    orders_complete: tuple[int, ...] = ()
    order: int | None = None
    certificate: ImplicationCountermodelCheckResult | None = None
    tables_examined: int = Field(default=0, ge=0)
    total_tables: int = Field(default=0, ge=0)
    current_order: int | None = None
    stop_reason: CountermodelFindStopReason | None = None

    @model_validator(mode="after")
    def require_find_payload(self) -> Self:
        if self.status == "UNKNOWN":
            if self.stop_reason is None:
                raise _validation_error(
                    "countermodel_find_reason_payload",
                    "an unknown search carries its stop reason",
                )
            if self.order is not None or self.certificate is not None:
                raise _validation_error(
                    "countermodel_find_unknown_witness",
                    "an unknown search carries no order or certificate",
                )
            if self.current_order is None:
                raise _validation_error(
                    "countermodel_find_current_order",
                    "an unknown search carries its current order",
                )
            return self
        if self.stop_reason is not None or self.current_order is not None:
            raise _validation_error(
                "countermodel_find_decided_payload",
                "a decided search carries no stop reason or current order",
            )
        if self.status == "FOUND":
            if self.order is None or self.certificate is None:
                raise _validation_error(
                    "countermodel_find_found_payload",
                    "a found search carries its order and certificate",
                )
        elif self.order is not None or self.certificate is not None:
            raise _validation_error(
                "countermodel_find_exhausted_payload",
                "an exhausted search carries no order or certificate",
            )
        return self

    @model_validator(mode="after")
    def require_find_certificate(self) -> Self:
        if self.status != "FOUND":
            return self
        certificate = self.certificate
        if certificate is None:
            raise _validation_error(
                "countermodel_find_found_payload",
                "a found search carries its order and certificate",
            )
        if not certificate.is_countermodel:
            raise _validation_error(
                "countermodel_find_certificate_status",
                "the certificate must establish a countermodel",
            )
        if len(certificate.algebra.carrier) != self.order:
            raise _validation_error(
                "countermodel_find_certificate_order",
                "the certificate carrier must match the found order",
            )
        if not self.min_order <= self.order <= self.max_order:
            raise _validation_error(
                "countermodel_find_certificate_order_range",
                "the found order must lie in the declared search range",
            )
        unique_premises: list[MagmaEquation] = []
        for premise in self.premises:
            if premise not in unique_premises:
                unique_premises.append(premise)
        profiles = certificate.premises
        if len(profiles) != len(unique_premises) or any(
            (profile.left, profile.right) != (premise.left, premise.right)
            for profile, premise in zip(profiles, unique_premises, strict=True)
        ):
            raise _validation_error(
                "countermodel_find_certificate_premises",
                "the found certificate must check the declared premise equations",
            )
        if (certificate.target.left, certificate.target.right) != (
            self.target.left,
            self.target.right,
        ):
            raise _validation_error(
                "countermodel_find_certificate_target",
                "the found certificate must check the declared target equation",
            )
        return self

    @model_validator(mode="after")
    def require_find_receipt(self) -> Self:
        _require_find_receipt_common(self)
        if self.status == "FOUND":
            _require_found_receipt(self)
        elif self.status == "EXHAUSTED_UP_TO_BOUND":
            _require_exhaustion_receipt(self)
        else:
            _require_unknown_receipt(self)
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _require_find_receipt_common(result: CountermodelFindResult) -> None:
    if result.min_order > result.max_order:
        raise _validation_error(
            "countermodel_find_order_range",
            "min_order must not exceed max_order",
        )
    expected_total = sum(
        order ** (order * order)
        for order in range(result.min_order, result.max_order + 1)
    )
    if expected_total < 1:
        raise _validation_error(
            "countermodel_find_total_tables",
            "the declared search range contains at least one table",
        )
    if result.total_tables != expected_total:
        raise _validation_error(
            "countermodel_find_total_tables_scope",
            "total_tables must equal the exact table count in the declared range",
        )
    if not 0 <= result.tables_examined <= min(result.total_tables, result.table_budget):
        raise _validation_error(
            "countermodel_find_examined_bounds",
            "examined tables must lie within the total and declared budget",
        )
    if list(result.orders_complete) != sorted(result.orders_complete) or len(
        set(result.orders_complete)
    ) != len(result.orders_complete):
        raise _validation_error(
            "countermodel_find_orders_complete",
            "completed orders are distinct and increasing",
        )


def _require_found_receipt(result: CountermodelFindResult) -> None:
    if result.order is None or not result.min_order <= result.order <= result.max_order:
        raise _validation_error(
            "countermodel_find_found_order",
            "a found order lies within the declared search range",
        )
    if result.orders_complete != tuple(range(result.min_order, result.order)):
        raise _validation_error(
            "countermodel_find_minimality_receipt",
            "every smaller searched order was completely examined",
        )
    before_order = sum(
        order ** (order * order) for order in range(result.min_order, result.order)
    )
    through_order = before_order + result.order ** (result.order * result.order)
    if not before_order < result.tables_examined <= through_order:
        raise _validation_error(
            "countermodel_find_found_examined_scope",
            "the examined count must reach the found order without passing it",
        )


def _require_exhaustion_receipt(result: CountermodelFindResult) -> None:
    if result.orders_complete != tuple(range(result.min_order, result.max_order + 1)):
        raise _validation_error(
            "countermodel_find_exhaustion_orders",
            "an exhausted search completed every declared order",
        )
    if result.tables_examined != result.total_tables:
        raise _validation_error(
            "countermodel_find_exhaustion_receipt",
            "an exhausted search examined every table",
        )
    if result.total_tables > result.table_budget:
        raise _validation_error(
            "countermodel_find_exhaustion_budget",
            "an exhausted search fit its table budget",
        )


def _require_unknown_receipt(result: CountermodelFindResult) -> None:
    if result.tables_examined != result.table_budget:
        raise _validation_error(
            "countermodel_find_budget_receipt",
            "a budget-exhausted search spent its full budget",
        )
    if result.total_tables <= result.table_budget:
        raise _validation_error(
            "countermodel_find_budget_scope",
            "a budget-exhausted search left tables unexamined",
        )
    if result.current_order is None or not (
        result.min_order <= result.current_order <= result.max_order
    ):
        raise _validation_error(
            "countermodel_find_current_order_range",
            "the current order lies within the declared search range",
        )
    if result.orders_complete != tuple(range(result.min_order, result.current_order)):
        raise _validation_error(
            "countermodel_find_unknown_orders",
            "unknown search receipts list exactly the completed prefix",
        )
    prior_tables = sum(
        order ** (order * order)
        for order in range(result.min_order, result.current_order)
    )
    current_tables = result.current_order ** (
        result.current_order * result.current_order
    )
    if not prior_tables <= result.tables_examined < prior_tables + current_tables:
        raise _validation_error(
            "countermodel_find_unknown_examined_scope",
            "the examined count locates the search inside current_order",
        )
