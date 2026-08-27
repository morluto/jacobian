"""Typed bounded contracts for finite universal-algebra operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
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
# The result adds at most 32 source indices in kernel blocks, 32 image indices,
# six flags/scalars, or one obstruction with two arity-at-most-four tuples and a
# 64-byte operation ID.  Four KiB conservatively bounds that JSON wrapper over
# the exactly measured retained carrier-map bytes.
_HOMOMORPHISM_RESULT_RESERVE_BYTES = 4_096
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

    @model_validator(mode="after")
    def require_total_assignment(self) -> Self:
        try:
            require_term_for_algebra(self.term, self.algebra)
        except UniversalAlgebraAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        if len(self.assignment) != self.term.variable_count:
            raise _validation_error(
                "assignment_length_mismatch",
                "assignment must cover exactly the referenced variables",
            )
        size = len(self.algebra.carrier)
        if any(not 0 <= value < size for value in self.assignment):
            raise _validation_error(
                "assignment_value_out_of_range", "assignment value out of carrier range"
            )
        return self


class EvaluateResult(StrictModel):
    value: int = Field(ge=0)


class EquationProfileRequest(StrictModel):
    algebra: FiniteAlgebra
    left: FlatTerm
    right: FlatTerm
    variable_count: int = Field(ge=1, le=8, strict=True)

    @model_validator(mode="after")
    def require_bounded_complete_profile(self) -> Self:
        try:
            require_term_for_algebra(self.left, self.algebra)
        except UniversalAlgebraAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        try:
            require_term_for_algebra(self.right, self.algebra)
        except UniversalAlgebraAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        if (
            max(self.left.variable_count, self.right.variable_count)
            > self.variable_count
        ):
            raise _validation_error(
                "variable_count_too_small",
                "variable_count must cover every referenced variable",
            )
        work = len(self.algebra.carrier) ** self.variable_count
        if work > MAX_ENUMERATION_WORK:
            raise _validation_error(
                "equation_profile_work_exceeded",
                "equation profile exceeds the assignment work budget",
            )
        return self


class EquationCounterexample(StrictModel):
    assignment: tuple[int, ...] = Field(max_length=8)
    left_value: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)
    right_value: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)


class EquationProfileResult(StrictModel):
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


class SubalgebraRequest(StrictModel):
    algebra: FiniteAlgebra
    generators: tuple[int, ...] = Field(
        default=(),
        max_length=MAX_CARRIER_SIZE,
    )

    @model_validator(mode="after")
    def require_valid_bounded_generators(self) -> Self:
        size = len(self.algebra.carrier)
        if any(not 0 <= generator < size for generator in self.generators):
            raise _validation_error(
                "generator_out_of_range", "generator out of carrier range"
            )
        work = sum(size**symbol.arity for symbol in self.algebra.operations) * size
        if work > MAX_ENUMERATION_WORK:
            raise _validation_error(
                "subalgebra_work_exceeded",
                "subalgebra closure exceeds the operation work budget",
            )
        return self


class SubalgebraResult(StrictModel):
    generated_carrier: tuple[int, ...]
    rounds: int = Field(ge=1)
    is_closed: bool


def _require_homomorphism_output_headroom(
    carrier_map: FiniteAlgebraCarrierMap,
) -> None:
    try:
        retained_source_bytes = len(
            encode_strict_json(carrier_map.model_dump(mode="json"))
        )
    except CanonicalizationError as exc:
        raise _validation_error(
            "carrier_map_output_exceeded",
            "finite algebra carrier map exceeds the canonical output limit",
        ) from exc
    output_limit = CanonicalLimits().max_output_bytes
    if retained_source_bytes + _HOMOMORPHISM_RESULT_RESERVE_BYTES > output_limit:
        raise _validation_error(
            "homomorphism_work_exceeded",
            "homomorphism profile retains the carrier map and would exceed the "
            f"{output_limit}-byte canonical output limit",
        )


class HomomorphismProfileRequest(StrictModel):
    """Check one total finite-algebra carrier map for operation preservation."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Check a complete carrier map between finite algebras with exactly "
                "matching ordered operation identifiers and arities. Every source "
                "operation-table cell is checked; the retained-map result is "
                "rejected before execution when that work or canonical output "
                "would exceed its bound."
            )
        }
    )

    carrier_map: FiniteAlgebraCarrierMap = Field(
        description=(
            "Total source-to-target carrier map. Source and target signatures must "
            "match exactly; carrier sizes may differ."
        )
    )

    @model_validator(mode="after")
    def require_bounded_complete_scan(self) -> Self:
        preservation_cells = sum(len(table) for table in self.carrier_map.source.tables)
        if preservation_cells > MAX_ENUMERATION_WORK:
            raise _validation_error(
                "homomorphism_work_exceeded",
                "homomorphism operation work exceeds the enumeration budget",
            )
        _require_homomorphism_output_headroom(self.carrier_map)
        return self


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
    method: Literal["DENSE_TABLE_SCAN"] = "DENSE_TABLE_SCAN"

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

    @classmethod
    def _from_kernel(cls, payload: dict[str, object]) -> Self:
        """Build a profile after the admitted preservation scan established it."""

        data: dict[str, Any] = payload.copy()
        obstruction = data.get("obstruction")
        if isinstance(obstruction, dict):
            data["obstruction"] = HomomorphismObstruction.model_construct(**obstruction)
        return cls.model_construct(**data)


class _PartitionRequest(StrictModel):
    algebra: FiniteAlgebra
    partition: tuple[CarrierBlock, ...] = Field(
        min_length=1,
        max_length=MAX_CARRIER_SIZE,
    )

    @model_validator(mode="after")
    def require_complete_bounded_partition(self) -> Self:
        _require_partition(self.algebra, self.partition)
        if _congruence_work(self.algebra) > MAX_ENUMERATION_WORK:
            raise _validation_error(
                "congruence_work_exceeded",
                "congruence check exceeds the operation work budget",
            )
        return self


class CongruenceRequest(_PartitionRequest):
    """Check one complete carrier partition for operation compatibility."""


class CongruenceResult(StrictModel):
    is_congruence: bool
    obstruction: str | None = None


class QuotientRequest(_PartitionRequest):
    """Construct ``A/theta`` for an admitted congruence partition."""

    @model_validator(mode="after")
    def require_source_bound_result_headroom(self) -> Self:
        """Bound the retained source, quotient tables, and carrier map."""

        quotient_size = len(self.partition)
        quotient_table_cells = sum(
            quotient_size**operation.arity for operation in self.algebra.operations
        )
        quotient_work = (
            _congruence_work(self.algebra)
            + sum(len(table) for table in self.algebra.tables)
            + quotient_table_cells
        )
        if quotient_work > MAX_ENUMERATION_WORK:
            raise _validation_error(
                "quotient_work_exceeded",
                "quotient construction exceeds the operation work budget",
            )
        try:
            source_bytes = len(encode_strict_json(self.algebra.model_dump(mode="json")))
            operation_bytes = len(
                encode_strict_json(
                    [
                        operation.model_dump(mode="json")
                        for operation in self.algebra.operations
                    ]
                )
            )
            quotient_carrier_bytes = sum(
                len(encode_strict_json(f"B{index}")) + 1
                for index in range(quotient_size)
            )
        except CanonicalizationError as exc:
            raise _validation_error(
                "quotient_source_output_exceeded",
                "quotient source exceeds the canonical output limit",
            ) from exc
        quotient_index_bytes = len(str(quotient_size - 1)) + 1
        quotient_table_bytes = quotient_table_cells * quotient_index_bytes
        mapping_bytes = len(self.algebra.carrier) * quotient_index_bytes
        predicted_bytes = (
            source_bytes
            + operation_bytes
            + quotient_carrier_bytes
            + quotient_table_bytes
            + mapping_bytes
            + _HOMOMORPHISM_RESULT_RESERVE_BYTES
        )
        output_limit = CanonicalLimits().max_output_bytes
        if predicted_bytes > output_limit:
            raise _validation_error(
                "quotient_output_exceeded",
                "canonical quotient homomorphism would exceed the "
                f"{output_limit}-byte output limit",
            )
        return self


__all__ = [
    "CongruenceRequest",
    "CongruenceResult",
    "EquationCounterexample",
    "EquationProfileRequest",
    "EquationProfileResult",
    "EvaluateRequest",
    "EvaluateResult",
    "HomomorphismObstruction",
    "HomomorphismProfileRequest",
    "HomomorphismProfileResult",
    "QuotientRequest",
    "SubalgebraRequest",
    "SubalgebraResult",
]
