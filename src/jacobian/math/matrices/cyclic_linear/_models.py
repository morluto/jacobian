"""Typed contracts for exact rational cyclic linear systems."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    RationalVectorSpaceBasis,
    SimpleNumberFieldMatrix,
    SimpleNumberFieldVectorSpaceBasis,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_CYCLIC_GLOBAL_AXIS = MAX_RATIONAL_MATRIX_ORDER
MAX_CYCLIC_PERIOD = MAX_CYCLIC_GLOBAL_AXIS
MAX_CYCLIC_SYMBOL_ENTRIES = MAX_CYCLIC_GLOBAL_AXIS**2
MAX_CYCLIC_INPUT_DIGITS = 64
MAX_CYCLIC_COMPONENTS = MAX_CYCLIC_PERIOD
MAX_CYCLIC_FIELD_WORK = 100_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.cyclic.{reason}", message)


def _euler_phi(value: int) -> int:
    return sum(gcd(candidate, value) == 1 for candidate in range(1, value + 1))


class CyclicRationalBlockSymbolEntry(StrictModel):
    """One nonzero coefficient of a rational block Laurent symbol.

    The entry contributes ``coefficient * x^shift`` from one source block
    coordinate to one target block coordinate in ``QQ[x]/(x^period - 1)``.
    Shifts are canonical residues selected by the enclosing symbol.
    """

    target_coordinate: int = Field(ge=0, lt=MAX_CYCLIC_GLOBAL_AXIS)
    source_coordinate: int = Field(ge=0, lt=MAX_CYCLIC_GLOBAL_AXIS)
    shift: int = Field(ge=0, lt=MAX_CYCLIC_PERIOD)
    coefficient: CanonicalRational


class CyclicRationalBlockSymbol(StrictModel):
    """A compact exact shift-equivariant map over ``QQ[C_period]``.

    If source blocks are polynomials ``v_j(x)`` modulo ``x^period-1``, the
    target coordinate ``i`` is ``sum_j A_ij(x) v_j(x)``.  Global rational
    coordinates are ordered first by the coefficient of ``x^shift`` and then
    by block coordinate.
    """

    domain: Literal["QQ"] = "QQ"
    action: Literal["LEFT_MULTIPLICATION_IN_QQ_X_MOD_X_POWER_N_MINUS_1"] = (
        "LEFT_MULTIPLICATION_IN_QQ_X_MOD_X_POWER_N_MINUS_1"
    )
    period: int = Field(ge=1, le=MAX_CYCLIC_PERIOD)
    target_block_dimension: int = Field(ge=1, le=MAX_CYCLIC_GLOBAL_AXIS)
    source_block_dimension: int = Field(ge=1, le=MAX_CYCLIC_GLOBAL_AXIS)
    entries: tuple[CyclicRationalBlockSymbolEntry, ...] = Field(
        default=(), max_length=MAX_CYCLIC_SYMBOL_ENTRIES
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_envelope(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        entries = data.get("entries")
        if isinstance(entries, (list, tuple)):
            if len(entries) > MAX_CYCLIC_SYMBOL_ENTRIES:
                raise _validation_error(
                    "support_bound",
                    f"cyclic symbols contain at most {MAX_CYCLIC_SYMBOL_ENTRIES:,} entries",
                )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                coefficient = entry.get("coefficient")
                if not isinstance(coefficient, Mapping):
                    continue
                for part in ("num", "den"):
                    raw = coefficient.get(part)
                    if (
                        isinstance(raw, (str, int))
                        and len(str(raw).lstrip("-")) > MAX_CYCLIC_INPUT_DIGITS
                    ):
                        raise _validation_error(
                            "coefficient_bound",
                            "cyclic-symbol rationals are limited to "
                            f"{MAX_CYCLIC_INPUT_DIGITS} decimal digits",
                        )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_symbol(self) -> Self:
        if self.period * self.source_block_dimension > MAX_CYCLIC_GLOBAL_AXIS:
            raise _validation_error(
                "source_axis_bound",
                "period times source block dimension exceeds the 128-coordinate envelope",
            )
        if self.period * self.target_block_dimension > MAX_CYCLIC_GLOBAL_AXIS:
            raise _validation_error(
                "target_axis_bound",
                "period times target block dimension exceeds the 128-coordinate envelope",
            )
        coordinates = tuple(
            (entry.target_coordinate, entry.source_coordinate, entry.shift)
            for entry in self.entries
        )
        if coordinates != tuple(sorted(set(coordinates))):
            raise _validation_error(
                "support_order",
                "cyclic-symbol entries must use unique canonical row-major coordinates",
            )
        for entry in self.entries:
            if entry.target_coordinate >= self.target_block_dimension:
                raise _validation_error(
                    "target_coordinate",
                    "cyclic-symbol target coordinate exceeds its block dimension",
                )
            if entry.source_coordinate >= self.source_block_dimension:
                raise _validation_error(
                    "source_coordinate",
                    "cyclic-symbol source coordinate exceeds its block dimension",
                )
            if entry.shift >= self.period:
                raise _validation_error(
                    "shift_residue",
                    "cyclic-symbol shifts must be canonical residues modulo the period",
                )
            if entry.coefficient.as_fraction() == 0:
                raise _validation_error(
                    "zero_coefficient", "zero cyclic-symbol entries must be omitted"
                )
            if (
                max(
                    len(entry.coefficient.num.lstrip("-")),
                    len(entry.coefficient.den),
                )
                > MAX_CYCLIC_INPUT_DIGITS
            ):
                raise _validation_error(
                    "coefficient_bound",
                    "cyclic-symbol rationals are limited to "
                    f"{MAX_CYCLIC_INPUT_DIGITS} decimal digits",
                )
        return self


class CyclicRationalRankKernelProfileRequest(StrictModel):
    """Request the exact rational Galois-component profile of one cyclic map."""

    symbol: CyclicRationalBlockSymbol


class CyclotomicNonzeroMinor(StrictModel):
    """One source-bound nonzero square minor establishing a rank lower bound."""

    row_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_CYCLIC_GLOBAL_AXIS
    )
    column_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_CYCLIC_GLOBAL_AXIS
    )
    determinant: SimpleNumberFieldElement

    @model_validator(mode="after")
    def require_square_canonical_minor(self) -> Self:
        if len(self.row_indices) != len(self.column_indices):
            raise _validation_error("minor_shape", "rank minor must be square")
        if self.row_indices != tuple(sorted(set(self.row_indices))) or (
            self.column_indices != tuple(sorted(set(self.column_indices)))
        ):
            raise _validation_error(
                "minor_order", "rank-minor indices must be unique and increasing"
            )
        if not any(
            coefficient.as_fraction()
            for coefficient in self.determinant.coefficients_ascending
        ):
            raise _validation_error(
                "zero_minor", "a rank-certificate minor determinant must be nonzero"
            )
        return self


class CyclotomicRankKernelComponent(StrictModel):
    """Rank and kernel over one rational cyclotomic Galois component.

    The component is ``QQ[alpha]/(Phi_order(alpha))``; entries and basis
    vectors use the ascending power basis fixed by ``field``.  For positive
    rank, ``nonzero_minor`` proves the lower bound.  The returned independent
    kernel basis has dimension ``source_dimension-rank`` and its annihilation
    proves the matching upper bound.
    """

    order: int = Field(ge=1, le=MAX_CYCLIC_PERIOD)
    field: SimpleNumberFieldPresentation
    component_matrix: SimpleNumberFieldMatrix
    rank: int = Field(ge=0, le=MAX_CYCLIC_GLOBAL_AXIS)
    nullity: int = Field(ge=0, le=MAX_CYCLIC_GLOBAL_AXIS)
    kernel_basis: SimpleNumberFieldVectorSpaceBasis
    nonzero_minor: CyclotomicNonzeroMinor | None = None
    crt_idempotent: RationalPolynomial
    generator: Literal["CLASS_OF_X"] = "CLASS_OF_X"

    @model_validator(mode="after")
    def require_component_invariants(self) -> Self:
        if self.field.degree != _euler_phi(self.order):
            raise _validation_error(
                "cyclotomic_degree",
                "component field degree must equal Euler phi(order)",
            )
        if self.component_matrix.presentation != self.field:
            raise _validation_error(
                "component_field", "component matrix must use its declared field"
            )
        if self.kernel_basis.presentation != self.field:
            raise _validation_error(
                "kernel_field", "component kernel must use its declared field"
            )
        source_dimension = len(self.component_matrix.entries[0])
        target_dimension = len(self.component_matrix.entries)
        if self.kernel_basis.ambient_dimension != source_dimension:
            raise _validation_error(
                "kernel_ambient",
                "component kernel ambient dimension must match columns",
            )
        if self.rank > min(source_dimension, target_dimension):
            raise _validation_error(
                "rank_bound", "component rank exceeds its matrix dimensions"
            )
        if self.rank + self.nullity != source_dimension:
            raise _validation_error(
                "rank_nullity",
                "component rank plus nullity must equal source dimension",
            )
        if len(self.kernel_basis.vectors) != self.nullity:
            raise _validation_error(
                "kernel_dimension", "component kernel basis size must equal nullity"
            )
        if (self.rank == 0) != (self.nonzero_minor is None):
            raise _validation_error(
                "minor_presence", "positive component rank requires one nonzero minor"
            )
        if self.nonzero_minor is not None:
            minor = self.nonzero_minor
            if len(minor.row_indices) != self.rank:
                raise _validation_error(
                    "minor_rank", "rank-minor order must equal component rank"
                )
            if any(index >= target_dimension for index in minor.row_indices) or any(
                index >= source_dimension for index in minor.column_indices
            ):
                raise _validation_error(
                    "minor_axis", "rank-minor indices exceed component dimensions"
                )
            if minor.determinant.presentation != self.field:
                raise _validation_error(
                    "minor_field", "rank-minor determinant must use the component field"
                )
        if self.crt_idempotent.variables != ("x",):
            raise _validation_error(
                "crt_ring", "CRT idempotents must belong to the canonical QQ[x] ring"
            )
        return self


class CyclicRationalRankKernelProfile(StrictModel):
    """The complete rational cyclotomic rank/kernel decomposition of a source.

    Components are the irreducible rational Galois factors indexed by divisors
    ``d`` of the period, not separately chosen complex embeddings.  The CRT
    idempotents and ascending component power bases reconstruct the global
    kernel as a deterministic rational basis in shift-then-block order.
    """

    symbol: CyclicRationalBlockSymbol
    decomposition: Literal["RATIONAL_GALOIS_CYCLOTOMIC_COMPONENTS"] = (
        "RATIONAL_GALOIS_CYCLOTOMIC_COMPONENTS"
    )
    components: tuple[CyclotomicRankKernelComponent, ...] = Field(
        min_length=1, max_length=MAX_CYCLIC_COMPONENTS
    )
    exceptional_component_orders: tuple[int, ...] = Field(
        max_length=MAX_CYCLIC_COMPONENTS
    )
    global_rank: int = Field(ge=0, le=MAX_CYCLIC_GLOBAL_AXIS)
    global_nullity: int = Field(ge=0, le=MAX_CYCLIC_GLOBAL_AXIS)
    global_kernel_basis: RationalVectorSpaceBasis
    global_coordinate_order: Literal["SHIFT_THEN_BLOCK_COORDINATE"] = (
        "SHIFT_THEN_BLOCK_COORDINATE"
    )
    reconstruction: Literal["CRT_IDEMPOTENT_TIMES_COMPONENT_POWER_BASIS_V1"] = (
        "CRT_IDEMPOTENT_TIMES_COMPONENT_POWER_BASIS_V1"
    )

    @model_validator(mode="before")
    @classmethod
    def canonicalize_transport_arrays(cls, data: Any) -> Any:
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_complete_source_bound_profile(self) -> Self:
        expected_orders = tuple(
            divisor
            for divisor in range(1, self.symbol.period + 1)
            if self.symbol.period % divisor == 0
        )
        orders = tuple(component.order for component in self.components)
        if orders != expected_orders:
            raise _validation_error(
                "component_orders",
                "components must contain every period divisor in increasing order",
            )
        for component in self.components:
            matrix = component.component_matrix
            if (
                len(matrix.entries) != self.symbol.target_block_dimension
                or len(matrix.entries[0]) != self.symbol.source_block_dimension
            ):
                raise _validation_error(
                    "component_shape",
                    "every component matrix must match the source block dimensions",
                )
        exceptional = tuple(
            component.order for component in self.components if component.nullity > 0
        )
        if self.exceptional_component_orders != exceptional:
            raise _validation_error(
                "exceptional_components",
                "exceptional orders must be exactly the positive-nullity components",
            )
        rank = sum(
            component.field.degree * component.rank for component in self.components
        )
        nullity = sum(
            component.field.degree * component.nullity for component in self.components
        )
        if (self.global_rank, self.global_nullity) != (rank, nullity):
            raise _validation_error(
                "global_dimensions",
                "global rank and nullity must be the degree-weighted component sums",
            )
        source_dimension = self.symbol.period * self.symbol.source_block_dimension
        if rank + nullity != source_dimension:
            raise _validation_error(
                "global_rank_nullity",
                "global rank plus nullity must equal the global source dimension",
            )
        if (
            self.global_kernel_basis.ambient_dimension != source_dimension
            or len(self.global_kernel_basis.vectors) != nullity
        ):
            raise _validation_error(
                "global_kernel",
                "global rational kernel basis must realize the declared nullity",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        symbol: CyclicRationalBlockSymbol,
        components: tuple[CyclotomicRankKernelComponent, ...],
        global_rank: int,
        global_nullity: int,
        global_kernel_basis: RationalVectorSpaceBasis,
    ) -> Self:
        return cls.model_construct(
            symbol=symbol,
            decomposition="RATIONAL_GALOIS_CYCLOTOMIC_COMPONENTS",
            components=components,
            exceptional_component_orders=tuple(
                component.order for component in components if component.nullity > 0
            ),
            global_rank=global_rank,
            global_nullity=global_nullity,
            global_kernel_basis=global_kernel_basis,
            global_coordinate_order="SHIFT_THEN_BLOCK_COORDINATE",
            reconstruction="CRT_IDEMPOTENT_TIMES_COMPONENT_POWER_BASIS_V1",
        )


__all__ = [
    "MAX_CYCLIC_COMPONENTS",
    "MAX_CYCLIC_FIELD_WORK",
    "MAX_CYCLIC_GLOBAL_AXIS",
    "MAX_CYCLIC_INPUT_DIGITS",
    "MAX_CYCLIC_PERIOD",
    "MAX_CYCLIC_SYMBOL_ENTRIES",
    "CyclicRationalBlockSymbol",
    "CyclicRationalBlockSymbolEntry",
    "CyclicRationalRankKernelProfile",
    "CyclicRationalRankKernelProfileRequest",
    "CyclotomicNonzeroMinor",
    "CyclotomicRankKernelComponent",
]
