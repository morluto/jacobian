"""Exact rational linear-system contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix

MAX_LINEAR_DIMENSION = 32
MAX_RATIONAL_DIGITS = 256

LinearVariableName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$",
        strict=True,
    ),
]


def _require_bounded_rationals(values: tuple[CanonicalRational, ...]) -> None:
    for value in values:
        require_bounded_rational(
            value,
            max_digits=MAX_RATIONAL_DIGITS,
            label="linear-system rational",
        )


class LinearRationalSystem(StrictModel):
    """One declared finite system ``A x = b`` over exact rationals."""

    system_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    relation: Literal["AX_EQUALS_B"] = "AX_EQUALS_B"
    variables: tuple[LinearVariableName, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    coefficients: RationalMatrix
    rhs: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )

    @model_validator(mode="after")
    def require_matching_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("linear-system variable names must be unique")
        if len(self.coefficients.entries[0]) != len(self.variables):
            raise ValueError(
                "the coefficient column count must equal the declared variable count"
            )
        if len(self.coefficients.entries) != len(self.rhs):
            raise ValueError(
                "the right-hand side length must equal the coefficient row count"
            )
        _require_bounded_rationals(
            tuple(value for row in self.coefficients.entries for value in row)
            + self.rhs
        )
        return self


class LinearRationalSolutionFindRequest(StrictModel):
    """Ask for one exact solution of a rational linear system."""

    system: LinearRationalSystem


class LinearRationalSolutionResult(StrictModel):
    """One exact solution outcome bound to its declared source system.

    Retains the canonical ``LinearRationalSystem`` so validation replays the
    defining relation: an admitted solution carries one coordinate per
    declared variable and satisfies ``A x = b`` exactly over QQ, while an
    inconsistent outcome carries no values and requires the retained system
    itself to be inconsistent (``rank(A) < rank([A | b])``).  The coefficient
    domain admits at least one row and column, so zero-row shapes are rejected
    by request admission rather than silently dropped.
    """

    system: LinearRationalSystem
    status: Literal["SOLUTION", "INCONSISTENT"] = "SOLUTION"
    values: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )

    @model_validator(mode="after")
    def bind_solution_to_source(self) -> Self:
        produced = self.status == "SOLUTION"
        if produced != (self.values is not None):
            raise ValueError("solution values must agree with the result status")
        if self.values is None:
            from jacobian.math.matrices._operations import _system_rank_replay

            coefficient_rank, augmented_rank = _system_rank_replay(
                self.system.coefficients, self.system.rhs
            )
            if coefficient_rank >= augmented_rank:
                raise ValueError(
                    "an inconsistent outcome requires rank(A) < rank([A | b]) "
                    "on the source system"
                )
            return self
        if len(self.values) != len(self.system.variables):
            raise ValueError("solution length must equal the source variable count")
        components = [value.as_fraction() for value in self.values]
        for row, bound in zip(
            self.system.coefficients.entries,
            self.system.rhs,
            strict=True,
        ):
            residual = sum(
                coefficient.as_fraction() * component
                for coefficient, component in zip(row, components, strict=True)
            )
            if residual != bound.as_fraction():
                raise ValueError("solution does not satisfy A x = b exactly")
        return self


class LinearRationalInconsistencyResult(StrictModel):
    """One exact inconsistency outcome bound to its declared source system.

    Retains the canonical ``LinearRationalSystem`` so validation replays the
    defining relations: an admitted separating witness ``y`` carries one
    coordinate per source row, annihilates every source column exactly
    (``y^T A = 0``), and its recorded pairing equals ``y^T b`` on the
    retained right-hand side and is nonzero.  The witness is defined up to a
    nonzero scaling; the producer emits the backend-scaled witness whose
    pairing equals one.  A consistent outcome carries no witness and requires
    the retained system itself to be consistent (``rank(A) == rank([A | b])``).
    The coefficient domain admits at least one row and column, so zero-row
    shapes are rejected by request admission rather than silently dropped.
    """

    system: LinearRationalSystem
    status: Literal["INCONSISTENT", "CONSISTENT"] = "INCONSISTENT"
    left_witness: tuple[CanonicalRational, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational | None = None

    @model_validator(mode="after")
    def bind_witness_to_source(self) -> Self:
        produced = self.status == "INCONSISTENT"
        if produced != (self.left_witness is not None and self.rhs_pairing is not None):
            raise ValueError("inconsistency witness must agree with the result status")
        if self.left_witness is None or self.rhs_pairing is None:
            from jacobian.math.matrices._operations import _system_rank_replay

            coefficient_rank, augmented_rank = _system_rank_replay(
                self.system.coefficients, self.system.rhs
            )
            if coefficient_rank != augmented_rank:
                raise ValueError(
                    "a consistent outcome requires rank(A) == rank([A | b]) "
                    "on the source system"
                )
            return self
        if len(self.left_witness) != len(self.system.rhs):
            raise ValueError("witness length must equal the source row count")
        coordinates = [value.as_fraction() for value in self.left_witness]
        columns = range(len(self.system.coefficients.entries[0]))
        for column in columns:
            if (
                sum(
                    row[column].as_fraction() * coordinate
                    for row, coordinate in zip(
                        self.system.coefficients.entries,
                        coordinates,
                        strict=True,
                    )
                )
                != 0
            ):
                raise ValueError("witness does not satisfy y^T A = 0 exactly")
        pairing = sum(
            bound.as_fraction() * coordinate
            for bound, coordinate in zip(self.system.rhs, coordinates, strict=True)
        )
        if pairing != self.rhs_pairing.as_fraction():
            raise ValueError("recorded pairing must equal y^T b on the source system")
        if pairing == 0:
            raise ValueError("separating witness must have a nonzero pairing")
        return self


class LinearRationalInconsistencyFindRequest(StrictModel):
    """Ask whether a rational linear system is inconsistent."""

    system: LinearRationalSystem
