"""Source-bound exact inertia strata of a one-parameter polynomial matrix."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.matrices.symbolic.values import RationalPolynomialMatrix
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)


class InertiaParameterBoundary(StrictModel):
    """Exact parameter value and a rational interval isolating that value.

    Irrational values use their primitive irreducible polynomial and selected
    real-root index. Rational values have singleton isolating intervals.
    Parsing is structural; the producer establishes the isolation relation.
    """

    value: CanonicalRational | RealAlgebraicValue
    isolating_interval: RationalIsolatingInterval


class _InertiaCounts(StrictModel):
    n_positive: int = Field(ge=0, le=128)
    n_negative: int = Field(ge=0, le=128)
    n_zero: int = Field(ge=0, le=128)


class InertiaPointCell(_InertiaCounts):
    """A singleton parameter cell, including both domain endpoints."""

    cell_type: Literal["POINT"] = "POINT"
    parameter: InertiaParameterBoundary


class InertiaOpenCell(_InertiaCounts):
    """The open interval between the two exact parameter boundaries."""

    cell_type: Literal["OPEN"] = "OPEN"
    lower: InertiaParameterBoundary
    upper: InertiaParameterBoundary


InertiaCell = Annotated[
    InertiaPointCell | InertiaOpenCell, Field(discriminator="cell_type")
]


class InertiaCellsRequest(StrictModel):
    """A square symmetric QQ[t] matrix on a closed rational interval."""

    matrix: RationalPolynomialMatrix
    interval: ClosedRationalInterval


class InertiaCellsResult(StrictModel):
    """Alternating point/open cells covering exactly the retained interval.

    Every rank-drop value is retained. Deterministic refinements from separate
    diagonal blocks are allowed; equal-inertia neighbors are not merged.
    """

    matrix: RationalPolynomialMatrix
    interval: ClosedRationalInterval
    cells: tuple[InertiaCell, ...] = Field(min_length=1, max_length=261)

    @model_validator(mode="after")
    def require_partition_shape(self) -> Self:
        n = self.matrix.row_count
        if self.matrix.column_count != n:
            raise ValueError("source matrix must be square")
        if len(self.cells) % 2 != 1:
            raise ValueError("cells must alternate POINT and OPEN, ending at POINT")
        for i, cell in enumerate(self.cells):
            if cell.n_positive + cell.n_negative + cell.n_zero != n:
                raise ValueError("cell inertia counts must sum to the matrix order")
            if i % 2 == 0:
                if not isinstance(cell, InertiaPointCell):
                    raise ValueError("even cells must be points")
            else:
                previous, following = self.cells[i - 1], self.cells[i + 1]
                if (
                    not isinstance(cell, InertiaOpenCell)
                    or not isinstance(previous, InertiaPointCell)
                    or not isinstance(following, InertiaPointCell)
                ):
                    raise ValueError("open cells must lie between points")
                if (
                    cell.lower != previous.parameter
                    or cell.upper != following.parameter
                ):
                    raise ValueError(
                        "open boundaries must agree with adjacent point cells"
                    )
        first, last = self.cells[0], self.cells[-1]
        assert isinstance(first, InertiaPointCell) and isinstance(
            last, InertiaPointCell
        )
        if (
            first.parameter.value != self.interval.lower
            or last.parameter.value != self.interval.upper
        ):
            raise ValueError("point cells must retain the closed domain endpoints")
        if self.interval.lower == self.interval.upper and len(self.cells) != 1:
            raise ValueError("singleton domains have exactly one point cell")
        return self
