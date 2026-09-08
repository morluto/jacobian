"""Exact fixed-node Lebesgue functions on closed rational intervals."""

from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisResult,
    RationalNodeSet,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.values import RationalPolynomial


class LebesgueIntervalSource(StrictModel):
    """A fixed node set and the interval on which its Lebesgue function is studied.

    The query interval may contain only some of the interpolation nodes, or
    lie outside their convex hull. Nodes retain their canonical increasing order.
    """

    nodes: RationalNodeSet
    interval: ClosedRationalInterval


class LebesgueIntervalRequest(StrictModel):
    source: LebesgueIntervalSource


class LebesgueCriticalPoint(StrictModel):
    """An isolated derivative root and the exact value of its cell polynomial."""

    point: RealAlgebraicValue
    point_isolating_interval: RationalIsolatingInterval
    derivative_multiplicity: int = Field(ge=1, le=31)
    value: RealAlgebraicValue
    value_isolating_interval: RationalIsolatingInterval


class LebesgueSignCell(StrictModel):
    """One positive-length cell, closed for evaluating its extrema.

    The sign vector refers to the open interior, where no basis polynomial
    vanishes. Endpoint values use the continuous extension. For a constant
    cell every point is stationary and maximizing; otherwise critical_points
    lists all derivative roots in the closed cell and maximizing_points lists
    every point attaining its maximum.
    """

    interval: ClosedRationalInterval
    basis_signs: tuple[Literal[-1, 1], ...] = Field(min_length=1, max_length=32)
    polynomial: RationalPolynomial
    endpoint_values: tuple[CanonicalRational, CanonicalRational]
    constant_on_cell: bool
    critical_points: tuple[LebesgueCriticalPoint, ...] = Field(max_length=31)
    maximum: RealAlgebraicValue
    maximum_isolating_interval: RationalIsolatingInterval
    maximizing_points: tuple[RealAlgebraicValue, ...] = Field(max_length=33)

    @model_validator(mode="after")
    def require_cell_structure(self) -> Self:
        if self.interval.lower.as_fraction() >= self.interval.upper.as_fraction():
            raise ValueError("Lebesgue sign cells must have positive length")
        if self.polynomial.variables != ("x",):
            raise ValueError("Lebesgue cell polynomials must use the basis variable x")
        if self.constant_on_cell and (self.critical_points or self.maximizing_points):
            raise ValueError(
                "constant cells represent the entire interval of maximizers"
            )
        if not self.constant_on_cell and not self.maximizing_points:
            raise ValueError("a nonconstant closed cell needs at least one maximizer")
        return self


class LebesgueIntervalProfile(StrictModel):
    """Complete sign cells, critical values, and the exact global maximum locus.

    The global maximizer set is the union of maximizing_intervals and the
    isolated maximizing_points. A singleton query has no positive-length
    sign cell and retains its one-point interval as the maximizing interval.
    """

    source: LebesgueIntervalSource
    basis: LagrangeBasisResult
    cells: tuple[LebesgueSignCell, ...] = Field(max_length=33)
    maximum: RealAlgebraicValue
    maximum_isolating_interval: RationalIsolatingInterval
    maximizing_points: tuple[RealAlgebraicValue, ...] = Field(max_length=1100)
    maximizing_intervals: tuple[ClosedRationalInterval, ...] = Field(max_length=33)

    @model_validator(mode="after")
    def require_complete_source_axes(self) -> Self:
        if self.basis.nodes != self.source.nodes:
            raise ValueError("the complete basis must retain the source node set")
        if any(
            len(cell.basis_signs) != len(self.source.nodes.nodes) for cell in self.cells
        ):
            raise ValueError("every sign vector must retain the complete basis axis")
        if self.source.interval.lower == self.source.interval.upper:
            if self.cells:
                raise ValueError("a singleton query has no positive-length sign cells")
        elif (
            not self.cells
            or self.cells[0].interval.lower != self.source.interval.lower
            or self.cells[-1].interval.upper != self.source.interval.upper
            or any(
                a.interval.upper != b.interval.lower for a, b in pairwise(self.cells)
            )
        ):
            raise ValueError(
                "sign cells must cover the closed source interval without gaps"
            )
        if not self.maximizing_points and not self.maximizing_intervals:
            raise ValueError("a closed interval must have a nonempty maximum locus")
        return self
