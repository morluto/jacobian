"""Provider-independent values for exact finite-field linear algebra."""

from __future__ import annotations

from typing import Any, Literal, Self

import rfc8785
from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import sha256_digest
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix, rank

_MAX_FIELD_ORDER = 4096
_MIN_MODULUS_COEFFICIENTS = 3
_MAX_MODULUS_COEFFICIENTS = 17
_MAX_AXIS_LABELS = 256
_MAX_DERIVATION_WORK = 1_000_000


def _digest(payload: dict[str, Any]) -> str:
    return sha256_digest(rfc8785.dumps(payload))


def _encoded_coordinates(value: FiniteFieldElement) -> int:
    return sum(
        coordinate * value.presentation.characteristic**power
        for power, coordinate in enumerate(value.coordinates)
    )


def _validate_presentation_shape(
    characteristic: int,
    modulus_coefficients: tuple[int, ...],
    generator: str,
    element_encoding_version: str,
) -> None:
    if type(characteristic) is not int or characteristic < 2:
        raise ValueError("characteristic must be a prime integer")
    if characteristic > _MAX_FIELD_ORDER:
        raise ValueError("characteristic exceeds the supported field-order bound")
    if not (
        _MIN_MODULUS_COEFFICIENTS
        <= len(modulus_coefficients)
        <= _MAX_MODULUS_COEFFICIENTS
    ):
        raise ValueError("finite extension modulus length is outside its bound")
    if modulus_coefficients[-1] != 1:
        raise ValueError("modulus must be monic")
    if any(
        type(value) is not int or not 0 <= value < characteristic
        for value in modulus_coefficients
    ):
        raise ValueError("modulus coefficients must be canonical field residues")
    if not generator:
        raise ValueError("generator must be nonempty")
    if element_encoding_version != "power-basis-v1":
        raise ValueError("unsupported finite-field element encoding")


class FiniteFieldPresentation(StrictModel):
    """An exact polynomial presentation with a fixed power-basis encoding."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "characteristic": 2,
                    "modulus_coefficients": [1, 1, 1],
                    "generator": "a",
                    "element_encoding_version": "power-basis-v1",
                }
            ]
        }
    )

    characteristic: int = Field(
        description="Prime p defining the base field GF(p).", examples=[2]
    )
    modulus_coefficients: tuple[int, ...] = Field(
        description=(
            "Constant-to-leading coefficients of a monic irreducible modulus over "
            "GF(characteristic); each coefficient is a canonical residue."
        ),
        examples=[[1, 1, 1]],
    )
    generator: str = Field(
        default="a",
        description="Name of the power-basis generator represented by the modulus.",
        examples=["a"],
    )
    element_encoding_version: str = Field(
        default="power-basis-v1",
        description="Fixed coordinate encoding for finite-field elements.",
        examples=["power-basis-v1"],
    )

    @model_validator(mode="after")
    def validate_presentation(self) -> Self:
        _validate_presentation_shape(
            self.characteristic,
            self.modulus_coefficients,
            self.generator,
            self.element_encoding_version,
        )
        if self.characteristic**self.degree > _MAX_FIELD_ORDER:
            raise ValueError("field order exceeds the supported bound")
        from sympy import Poly, isprime, symbols

        if not isprime(self.characteristic):
            raise ValueError("characteristic must be a prime integer")
        variable = symbols("x")
        polynomial = Poly(
            sum(
                coefficient * variable**power
                for power, coefficient in enumerate(self.modulus_coefficients)
            ),
            variable,
            modulus=self.characteristic,
        )
        if not polynomial.is_irreducible:
            raise ValueError("modulus must be irreducible over the prime field")
        return self

    @property
    def degree(self) -> int:
        return len(self.modulus_coefficients) - 1

    @property
    def order(self) -> int:
        return int(pow(self.characteristic, self.degree))

    @property
    def ordered_basis(self) -> tuple[str, ...]:
        return (
            "1",
            self.generator,
            *(f"{self.generator}^{power}" for power in range(2, self.degree)),
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "characteristic": self.characteristic,
                "element_encoding_version": self.element_encoding_version,
                "generator": self.generator,
                "modulus_coefficients": list(self.modulus_coefficients),
                "ordered_basis": list(self.ordered_basis),
                "value_type": "finite-field-presentation-v1",
            }
        )


class FiniteFieldElement(StrictModel):
    """Power-basis coordinates bound to one exact field presentation."""

    presentation: FiniteFieldPresentation
    coordinates: tuple[int, ...]

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if len(self.coordinates) != self.presentation.degree:
            raise ValueError("element coordinates must match the presentation degree")
        if any(
            type(value) is not int or not 0 <= value < self.presentation.characteristic
            for value in self.coordinates
        ):
            raise ValueError("element coordinates must be canonical field residues")
        return self

    @property
    def is_zero(self) -> bool:
        return not any(self.coordinates)

    @property
    def is_one(self) -> bool:
        return self.coordinates == (1,) + (0,) * (self.presentation.degree - 1)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "coordinates": list(self.coordinates),
                "presentation": self.presentation.digest,
                "value_type": "finite-field-element-v1",
            }
        )


class Axis(StrictModel):
    """An ordered semantic axis."""

    name: str
    labels: tuple[str, ...]

    @model_validator(mode="after")
    def validate_axis(self) -> Self:
        if not self.name:
            raise ValueError("axis name must be nonempty")
        if not self.labels or any(not label for label in self.labels):
            raise ValueError("axis labels must be nonempty")
        if len(self.labels) > _MAX_AXIS_LABELS:
            raise ValueError("axis exceeds the supported label bound")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("axis labels must be unique")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "labels": list(self.labels),
                "name": self.name,
                "value_type": "axis-v1",
            }
        )


class AxisBoundMatrix(StrictModel):
    """An immutable matrix bound to a field presentation and ordered axes."""

    presentation: FiniteFieldPresentation
    row_axis: Axis
    column_axis: Axis
    entries: tuple[tuple[FiniteFieldElement, ...], ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if len(self.entries) != len(self.row_axis.labels):
            raise ValueError("matrix rows must match the row axis")
        if any(len(row) != len(self.column_axis.labels) for row in self.entries):
            raise ValueError("matrix columns must match the column axis")
        if any(
            element.presentation != self.presentation
            for row in self.entries
            for element in row
        ):
            raise ValueError("matrix entries must use the matrix field presentation")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "column_axis": self.column_axis.digest,
                "entries": [
                    [list(element.coordinates) for element in row]
                    for row in self.entries
                ],
                "presentation": self.presentation.digest,
                "row_axis": self.row_axis.digest,
                "value_type": "axis-bound-matrix-v1",
            }
        )


class FiniteDimensionalSubspace(StrictModel):
    """An ordered independent matrix basis over the presentation's prime field."""

    presentation: FiniteFieldPresentation
    basis_axis: Axis
    basis: tuple[AxisBoundMatrix, ...]

    @model_validator(mode="after")
    def validate_subspace(self) -> Self:
        if len(self.basis) != len(self.basis_axis.labels):
            raise ValueError("subspace basis must match its basis axis")
        if not self.basis:
            raise ValueError("subspace basis must be nonempty")
        first = self.basis[0]
        if any(
            matrix.presentation != self.presentation
            or matrix.row_axis != first.row_axis
            or matrix.column_axis != first.column_axis
            for matrix in self.basis
        ):
            raise ValueError("subspace matrices must share their parent and axes")
        flattened_dimension = (
            len(first.row_axis.labels)
            * len(first.column_axis.labels)
            * self.presentation.degree
        )
        if flattened_dimension * len(self.basis) > _MAX_AXIS_LABELS**2:
            raise ValueError("subspace rank matrix exceeds its supported bound")
        flattened = tuple(
            tuple(
                coordinate
                for row in matrix.entries
                for element in row
                for coordinate in element.coordinates
            )
            for matrix in self.basis
        )
        coordinate_rows = tuple(zip(*flattened, strict=True))
        basis_matrix = PrimeFieldMatrix(
            prime=self.presentation.characteristic,
            entries=coordinate_rows,
            columns=len(self.basis),
        )
        if rank(basis_matrix) != len(self.basis):
            raise ValueError("subspace basis matrices must be linearly independent")
        return self

    @property
    def row_axis(self) -> Axis:
        return self.basis[0].row_axis

    @property
    def column_axis(self) -> Axis:
        return self.basis[0].column_axis

    @property
    def digest(self) -> str:
        return _digest(
            {
                "basis": [matrix.digest for matrix in self.basis],
                "basis_axis": self.basis_axis.digest,
                "presentation": self.presentation.digest,
                "value_type": "finite-dimensional-subspace-v1",
            }
        )


class ProjectivePoint(StrictModel):
    """A normalized projective point over one field and coordinate axis."""

    presentation: FiniteFieldPresentation
    axis: Axis
    coordinates: tuple[FiniteFieldElement, ...]

    @model_validator(mode="after")
    def validate_point(self) -> Self:
        if len(self.coordinates) != len(self.axis.labels):
            raise ValueError("projective coordinates must match their axis")
        if any(
            coordinate.presentation != self.presentation
            for coordinate in self.coordinates
        ):
            raise ValueError("projective coordinates must share their presentation")
        first_nonzero = next(
            (coordinate for coordinate in self.coordinates if not coordinate.is_zero),
            None,
        )
        if first_nonzero is None:
            raise ValueError("projective coordinates cannot all be zero")
        if not first_nonzero.is_one:
            raise ValueError("projective coordinates must be normalized")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "axis": self.axis.digest,
                "coordinates": [list(value.coordinates) for value in self.coordinates],
                "presentation": self.presentation.digest,
                "value_type": "projective-point-v1",
            }
        )


class ProjectiveLine(StrictModel):
    """The complete ordered projective line for one presentation and axis."""

    presentation: FiniteFieldPresentation
    axis: Axis
    points: tuple[ProjectivePoint, ...]

    @model_validator(mode="after")
    def validate_line(self) -> Self:
        expected = (self.presentation.order ** len(self.axis.labels) - 1) // (
            self.presentation.order - 1
        )
        if expected > _MAX_FIELD_ORDER:
            raise ValueError("projective line exceeds the supported direction bound")
        if len(self.points) != expected:
            raise ValueError("projective line must contain every direction")
        if any(
            point.presentation != self.presentation or point.axis != self.axis
            for point in self.points
        ):
            raise ValueError("projective line points must share their parent and axis")
        if len({point.digest for point in self.points}) != len(self.points):
            raise ValueError("projective line cannot repeat a direction")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "axis": self.axis.digest,
                "points": [point.digest for point in self.points],
                "presentation": self.presentation.digest,
                "value_type": "projective-line-v1",
            }
        )


class FiniteLinearMap(StrictModel):
    """A matrix-defined linear map with exact source and target axes."""

    source_axis: Axis
    target_axis: Axis
    matrix: PrimeFieldMatrix

    @model_validator(mode="after")
    def validate_linear_map(self) -> Self:
        if self.matrix.columns != len(self.source_axis.labels):
            raise ValueError("linear-map columns must match the source axis")
        if len(self.matrix.entries) != len(self.target_axis.labels):
            raise ValueError("linear-map rows must match the target axis")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "entries": [list(row) for row in self.matrix.entries],
                "prime": self.matrix.prime,
                "source_axis": self.source_axis.digest,
                "target_axis": self.target_axis.digest,
                "value_type": "finite-linear-map-v1",
            }
        )


class RankResult(StrictModel):
    """The exact rank of a direction-bound finite linear map."""

    direction: ProjectivePoint
    linear_map: FiniteLinearMap
    rank: int

    @model_validator(mode="after")
    def validate_rank(self) -> Self:
        if self.linear_map.matrix.prime != self.direction.presentation.characteristic:
            raise ValueError("rank map must use the direction's prime field")
        maximum_rank = min(
            len(self.linear_map.matrix.entries),
            self.linear_map.matrix.columns,
        )
        if type(self.rank) is not int or not 0 <= self.rank <= maximum_rank:
            raise ValueError("rank is outside the linear-map dimensions")
        if self.rank != rank(self.linear_map.matrix):
            raise ValueError("rank must match the exact bound linear map")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "direction": self.direction.digest,
                "linear_map": self.linear_map.digest,
                "rank": self.rank,
                "value_type": "rank-result-v1",
            }
        )


def _direction_rank_work(
    subspace: FiniteDimensionalSubspace,
    direction_count: int,
) -> int:
    source_dimension = len(subspace.basis)
    target_dimension = len(subspace.column_axis.labels) * subspace.presentation.degree
    restriction = source_dimension * len(subspace.row_axis.labels) * target_dimension
    rank_work = (
        target_dimension * source_dimension * min(target_dimension, source_dimension)
    )
    return direction_count * (restriction + rank_work)


class DirectionRankLedger(StrictModel):
    """An ordered, exact binding from projective directions to rank results."""

    subspace: FiniteDimensionalSubspace
    entries: tuple[RankResult, ...] = Field(min_length=1, max_length=_MAX_FIELD_ORDER)

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        first = self.entries[0]
        expected_directions = (
            self.subspace.presentation.order ** len(self.subspace.row_axis.labels) - 1
        ) // (self.subspace.presentation.order - 1)
        if len(self.entries) != expected_directions:
            raise ValueError(
                "direction-rank ledger must contain every projective direction"
            )
        if len({entry.direction.digest for entry in self.entries}) != len(self.entries):
            raise ValueError("direction-rank ledger cannot repeat a direction")
        if any(
            entry.direction.presentation != first.direction.presentation
            or entry.direction.axis != first.direction.axis
            or entry.linear_map.source_axis != first.linear_map.source_axis
            or entry.linear_map.target_axis != first.linear_map.target_axis
            or entry.linear_map.matrix.prime != first.linear_map.matrix.prime
            for entry in self.entries
        ):
            raise ValueError("direction-rank entries must share their bound semantics")
        if first.direction.presentation != self.subspace.presentation:
            raise ValueError("ledger directions must use the subspace presentation")
        if first.direction.axis != self.subspace.row_axis:
            raise ValueError("ledger directions must use the subspace row axis")
        if first.linear_map.source_axis != self.subspace.basis_axis:
            raise ValueError("ledger maps must use the subspace basis axis")
        work = _direction_rank_work(self.subspace, len(self.entries))
        if work > _MAX_DERIVATION_WORK:
            raise ValueError("direction-rank ledger exceeds its derivation work budget")
        from jacobian.math.finite_fields import _sympy

        for entry in self.entries:
            if entry.linear_map != _sympy.restrict_scalars(
                self.subspace, entry.direction
            ):
                raise ValueError(
                    "direction-rank ledger map does not match the bound subspace"
                )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "entries": [entry.digest for entry in self.entries],
                "subspace": self.subspace.digest,
                "value_type": "direction-rank-ledger-v1",
            }
        )


class OrbitDistribution(StrictModel):
    """Orbit-size counts derived from one exact direction-rank ledger."""

    ledger: DirectionRankLedger
    counts: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if self.counts != _orbit_counts(self.ledger):
            raise ValueError("orbit counts do not match the direction-rank ledger")
        return self

    @classmethod
    def from_ledger(cls, ledger: DirectionRankLedger) -> Self:
        return cls(ledger=ledger, counts=_orbit_counts(ledger))

    @property
    def digest(self) -> str:
        return _digest(
            {
                "counts": [list(item) for item in self.counts],
                "ledger": self.ledger.digest,
                "value_type": "orbit-distribution-v1",
            }
        )


class FinitePolynomial(StrictModel):
    """A canonical univariate polynomial over one exact field presentation."""

    presentation: FiniteFieldPresentation
    variable: str
    coefficients: tuple[FiniteFieldElement, ...]

    @model_validator(mode="after")
    def validate_polynomial(self) -> Self:
        if not self.variable:
            raise ValueError("finite polynomial variable must be nonempty")
        if not self.coefficients:
            raise ValueError("finite polynomial requires a constant coefficient")
        if len(self.coefficients) > _MAX_FIELD_ORDER:
            raise ValueError("finite polynomial exceeds the supported degree bound")
        if any(
            coefficient.presentation != self.presentation
            for coefficient in self.coefficients
        ):
            raise ValueError("finite polynomial coefficients must share their parent")
        if len(self.coefficients) > 1 and self.coefficients[-1].is_zero:
            raise ValueError(
                "finite polynomial cannot have a trailing zero coefficient"
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "coefficients": [
                    list(value.coordinates) for value in self.coefficients
                ],
                "presentation": self.presentation.digest,
                "value_type": "finite-polynomial-v1",
                "variable": self.variable,
            }
        )


class FinitePolynomialMap(StrictModel):
    """A polynomial self-map of one exactly presented finite field."""

    domain: FiniteFieldPresentation
    codomain: FiniteFieldPresentation
    polynomial: FinitePolynomial

    @model_validator(mode="after")
    def validate_map(self) -> Self:
        if self.polynomial.presentation != self.domain or self.codomain != self.domain:
            raise ValueError(
                "finite polynomial map must use one exact field presentation"
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "codomain": self.codomain.digest,
                "domain": self.domain.digest,
                "polynomial": self.polynomial.digest,
                "value_type": "finite-polynomial-map-v1",
            }
        )


class FiniteMapTable(StrictModel):
    """A complete ordered evaluation table for one exact finite map."""

    map: FinitePolynomialMap
    entries: tuple[tuple[FiniteFieldElement, FiniteFieldElement], ...] = Field(
        min_length=1,
        max_length=_MAX_FIELD_ORDER,
    )

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        if len(self.entries) != self.map.domain.order:
            raise ValueError("finite map table must enumerate the complete domain")
        if self.map.domain.order > _MAX_FIELD_ORDER:
            raise ValueError("finite map table exceeds the supported domain bound")
        inputs = tuple(source for source, _ in self.entries)
        if any(value.presentation != self.map.domain for value in inputs):
            raise ValueError("finite map table inputs must use the exact domain")
        if len({value.digest for value in inputs}) != len(inputs):
            raise ValueError("finite map table cannot repeat a domain element")
        if tuple(map(_encoded_coordinates, inputs)) != tuple(
            range(self.map.domain.order)
        ):
            raise ValueError("finite map table inputs must use canonical domain order")
        if any(value.presentation != self.map.codomain for _, value in self.entries):
            raise ValueError("finite map table outputs must use the exact codomain")
        work = (
            len(self.entries)
            * len(self.map.polynomial.coefficients)
            * self.map.domain.degree
        )
        if work > _MAX_DERIVATION_WORK:
            raise ValueError("finite map table exceeds its derivation work budget")
        from jacobian.math.finite_fields import _sympy

        expected = _sympy.evaluate_polynomial_values(self.map.polynomial, inputs)
        if any(
            target.coordinates != coordinates
            for (_, target), coordinates in zip(self.entries, expected, strict=True)
        ):
            raise ValueError("finite map table targets must match the bound polynomial")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "entries": [
                    [source.digest, target.digest] for source, target in self.entries
                ],
                "map": self.map.digest,
                "value_type": "finite-map-table-v1",
            }
        )


def _fibers_for_table(
    table: FiniteMapTable,
) -> tuple[tuple[FiniteFieldElement, tuple[FiniteFieldElement, ...]], ...]:
    grouped: dict[str, tuple[FiniteFieldElement, list[FiniteFieldElement]]] = {}
    for source, target in table.entries:
        _, sources = grouped.setdefault(target.digest, (target, []))
        sources.append(source)
    return tuple((target, tuple(sources)) for target, sources in grouped.values())


class FiberPartition(StrictModel):
    """The nonempty fibers of one complete finite map table."""

    table: FiniteMapTable
    fibers: tuple[tuple[FiniteFieldElement, tuple[FiniteFieldElement, ...]], ...]

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if not self.fibers or any(not sources for _, sources in self.fibers):
            raise ValueError("fiber partition requires nonempty fibers")
        if self.fibers != _fibers_for_table(self.table):
            raise ValueError("fibers must partition the exact evaluated table")
        return self

    @classmethod
    def from_table(cls, table: FiniteMapTable) -> Self:
        return cls(table=table, fibers=_fibers_for_table(table))

    @property
    def digest(self) -> str:
        return _digest(
            {
                "fibers": [
                    [target.digest, [source.digest for source in sources]]
                    for target, sources in self.fibers
                ],
                "table": self.table.digest,
                "value_type": "fiber-partition-v1",
            }
        )


class CollisionResult(StrictModel):
    """Whether a complete finite map table has a collision."""

    table: FiniteMapTable
    status: Literal["COLLISION", "INJECTIVE"]
    left: FiniteFieldElement | None = None
    right: FiniteFieldElement | None = None
    image: FiniteFieldElement | None = None

    @model_validator(mode="after")
    def validate_collision(self) -> Self:
        if self.status == "INJECTIVE":
            if any(value is not None for value in (self.left, self.right, self.image)):
                raise ValueError("an injective table cannot carry collision values")
            return self
        if self.left is None or self.right is None or self.image is None:
            raise ValueError("a collision result requires both inputs and their image")
        if self.left == self.right:
            raise ValueError("collision inputs must be distinct")
        evaluated = {source.digest: target for source, target in self.table.entries}
        if (
            evaluated.get(self.left.digest) != self.image
            or evaluated.get(self.right.digest) != self.image
        ):
            raise ValueError("collision must occur in the exact bound table")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "image": self.image.digest if self.image is not None else None,
                "left": self.left.digest if self.left is not None else None,
                "right": self.right.digest if self.right is not None else None,
                "table": self.table.digest,
                "status": self.status,
                "value_type": "finite-map-collision-v1",
            }
        )


class PermutationResult(StrictModel):
    """Whether a complete finite map table is a permutation."""

    table: FiniteMapTable
    status: Literal["PERMUTATION", "NOT_PERMUTATION"]
    inverse_entries: tuple[tuple[FiniteFieldElement, FiniteFieldElement], ...] = ()

    @model_validator(mode="after")
    def validate_permutation(self) -> Self:
        injective = len({target.digest for _, target in self.table.entries}) == len(
            self.table.entries
        )
        if self.status == "NOT_PERMUTATION":
            if injective or self.inverse_entries:
                raise ValueError("a non-permutation result cannot carry an inverse")
            return self
        if not injective:
            raise ValueError("a permutation result requires an injective table")
        expected = tuple(
            sorted(
                ((target, source) for source, target in self.table.entries),
                key=lambda entry: _encoded_coordinates(entry[0]),
            )
        )
        if self.inverse_entries != expected:
            raise ValueError("inverse table does not bind the exact permutation")
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "inverse_entries": [
                    [target.digest, source.digest]
                    for target, source in self.inverse_entries
                ],
                "table": self.table.digest,
                "status": self.status,
                "value_type": "finite-map-permutation-v1",
            }
        )


def _orbit_counts(ledger: DirectionRankLedger) -> tuple[tuple[int, int], ...]:
    first = ledger.entries[0]
    presentation = first.direction.presentation
    expected_directions = (
        presentation.order ** len(first.direction.axis.labels) - 1
    ) // (presentation.order - 1)
    if len(ledger.entries) != expected_directions:
        raise ValueError("orbit aggregation requires every projective direction")
    prime = presentation.characteristic
    target_dimension = len(first.linear_map.target_axis.labels)
    counts: dict[int, int] = {1: expected_directions}
    for entry in ledger.entries:
        orbit_size = prime**entry.rank
        counts[orbit_size] = counts.get(orbit_size, 0) + prime ** (
            target_dimension - entry.rank
        )
    return tuple(sorted(counts.items()))
