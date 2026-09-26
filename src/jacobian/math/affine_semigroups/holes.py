"""Exact rank-two affine-semigroup holes through a positive degree bound."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from math import ceil, floor
from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    DecimalIntegerEncoding,
    ExactInteger,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.group_lattice import compute_group_lattice
from jacobian.math.affine_semigroups.semigroup import (
    MAX_AFFINE_DIGITS,
    MAX_AFFINE_GENERATORS,
    AffineConfiguration,
    PositiveAffineSemigroup,
    _admit_semigroup,
    _hilbert_rays,
)

MAX_AFFINE_HOLE_SCALAR_DIGITS = 64
MAX_AFFINE_HOLE_CANDIDATES = 50_000
MAX_AFFINE_HOLE_WORK = 2_000_000
MAX_AFFINE_HOLE_OUTPUT_BYTES = 2_000_000
MAX_AFFINE_HOLE_LABEL_CHARS = 4_096
AffineHoleDegree = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_AFFINE_HOLE_SCALAR_DIGITS)
]


class AffineSemigroupHolesRequest(StrictModel):
    """Request all holes of one positive rank-two semigroup through a degree."""

    semigroup: PositiveAffineSemigroup
    max_degree: AffineHoleDegree = Field(
        description=(
            "Inclusive bound for the retained positive rational grading. The "
            "operation admits the containing lattice-coordinate box, generated "
            "state work, and complete exact output before enumerating points."
        )
    )


class AffineSemigroupHoleProfile(StrictModel):
    """Every normalized-lattice point through a degree bound absent from S."""

    semigroup: PositiveAffineSemigroup
    max_degree: AffineHoleDegree
    holes: tuple[tuple[ExactInteger, ExactInteger], ...]

    @model_validator(mode="after")
    def require_canonical_hole_rows(self) -> Self:
        if self.semigroup.configuration.rows != 2:
            raise PydanticCustomError(
                "affine_semigroup.hole_profile_rows",
                "hole profiles retain a two-coordinate ambient row axis",
            )
        if len(self.holes) > MAX_AFFINE_HOLE_CANDIDATES:
            raise PydanticCustomError(
                "affine_semigroup.hole_profile_size",
                "hole profile exceeds the admitted candidate count",
            )
        if self.holes != tuple(sorted(set(self.holes))):
            raise PydanticCustomError(
                "affine_semigroup.hole_profile_order",
                "hole vectors must be distinct and in lexicographic order",
            )
        return self


def _preflight_source(
    value: object, max_degree: object
) -> tuple[PositiveAffineSemigroup, int]:
    """Bound forged typed axes and scalars before revalidation serialization."""
    if type(max_degree) is not int:
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="affine_semigroup.hole_degree_type",
            message="maximum degree must be an exact integer",
        )
    degree = max_degree
    if abs(degree) >= 10**MAX_AFFINE_HOLE_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="affine_semigroup.hole_degree_digits",
            message=(
                "maximum degree exceeds the "
                f"{MAX_AFFINE_HOLE_SCALAR_DIGITS}-digit hole-profile envelope"
            ),
        )
    return _preflight_hole_semigroup(value), degree


def _preflight_hole_semigroup(value: object) -> PositiveAffineSemigroup:
    """Validate the shared typed rank-two source before any model copy."""
    if type(value) is not PositiveAffineSemigroup:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.semigroup",
            message="semigroup must be a canonical positive affine semigroup",
        )
    configuration = value.configuration
    if type(configuration) is not AffineConfiguration:
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.configuration",
            message="configuration must be a canonical affine configuration",
        )
    if (
        type(configuration.row_labels) is not tuple
        or len(configuration.row_labels) != 2
        or type(configuration.generator_labels) is not tuple
        or not 2 <= len(configuration.generator_labels) <= MAX_AFFINE_GENERATORS
        or type(configuration.entries) is not tuple
        or len(configuration.entries) != 2
        or any(type(row) is not tuple for row in configuration.entries)
        or any(
            len(row) != len(configuration.generator_labels)
            for row in configuration.entries
        )
    ):
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.hole_configuration_shape",
            message=(
                "hole profiles require a full two-row configuration with "
                f"2..{MAX_AFFINE_GENERATORS} generators"
            ),
        )
    labels = (*configuration.row_labels, *configuration.generator_labels)
    if (
        any(type(label) is not str for label in labels)
        or sum(len(label) for label in labels) > MAX_AFFINE_HOLE_LABEL_CHARS
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.hole_label_bytes",
            message="source labels exceed the hole-profile output envelope",
        )
    if any(
        type(entry) is not int or abs(entry) >= 10**MAX_AFFINE_DIGITS
        for row in configuration.entries
        for entry in row
    ):
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.configuration_digits",
            message="configuration entries must be bounded exact integers",
        )
    if type(value.grading) is not tuple or len(value.grading) != 2:
        raise OperationDomainValidationError(
            location=("semigroup", "grading"),
            code="affine_semigroup.grading_shape",
            message="grading must have one canonical rational per ambient row",
        )
    for component in value.grading:
        if type(component) is not CanonicalRational:
            raise OperationDomainValidationError(
                location=("semigroup", "grading"),
                code="affine_semigroup.grading",
                message="grading components must be canonical rationals",
            )
        if type(component.num) is not int or type(component.den) is not int:
            raise OperationDomainValidationError(
                location=("semigroup", "grading"),
                code="affine_semigroup.grading",
                message="grading numerators and denominators must be exact integers",
            )
        try:
            require_bounded_rational(
                component,
                max_digits=MAX_AFFINE_HOLE_SCALAR_DIGITS,
                label="hole-profile grading",
            )
        except ValueError as exc:
            raise OperationResourceAdmissionError(
                location=("semigroup", "grading"),
                code="affine_semigroup.hole_grading_digits",
                message="grading components exceed the 64-digit hole-profile envelope",
            ) from exc
    try:
        return _admit_semigroup(value)
    except OperationDomainValidationError:
        raise
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.semigroup",
            message="semigroup is malformed or not positive",
        ) from exc


def _coordinate_configuration(
    source: PositiveAffineSemigroup,
) -> tuple[AffineConfiguration, tuple[tuple[int, int], tuple[int, int]]]:
    """Transport source columns to the exact generated-lattice basis."""
    group = compute_group_lattice(source.configuration)
    basis = group.lattice.basis.entries
    if len(basis) != 2 or any(len(row) != 2 for row in basis):
        raise ValueError("hole profiles require a full-rank generated lattice")
    coordinates = group.generator_lattice_coordinates.entries
    if len(coordinates) != source.configuration.columns:
        raise ArithmeticError("generated-lattice coordinates lost source columns")
    if any(abs(value) >= 10**MAX_AFFINE_DIGITS for row in coordinates for value in row):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.hole_lattice_coordinates",
            message="generated-lattice coordinates exceed the 8-digit envelope",
        )
    coordinate_configuration = AffineConfiguration(
        row_labels=("lattice_0", "lattice_1"),
        generator_labels=source.configuration.generator_labels,
        entries=tuple(
            tuple(coordinates[column][row] for column in range(len(coordinates)))
            for row in range(2)
        ),
    )
    basis_vectors = (tuple(basis[0]), tuple(basis[1]))
    return coordinate_configuration, basis_vectors  # type: ignore[return-value]


def _candidate_box(
    configuration: AffineConfiguration,
    grading: tuple[Fraction, Fraction],
    max_degree: int,
) -> tuple[
    tuple[int, int], tuple[int, int], int, tuple[tuple[int, int], tuple[int, int]]
]:
    """Return an exact coordinate box containing the admitted cone slice."""
    lower, upper = _hilbert_rays(configuration)
    lower_degree = sum(grading[row] * lower[row] for row in range(2))
    upper_degree = sum(grading[row] * upper[row] for row in range(2))
    if lower_degree <= 0 or upper_degree <= 0:
        raise ArithmeticError("source grading is not positive on the cone rays")
    vertices = (
        (Fraction(0), Fraction(0)),
        tuple(Fraction(max_degree, 1) * lower[row] / lower_degree for row in range(2)),
        tuple(Fraction(max_degree, 1) * upper[row] / upper_degree for row in range(2)),
    )
    low = tuple(ceil(min(vertex[row] for vertex in vertices)) for row in range(2))
    high = tuple(floor(max(vertex[row] for vertex in vertices)) for row in range(2))
    widths = tuple(max(0, high[row] - low[row] + 1) for row in range(2))
    box_count = widths[0] * widths[1]
    return (low, high, box_count, (lower, upper))  # type: ignore[return-value]


def _iter_candidate_coordinates(
    low: tuple[int, int],
    high: tuple[int, int],
    rays: tuple[tuple[int, int], tuple[int, int]],
    grading: tuple[Fraction, Fraction],
    max_degree: int,
) -> Iterator[tuple[int, int]]:
    lower, upper = rays
    for x in range(low[0], high[0] + 1):
        for y in range(low[1], high[1] + 1):
            if (
                lower[0] * y - lower[1] * x >= 0
                and x * upper[1] - y * upper[0] >= 0
                and grading[0] * x + grading[1] * y <= max_degree
            ):
                yield (x, y)


def holes_through_degree(
    semigroup: PositiveAffineSemigroup, max_degree: int
) -> AffineSemigroupHoleProfile:
    """Return every hole in ``cone(S) ∩ gp(S)`` of grading at most D.

    Only full-rank pointed cones in two ambient dimensions are admitted. The
    semigroup is represented in its exact generated-lattice basis; a bounded
    lattice-coordinate box is admitted before point enumeration. Reachable
    semigroup points are then generated by a finite positive-degree closure,
    avoiding a separate coefficient-box search for every candidate point.
    """
    source, degree_bound = _preflight_source(semigroup, max_degree)
    # Validate the requested geometric envelope even for a negative cutoff.
    lower, upper = _hilbert_rays(source.configuration)
    source_grading = tuple(value.as_fraction() for value in source.grading)
    if any(
        sum(source_grading[row] * ray[row] for row in range(2)) <= 0
        for ray in (lower, upper)
    ):
        raise ArithmeticError("source grading is not positive on the cone")
    if degree_bound < 0:
        return AffineSemigroupHoleProfile(
            semigroup=source, max_degree=degree_bound, holes=()
        )

    coordinate_configuration, basis = _coordinate_configuration(source)
    coordinate_grading: tuple[Fraction, Fraction] = (
        sum(
            (source_grading[row] * basis[0][row] for row in range(2)),
            Fraction(0),
        ),
        sum(
            (source_grading[row] * basis[1][row] for row in range(2)),
            Fraction(0),
        ),
    )
    low, high, box_count, rays = _candidate_box(
        coordinate_configuration,
        coordinate_grading,
        degree_bound,
    )
    if box_count > MAX_AFFINE_HOLE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="affine_semigroup.hole_candidate_bound",
            message=(
                f"lattice-coordinate candidate box {box_count} exceeds "
                f"{MAX_AFFINE_HOLE_CANDIDATES} points"
            ),
        )

    generator_vectors = tuple(dict.fromkeys(coordinate_configuration.columns_vectors))
    work_bound = box_count * (2 * len(generator_vectors) + 3)
    if work_bound > MAX_AFFINE_HOLE_WORK:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="affine_semigroup.hole_work_bound",
            message=(
                f"degree-bounded hole work {work_bound} exceeds "
                f"{MAX_AFFINE_HOLE_WORK} units"
            ),
        )

    max_coordinates = tuple(max(abs(low[axis]), abs(high[axis])) for axis in range(2))
    ambient_bounds = tuple(
        sum(max_coordinates[axis] * abs(basis[axis][row]) for axis in range(2))
        for row in range(2)
    )
    coordinate_digits = tuple(len(str(value)) for value in ambient_bounds)
    point_bytes = 2 * max(coordinate_digits) + 10
    source_label_chars = sum(
        len(label)
        for label in (
            *source.configuration.row_labels,
            *source.configuration.generator_labels,
        )
    )
    source_bytes = (
        6 * source_label_chars
        + source.configuration.rows
        * source.configuration.columns
        * (MAX_AFFINE_DIGITS + 4)
        + 2 * MAX_AFFINE_HOLE_SCALAR_DIGITS * len(source.grading)
        + MAX_AFFINE_HOLE_SCALAR_DIGITS
        + 512
    )
    output_bound = source_bytes + box_count * point_bytes
    if output_bound > MAX_AFFINE_HOLE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="affine_semigroup.hole_output_bound",
            message=(
                f"degree-bounded hole output {output_bound} bytes exceeds "
                f"{MAX_AFFINE_HOLE_OUTPUT_BYTES} bytes"
            ),
        )

    candidates = tuple(
        _iter_candidate_coordinates(low, high, rays, coordinate_grading, degree_bound)
    )
    candidate_set = set(candidates)
    reachable = {(0, 0)}
    pending = [(0, 0)]
    while pending:
        point = pending.pop()
        for generator in generator_vectors:
            neighbor = (point[0] + generator[0], point[1] + generator[1])
            if neighbor in candidate_set and neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)

    holes = tuple(
        sorted(
            (
                basis[0][0] * point[0] + basis[1][0] * point[1],
                basis[0][1] * point[0] + basis[1][1] * point[1],
            )
            for point in candidates
            if point not in reachable
        )
    )
    return AffineSemigroupHoleProfile(
        semigroup=source,
        max_degree=degree_bound,
        holes=holes,
    )


__all__ = [
    "AffineHoleDegree",
    "AffineSemigroupHoleProfile",
    "AffineSemigroupHolesRequest",
    "holes_through_degree",
]
