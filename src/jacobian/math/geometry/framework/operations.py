"""Exact planar framework rigidity operations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import NoReturn

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.framework._bounds import (
    MAX_FRAMEWORK_COORDINATE_WORK,
    difference_work,
    rational_parse_work,
)
from jacobian.math.geometry.framework._models import (
    PlanarRigidityProfile,
    _require_planar_framework_shape,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices._operation_models import MAX_INPUT_SCALAR_DIGITS
from jacobian.math.matrices.operations import rank_result
from jacobian.math.matrices.values import (
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)

_MATRIX_SCALAR_LIMIT = 10**MAX_INPUT_SCALAR_DIGITS


@dataclass(frozen=True, slots=True)
class _FrameworkAdmission:
    vertex_axis: tuple[str, ...]
    edge_axis: tuple[tuple[str, str], ...]
    coordinates: dict[str, tuple[Fraction, Fraction]]
    source_parse_work: int
    coordinate_difference_work: int


def _reject(location: tuple[str | int, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=(
            code
            if code.startswith("geometry.framework.")
            else f"geometry.framework.{code}"
        ),
        message=message,
    )


def _admit_framework(
    configuration: PointConfiguration,
    graph: SimpleUndirectedGraph,
) -> _FrameworkAdmission:
    try:
        _require_planar_framework_shape(configuration, graph)
    except PydanticCustomError as exc:
        _reject(("configuration",), exc.type, exc.message())

    vertex_axis = tuple(point.label for point in configuration.points)
    edge_axis = tuple(sorted(graph.edges))
    canonical_coordinates = {
        point.label: (point.coordinates[0], point.coordinates[1])
        for point in configuration.points
    }
    source_parse_work = sum(
        rational_parse_work((coordinate.num, coordinate.den))
        for coordinates in canonical_coordinates.values()
        for coordinate in coordinates
    )
    if source_parse_work > MAX_FRAMEWORK_COORDINATE_WORK:
        _reject(
            ("configuration", "points"),
            "coordinate_work_exceeds_bound",
            "framework coordinate normalization and edge differences exceed "
            f"the {MAX_FRAMEWORK_COORDINATE_WORK:,}-unit work bound",
        )
    coordinate_difference_work = 0
    for left_label, right_label in edge_axis:
        left = canonical_coordinates[left_label]
        right = canonical_coordinates[right_label]
        coordinate_difference_work += sum(
            difference_work(
                (left_coordinate.num, left_coordinate.den),
                (right_coordinate.num, right_coordinate.den),
            )
            for left_coordinate, right_coordinate in zip(left, right, strict=True)
        )
        if (
            source_parse_work + coordinate_difference_work
            > MAX_FRAMEWORK_COORDINATE_WORK
        ):
            _reject(
                ("configuration", "points"),
                "coordinate_work_exceeds_bound",
                "framework coordinate normalization and edge differences exceed "
                f"the {MAX_FRAMEWORK_COORDINATE_WORK:,}-unit work bound",
            )

    coordinates = {
        label: (values[0].as_fraction(), values[1].as_fraction())
        for label, values in canonical_coordinates.items()
    }
    return _FrameworkAdmission(
        vertex_axis=vertex_axis,
        edge_axis=edge_axis,
        coordinates=coordinates,
        source_parse_work=source_parse_work,
        coordinate_difference_work=coordinate_difference_work,
    )


def _matrix_scalar(value: Fraction) -> CanonicalRational:
    numerator = value.numerator
    denominator = value.denominator
    if abs(numerator) >= _MATRIX_SCALAR_LIMIT or denominator >= _MATRIX_SCALAR_LIMIT:
        _reject(
            ("configuration", "points"),
            "rigidity_matrix_scalar_exceeds_rank_bound",
            "a derived rigidity-matrix scalar exceeds matrix.rank.compute's "
            f"{MAX_INPUT_SCALAR_DIGITS}-digit input bound",
        )
    return CanonicalRational.from_fraction(value)


def _rigidity_matrix(admission: _FrameworkAdmission) -> SparseRationalMatrix:
    vertex_positions = {
        label: index for index, label in enumerate(admission.vertex_axis)
    }
    entries: list[SparseRationalMatrixEntry] = []
    for row, (left_label, right_label) in enumerate(admission.edge_axis):
        left = admission.coordinates[left_label]
        right = admission.coordinates[right_label]
        left_vertex = vertex_positions[left_label]
        right_vertex = vertex_positions[right_label]
        row_entries: list[SparseRationalMatrixEntry] = []
        for coordinate, (left_value, right_value) in enumerate(
            zip(left, right, strict=True)
        ):
            difference = left_value - right_value
            if difference == 0:
                continue
            row_entries.extend(
                (
                    SparseRationalMatrixEntry(
                        row=row,
                        column=2 * left_vertex + coordinate,
                        value=_matrix_scalar(difference),
                    ),
                    SparseRationalMatrixEntry(
                        row=row,
                        column=2 * right_vertex + coordinate,
                        value=_matrix_scalar(-difference),
                    ),
                )
            )
        entries.extend(sorted(row_entries, key=lambda entry: entry.column))
    return SparseRationalMatrix(
        row_count=len(admission.edge_axis),
        column_count=2 * len(admission.vertex_axis),
        entries=tuple(entries),
    )


def planar_rigidity_profile(
    configuration: PointConfiguration,
    graph: SimpleUndirectedGraph,
) -> PlanarRigidityProfile:
    """Return the exact planar rigidity matrix and rational rank."""

    admission = _admit_framework(configuration, graph)
    matrix = _rigidity_matrix(admission)
    try:
        matrix_rank = rank_result(matrix)
    except OperationDomainValidationError as exc:
        error = exc.errors()[0]
        _reject(
            ("configuration", "points"),
            "rigidity_matrix_rank_admission_failed",
            "the derived rigidity matrix is outside matrix.rank.compute "
            f"admission: {error['msg']}",
        )

    maximal_rank = 2 * len(admission.vertex_axis) - 3
    return PlanarRigidityProfile._from_kernel(
        configuration=configuration,
        graph=graph,
        vertex_axis=admission.vertex_axis,
        edge_axis=admission.edge_axis,
        matrix_rank=matrix_rank,
        maximal_infinitesimal_rigidity_rank=maximal_rank,
        is_infinitesimally_rigid=matrix_rank.rank == maximal_rank,
    )


def verify_planar_rigidity_profile(claim: PlanarRigidityProfile) -> bool:
    """Verify the canonical rigidity matrix and rank claim for a framework."""
    try:
        admission = _admit_framework(claim.configuration, claim.graph)
        matrix = _rigidity_matrix(admission)
        expected_rank = rank_result(matrix)
        maximal_rank = 2 * len(admission.vertex_axis) - 3
        return (
            claim.vertex_axis == admission.vertex_axis
            and claim.edge_axis == admission.edge_axis
            and claim.matrix_rank == expected_rank
            and claim.maximal_infinitesimal_rigidity_rank == maximal_rank
            and claim.is_infinitesimally_rigid == (expected_rank.rank == maximal_rank)
        )
    except (OperationDomainValidationError, ValueError, RuntimeError):
        return False


__all__ = ["planar_rigidity_profile", "verify_planar_rigidity_profile"]
