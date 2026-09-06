"""Contract evidence for exact planar framework rigidity profiles."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.framework._bounds import (
    MAX_FRAMEWORK_COORDINATE_WORK,
    difference_work,
    rational_parse_work,
)
from jacobian.math.geometry.framework._models import (
    PlanarRigidityProfile,
    PlanarRigidityProfileRequest,
)
from jacobian.math.geometry.framework._tools import TOOLS
from jacobian.math.geometry.framework.operations import (
    _admit_framework,
    _rigidity_matrix,
    planar_rigidity_profile,
    verify_planar_rigidity_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices._operation_models import MatrixRankRequest
from jacobian.math.matrices.operations import rank_result
from jacobian.math.matrices.values import SparseRationalMatrix


def q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def configuration(
    points: tuple[tuple[str, int, int], ...],
) -> PointConfiguration:
    return PointConfiguration(
        points=tuple(
            LabelledRationalPoint(label=label, coordinates=(q(x), q(y)))
            for label, x, y in points
        )
    )


def graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def sparse_entries(
    matrix: SparseRationalMatrix,
) -> dict[tuple[int, int], Fraction]:
    return {
        (entry.row, entry.column): entry.value.as_fraction() for entry in matrix.entries
    }


def rigidity_matrix(profile: PlanarRigidityProfile) -> SparseRationalMatrix:
    matrix = profile.matrix_rank.matrix
    assert isinstance(matrix, SparseRationalMatrix)
    return matrix


def test_catalog_publishes_only_the_planar_rigidity_profile() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "geometry.framework.planar_rigidity_profile.compute"
    }


def test_catalog_request_parses_strict_canonical_json_arrays() -> None:
    payload = TOOLS[0].examples[0].input

    request = PlanarRigidityProfileRequest.model_validate_json(
        encode_strict_json(payload), strict=True
    )

    assert tuple(point.label for point in request.configuration.points) == (
        "a",
        "b",
        "c",
    )
    assert request.graph.edges == (("b", "c"), ("a", "c"), ("a", "b"))


def test_native_api_exports_the_reused_framework_values() -> None:
    import jacobian.math.geometry.framework as framework

    assert framework.__all__ == [
        "LabelledRationalPoint",
        "PlanarRigidityProfile",
        "PointConfiguration",
        "planar_rigidity_profile",
        "verify_planar_rigidity_profile",
    ]
    assert framework.LabelledRationalPoint is LabelledRationalPoint
    assert framework.PointConfiguration is PointConfiguration
    assert not hasattr(framework, "SimpleUndirectedGraph")
    assert framework.PlanarRigidityProfile is PlanarRigidityProfile
    assert framework.planar_rigidity_profile is planar_rigidity_profile


@pytest.mark.parametrize("point_count", (0, 1))
def test_framework_source_excludes_zero_and_one_vertex_configurations(
    point_count: int,
) -> None:
    points = tuple(
        LabelledRationalPoint(label=f"v{index}", coordinates=(q(index), q(0)))
        for index in range(point_count)
    )

    with pytest.raises(ValidationError, match="at least 2 items"):
        PointConfiguration(points=points)


def test_noncollinear_triangle_has_full_infinitesimal_rank() -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 0, 1)))
    source_graph = graph(("a", "b", "c"), (("b", "c"), ("a", "c"), ("a", "b")))

    result = planar_rigidity_profile(source, source_graph)

    assert result.configuration is source
    assert result.graph is source_graph
    assert result.vertex_axis == ("a", "b", "c")
    assert result.edge_axis == (("a", "b"), ("a", "c"), ("b", "c"))
    assert result.matrix_rank.rank == 3
    assert result.matrix_rank.pivot_columns == (0, 1, 2)
    assert result.maximal_infinitesimal_rigidity_rank == 3
    assert result.is_infinitesimally_rigid is True
    assert sparse_entries(rigidity_matrix(result)) == {
        (0, 0): Fraction(-1),
        (0, 2): Fraction(1),
        (1, 1): Fraction(-1),
        (1, 5): Fraction(1),
        (2, 2): Fraction(1),
        (2, 3): Fraction(-1),
        (2, 4): Fraction(-1),
        (2, 5): Fraction(1),
    }


@pytest.mark.parametrize(
    ("edges", "expected_rank", "expected_rigid"),
    (
        ((("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")), 4, False),
        (
            (
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "d"),
                ("a", "c"),
            ),
            5,
            True,
        ),
    ),
)
def test_square_cycle_and_diagonal_distinguish_infinitesimal_rank(
    edges: tuple[tuple[str, str], ...], expected_rank: int, expected_rigid: bool
) -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 1, 1), ("d", 0, 1)))

    result = planar_rigidity_profile(source, graph(("a", "b", "c", "d"), edges))

    assert result.matrix_rank.rank == expected_rank
    assert result.maximal_infinitesimal_rigidity_rank == 5
    assert result.is_infinitesimally_rigid is expected_rigid


def test_collinear_complete_triangle_fails_only_the_infinitesimal_criterion() -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 2, 0)))
    source_graph = graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))

    result = planar_rigidity_profile(source, source_graph)

    assert result.matrix_rank.rank == 2
    assert result.maximal_infinitesimal_rigidity_rank == 3
    assert result.is_infinitesimally_rigid is False


def test_edge_tuple_permutation_has_one_deterministic_derived_axis() -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 0, 1)))
    first = graph(("a", "b", "c"), (("b", "c"), ("a", "b"), ("a", "c")))
    second = graph(("c", "a", "b"), (("a", "c"), ("b", "c"), ("a", "b")))

    left = planar_rigidity_profile(source, first)
    right = planar_rigidity_profile(source, second)

    assert left.edge_axis == right.edge_axis
    assert left.matrix_rank.matrix == right.matrix_rank.matrix
    assert left.matrix_rank.rank == right.matrix_rank.rank
    assert left.matrix_rank.pivot_columns == right.matrix_rank.pivot_columns


def test_empty_edge_graph_retains_zero_row_and_coordinate_axes() -> None:
    source = configuration((("right", 1, 0), ("left", 0, 0)))

    result = planar_rigidity_profile(source, graph(("left", "right"), ()))

    matrix = rigidity_matrix(result)
    assert matrix.row_count == 0
    assert matrix.column_count == 4
    assert matrix.entries == ()
    assert result.vertex_axis == ("right", "left")
    assert result.edge_axis == ()
    assert result.matrix_rank.rank == 0
    assert result.matrix_rank.pivot_columns == ()
    assert result.is_infinitesimally_rigid is False


def test_repeated_coordinates_produce_an_exact_zero_edge_row() -> None:
    source = configuration((("a", 0, 0), ("b", 0, 0), ("c", 1, 0)))

    result = planar_rigidity_profile(
        source,
        graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))),
    )

    matrix = rigidity_matrix(result)
    assert matrix.row_count == 3
    assert not any(entry.row == 0 for entry in matrix.entries)
    assert result.matrix_rank.rank == 2
    assert result.is_infinitesimally_rigid is False


@pytest.mark.parametrize(
    "bad_graph",
    (
        graph(("a", "b"), (("a", "b"),)),
        graph(("a", "b", "c", "d"), (("a", "b"),)),
        graph(("a", "b", "d"), (("a", "b"),)),
    ),
)
def test_graph_labels_must_equal_configuration_labels_before_execution(
    bad_graph: SimpleUndirectedGraph,
) -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 0, 1)))

    with pytest.raises(ValidationError, match="point-label set"):
        PlanarRigidityProfileRequest(configuration=source, graph=bad_graph)
    with pytest.raises(OperationDomainValidationError, match="point-label set"):
        planar_rigidity_profile(source, bad_graph)


def test_configuration_must_be_planar() -> None:
    source = PointConfiguration(
        points=(
            LabelledRationalPoint(label="a", coordinates=(q(0), q(0), q(0))),
            LabelledRationalPoint(label="b", coordinates=(q(1), q(0), q(0))),
        )
    )
    source_graph = graph(("a", "b"), (("a", "b"),))

    with pytest.raises(ValidationError, match="exactly two coordinates"):
        PlanarRigidityProfileRequest(configuration=source, graph=source_graph)
    with pytest.raises(OperationDomainValidationError, match="exactly two coordinates"):
        planar_rigidity_profile(source, source_graph)


def test_derived_matrix_composes_unchanged_with_matrix_rank_compute() -> None:
    source = configuration((("a", 0, 0), ("b", 2, 0), ("c", 2, 1), ("d", 0, 1)))
    source_graph = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")),
    )

    profile = planar_rigidity_profile(source, source_graph)
    serialized_matrix = profile.matrix_rank.matrix.model_dump(mode="json")
    request = MatrixRankRequest.model_validate({"matrix": serialized_matrix})
    composed = rank_result(request.matrix)

    assert request.matrix == profile.matrix_rank.matrix
    assert composed == profile.matrix_rank


def test_profile_round_trip_is_structural_and_source_bound() -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 0, 1)))
    result = planar_rigidity_profile(
        source,
        graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))),
    )

    assert (
        PlanarRigidityProfile.model_validate_json(result.model_dump_json(), strict=True)
        == result
    )
    forged = result.model_dump(mode="json")
    forged["edge_axis"] = [["a", "c"], ["a", "b"], ["b", "c"]]
    with pytest.raises(ValidationError, match="lexicographically sorted"):
        PlanarRigidityProfile.model_validate(forged)


def test_serialized_profile_verifier_rejects_forged_matrix_claim() -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 0, 1)))
    source_graph = graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    result = planar_rigidity_profile(source, source_graph)
    decoded = PlanarRigidityProfile.model_validate_json(
        result.model_dump_json(), strict=True
    )
    assert verify_planar_rigidity_profile(decoded)

    payload = json.loads(result.model_dump_json())
    payload["matrix_rank"]["matrix"]["entries"][0]["value"]["num"] = "7"
    forged = PlanarRigidityProfile.model_validate_json(json.dumps(payload), strict=True)
    assert not verify_planar_rigidity_profile(forged)


def test_profile_contains_sources_axes_and_one_nested_matrix() -> None:
    source = configuration((("a", 0, 0), ("b", 1, 0), ("c", 0, 1)))
    source_graph = graph(("a", "b", "c"), (("b", "c"), ("a", "c"), ("a", "b")))
    admission = _admit_framework(source, source_graph)
    matrix = _rigidity_matrix(admission)

    result = planar_rigidity_profile(source, source_graph)
    payload = result.model_dump(mode="json")

    assert set(payload) == {
        "configuration",
        "graph",
        "vertex_axis",
        "edge_axis",
        "matrix_rank",
        "maximal_infinitesimal_rigidity_rank",
        "is_infinitesimally_rigid",
    }
    assert payload["matrix_rank"]["matrix"] == matrix.model_dump(mode="json")


def test_admission_derives_the_maximal_framework_matrix_shape() -> None:
    source = configuration(
        tuple((f"v{index:02d}", index, index * index) for index in range(64))
    )
    labels = tuple(point.label for point in source.points)
    edges = tuple(
        (labels[left], labels[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    )
    source_graph = graph(labels, tuple(reversed(edges)))
    admission = _admit_framework(source, source_graph)
    matrix = _rigidity_matrix(admission)

    assert matrix.row_count == 2_016
    assert matrix.column_count == 128
    assert len(matrix.entries) == 8_064
    assert admission.edge_axis == edges


def test_rigidity_rows_replay_the_squared_length_differentials() -> None:
    source = PointConfiguration(
        points=(
            LabelledRationalPoint(label="v", coordinates=(q(1, 3), q(-2, 5))),
            LabelledRationalPoint(label="u", coordinates=(q(-4, 7), q(3, 2))),
            LabelledRationalPoint(label="w", coordinates=(q(5, 6), q(1, 4))),
        )
    )
    source_graph = graph(("u", "v", "w"), (("v", "w"), ("u", "w"), ("u", "v")))

    result = planar_rigidity_profile(source, source_graph)
    matrix = rigidity_matrix(result)
    actual = sparse_entries(matrix)
    point_coordinates = {
        point.label: tuple(value.as_fraction() for value in point.coordinates)
        for point in source.points
    }
    positions = {label: index for index, label in enumerate(result.vertex_axis)}
    expected: dict[tuple[int, int], Fraction] = {}
    for row, (left, right) in enumerate(result.edge_axis):
        for coordinate in range(2):
            difference = (
                point_coordinates[left][coordinate]
                - point_coordinates[right][coordinate]
            )
            if difference:
                expected[(row, 2 * positions[left] + coordinate)] = difference
                expected[(row, 2 * positions[right] + coordinate)] = -difference

    assert matrix.row_count == len(result.edge_axis)
    assert matrix.column_count == 2 * len(result.vertex_axis)
    assert actual == expected


def test_coordinate_scalar_boundary_is_owned_before_rank_backend() -> None:
    accepted_value = 10**255
    accepted = configuration((("a", 0, 0), ("b", accepted_value, 0)))
    source_graph = graph(("a", "b"), (("a", "b"),))

    result = planar_rigidity_profile(accepted, source_graph)

    assert result.matrix_rank.rank == 1
    rejected_value = 10**256
    rejected = configuration((("a", 0, 0), ("b", rejected_value, 0)))
    with pytest.raises(
        OperationDomainValidationError,
        match="256-digit input bound",
    ):
        planar_rigidity_profile(rejected, source_graph)


def test_coordinate_work_admission_charges_every_derived_difference() -> None:
    height = 50
    value = 10 ** (height - 1)
    source = configuration(
        tuple((f"v{index:02d}", index * value, index) for index in range(32))
    )
    labels = tuple(point.label for point in source.points)
    edges = tuple(
        (labels[left], labels[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    )
    source_graph = graph(labels, edges)

    admission = _admit_framework(source, source_graph)

    canonical_coordinates = {point.label: point.coordinates for point in source.points}
    executed = {
        "source_parse": sum(
            rational_parse_work((coordinate.num, coordinate.den))
            for point in source.points
            for coordinate in point.coordinates
        ),
        "coordinate_difference": sum(
            difference_work(
                (
                    canonical_coordinates[left][coordinate].num,
                    canonical_coordinates[left][coordinate].den,
                ),
                (
                    canonical_coordinates[right][coordinate].num,
                    canonical_coordinates[right][coordinate].den,
                ),
            )
            for left, right in edges
            for coordinate in range(2)
        ),
    }
    charged = {
        "source_parse": admission.source_parse_work,
        "coordinate_difference": admission.coordinate_difference_work,
    }
    assert admission.source_parse_work + admission.coordinate_difference_work <= (
        MAX_FRAMEWORK_COORDINATE_WORK
    )
    assert charged == executed
    assert_charged_work_parity(charged=charged, executed=executed)


def test_coordinate_work_accepts_and_rejects_the_exact_edge_boundary() -> None:
    base = 10**39
    source = configuration(
        tuple((f"v{index:02d}", base + index, base + 2 * index) for index in range(64))
    )
    labels = tuple(point.label for point in source.points)
    all_edges = tuple(
        (labels[left], labels[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    )
    source_parse_work = sum(
        rational_parse_work((coordinate.num, coordinate.den))
        for point in source.points
        for coordinate in point.coordinates
    )
    first_left, first_right = all_edges[0]
    coordinates = {point.label: point.coordinates for point in source.points}
    edge_work = sum(
        difference_work(
            (coordinates[first_left][axis].num, coordinates[first_left][axis].den),
            (
                coordinates[first_right][axis].num,
                coordinates[first_right][axis].den,
            ),
        )
        for axis in range(2)
    )
    accepted_edge_count = (
        MAX_FRAMEWORK_COORDINATE_WORK - source_parse_work
    ) // edge_work
    accepted_graph = graph(labels, all_edges[:accepted_edge_count])
    rejected_graph = graph(labels, all_edges[: accepted_edge_count + 1])

    admission = _admit_framework(source, accepted_graph)

    assert admission.source_parse_work == source_parse_work
    assert admission.coordinate_difference_work == accepted_edge_count * edge_work
    assert source_parse_work + accepted_edge_count * edge_work <= (
        MAX_FRAMEWORK_COORDINATE_WORK
    )
    assert source_parse_work + (accepted_edge_count + 1) * edge_work > (
        MAX_FRAMEWORK_COORDINATE_WORK
    )

    with pytest.raises(OperationDomainValidationError, match="work bound"):
        planar_rigidity_profile(source, rejected_graph)

    with pytest.raises(ValidationError, match="work bound"):
        PlanarRigidityProfileRequest.model_validate(
            {
                "configuration": source.model_dump(mode="json"),
                "graph": rejected_graph.model_dump(mode="json"),
            }
        )


def test_edgeless_native_call_still_enforces_source_parse_work() -> None:
    large_coordinate = CanonicalRational(
        num="1" + "0" * (MAX_CANONICAL_RATIONAL_DIGITS - 1),
        den="9" * MAX_CANONICAL_RATIONAL_DIGITS,
    )
    source = PointConfiguration(
        points=(
            LabelledRationalPoint(
                label="a", coordinates=(large_coordinate, large_coordinate)
            ),
            LabelledRationalPoint(
                label="b", coordinates=(large_coordinate, large_coordinate)
            ),
        )
    )
    source_work = sum(
        rational_parse_work((coordinate.num, coordinate.den))
        for point in source.points
        for coordinate in point.coordinates
    )

    assert source_work > MAX_FRAMEWORK_COORDINATE_WORK
    with pytest.raises(OperationDomainValidationError, match="work bound"):
        planar_rigidity_profile(source, graph(("a", "b"), ()))
