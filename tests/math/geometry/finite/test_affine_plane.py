"""Tests for the prime-field affine plane constructor operation."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry.finite._models import (
    PrimeFieldAffinePlaneRequest,
)
from jacobian.math.geometry.finite._operations import (
    compute_prime_field_affine_plane,
)
from jacobian.math.geometry.finite._tools import TOOLS

# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def test_catalog_contains_affine_plane_operation() -> None:
    operation_ids = {tool.operation_id for tool in TOOLS}
    assert "finite_geometry.affine_plane.prime_field.construct" in operation_ids


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_q2_structure() -> None:
    """AG(2,2): 4 points, 6 lines, 3 parallel classes."""
    result = compute_prime_field_affine_plane(
        PrimeFieldAffinePlaneRequest(prime_order=2)
    )
    assert result.prime_order == 2
    assert len(result.incidence.points) == 4
    assert len(result.incidence.block_ids) == 6
    assert len(result.parallel_classes) == 3
    assert result.total_incidences == 12


def test_q3_structure() -> None:
    """AG(2,3): 9 points, 12 lines, 4 parallel classes."""
    result = compute_prime_field_affine_plane(
        PrimeFieldAffinePlaneRequest(prime_order=3)
    )
    assert len(result.incidence.points) == 9
    assert len(result.incidence.block_ids) == 12
    assert len(result.parallel_classes) == 4
    assert result.total_incidences == 36


def test_q5_structure() -> None:
    """AG(2,5): 25 points, 30 lines, 6 parallel classes, 150 incidences."""
    result = compute_prime_field_affine_plane(
        PrimeFieldAffinePlaneRequest(prime_order=5)
    )
    assert len(result.incidence.points) == 25
    assert len(result.incidence.block_ids) == 30
    assert len(result.parallel_classes) == 6
    assert result.total_incidences == 150


# ---------------------------------------------------------------------------
# Lexicographic point ordering
# ---------------------------------------------------------------------------


def test_q3_point_ordering() -> None:
    """Points are (x,y) in lexicographic order with index = x*q + y."""
    result = compute_prime_field_affine_plane(
        PrimeFieldAffinePlaneRequest(prime_order=3)
    )
    expected = [
        "0,0",
        "0,1",
        "0,2",
        "1,0",
        "1,1",
        "1,2",
        "2,0",
        "2,1",
        "2,2",
    ]
    assert list(result.incidence.points) == expected


# ---------------------------------------------------------------------------
# Incidence properties
# ---------------------------------------------------------------------------


def test_every_line_has_q_points() -> None:
    """Every line in AG(2,q) contains exactly q points."""
    for q in (2, 3, 5, 7):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        for block in result.incidence.blocks:
            assert len(block) == q, f"q={q}: line has {len(block)} points, expected {q}"


def test_every_point_on_q_plus_1_lines() -> None:
    """Every point in AG(2,q) lies on exactly q+1 lines."""
    for q in (2, 3, 5, 7):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        point_to_count: dict[str, int] = dict.fromkeys(result.incidence.points, 0)
        for block in result.incidence.blocks:
            for member in block:
                point_to_count[member] += 1
        for point, count in point_to_count.items():
            assert count == q + 1, (
                f"q={q}: point {point} is on {count} lines, expected {q + 1}"
            )


def test_every_pair_of_points_on_exactly_one_line() -> None:
    """Every pair of distinct points lies on exactly one common line."""
    from itertools import combinations

    for q in (2, 3, 5):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        points = list(result.incidence.points)
        for p1, p2 in combinations(points, 2):
            common_count = 0
            for block in result.incidence.blocks:
                if p1 in block and p2 in block:
                    common_count += 1
            assert common_count == 1, (
                f"q={q}: points {p1},{p2} on {common_count} lines, expected 1"
            )


def test_total_incidences() -> None:
    """Total incidences = q^2 * (q+1)."""
    for q in (2, 3, 5, 7):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        expected = q * q * (q + 1)
        assert result.total_incidences == expected
        actual = sum(len(block) for block in result.incidence.blocks)
        assert actual == expected


# ---------------------------------------------------------------------------
# Parallel classes
# ---------------------------------------------------------------------------


def test_parallel_classes_partition() -> None:
    """Parallel classes partition line IDs into q+1 disjoint classes."""
    for q in (2, 3, 5, 7):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        assert len(result.parallel_classes) == q + 1
        all_ids: list[int] = []
        for cls in result.parallel_classes:
            all_ids.extend(cls.line_ids)
        assert all_ids == list(range(q * (q + 1)))


def test_parallel_class_sizes() -> None:
    """Each parallel class contains exactly q lines."""
    for q in (2, 3, 5, 7):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        for cls in result.parallel_classes:
            assert len(cls.line_ids) == q


def test_parallel_classes_labels() -> None:
    """Slope classes have labels slope_0..slope_{q-1}, vertical class labeled."""
    result = compute_prime_field_affine_plane(
        PrimeFieldAffinePlaneRequest(prime_order=3)
    )
    labels = [cls.label for cls in result.parallel_classes]
    assert labels == ["slope_0", "slope_1", "slope_2", "vertical"]


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_q", [1, 4, 6, 9, 10, 15, 21, 100])
def test_rejects_composite_and_one(bad_q: int) -> None:
    with pytest.raises((ValidationError, ValueError)):
        compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=bad_q)
        )


def test_rejects_q_too_large() -> None:
    """q > MAX_AFFINE_PLANE_FIELD_ORDER is rejected."""
    # 11 is prime but exceeds the transport budget
    with pytest.raises(ValidationError):
        compute_prime_field_affine_plane(PrimeFieldAffinePlaneRequest(prime_order=11))


# ---------------------------------------------------------------------------
# Parallel lines within a class are disjoint
# ---------------------------------------------------------------------------


def test_parallel_lines_are_disjoint() -> None:
    """Lines in the same parallel class are pairwise disjoint."""
    for q in (2, 3, 5):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        blocks = list(result.incidence.blocks)
        for cls in result.parallel_classes:
            line_indices = list(cls.line_ids)
            for i in range(len(line_indices)):
                for j in range(i + 1, len(line_indices)):
                    bi = set(blocks[line_indices[i]])
                    bj = set(blocks[line_indices[j]])
                    assert bi.isdisjoint(bj), (
                        f"q={q}: parallel lines {line_indices[i]} and "
                        f"{line_indices[j]} share points"
                    )


# ---------------------------------------------------------------------------
# Two lines from different parallel classes intersect in exactly one point
# ---------------------------------------------------------------------------


def test_non_parallel_lines_intersect_in_one_point() -> None:
    """Lines from different parallel classes intersect in exactly one point."""
    for q in (2, 3, 5):
        result = compute_prime_field_affine_plane(
            PrimeFieldAffinePlaneRequest(prime_order=q)
        )
        blocks = list(result.incidence.blocks)
        classes = result.parallel_classes
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                for li in classes[i].line_ids:
                    for lj in classes[j].line_ids:
                        bi = set(blocks[li])
                        bj = set(blocks[lj])
                        assert len(bi & bj) == 1, (
                            f"q={q}: lines {li} and {lj} from different "
                            f"classes intersect in {len(bi & bj)} points"
                        )
