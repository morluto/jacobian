from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.pinned_distance.operations import (
    compute_pinned_distance_support_profile,
)


def _pt(label, coords):
    return LabelledRationalPoint(
        label=label,
        coordinates=tuple(CanonicalRational.from_fraction(Fraction(c)) for c in coords),
    )


def test_unit_square() -> None:
    """Unit square: each point has 2 distance classes (distance 1 and sqrt(2))."""
    config = PointConfiguration(
        points=(
            _pt("a", (0, 0)),
            _pt("b", (1, 0)),
            _pt("c", (0, 1)),
            _pt("d", (1, 1)),
        )
    )
    result = compute_pinned_distance_support_profile(config)
    assert len(result.entries) == 4
    entry_a = next(e for e in result.entries if e.source_label == "a")
    assert len(entry_a.distance_classes) == 2
    assert entry_a.distance_classes[0].squared_distance.as_fraction() == Fraction(1)
    assert entry_a.distance_classes[1].squared_distance.as_fraction() == Fraction(2)


def test_collinear_points() -> None:
    """Three collinear points at 0, 1, 3."""
    config = PointConfiguration(
        points=(
            _pt("a", (0,)),
            _pt("b", (1,)),
            _pt("c", (3,)),
        )
    )
    result = compute_pinned_distance_support_profile(config)
    entry_a = next(e for e in result.entries if e.source_label == "a")
    dists = {
        dc.squared_distance.as_fraction(): set(dc.target_labels)
        for dc in entry_a.distance_classes
    }
    assert dists[Fraction(1)] == {"b"}
    assert dists[Fraction(9)] == {"c"}


def test_directed_pair_count() -> None:
    """Each unordered pair occurs twice (once from each endpoint)."""
    config = PointConfiguration(
        points=(
            _pt("a", (0, 0)),
            _pt("b", (1, 0)),
            _pt("c", (2, 0)),
        )
    )
    result = compute_pinned_distance_support_profile(config)
    total = sum(
        len(dc.target_labels) for e in result.entries for dc in e.distance_classes
    )
    assert total == 6  # 3 points * 2 others = 6 directed pairs


def test_replay_squared_distance() -> None:
    """Each target's squared distance matches the computed value."""
    config = PointConfiguration(
        points=(
            _pt("a", (0, 0)),
            _pt("b", (1, 2)),
            _pt("c", (3, 1)),
        )
    )
    result = compute_pinned_distance_support_profile(config)
    for entry in result.entries:
        for dc in entry.distance_classes:
            for target in dc.target_labels:
                target_pt = next(p for p in config.points if p.label == target)
                source_pt = next(
                    p for p in config.points if p.label == entry.source_label
                )
                sq_dist = sum(
                    (a.as_fraction() - b.as_fraction()) ** 2
                    for a, b in zip(
                        source_pt.coordinates, target_pt.coordinates, strict=True
                    )
                )
                assert sq_dist == dc.squared_distance.as_fraction()


def test_sorted_by_distance() -> None:
    """Distance classes are sorted by increasing distance."""
    config = PointConfiguration(
        points=(
            _pt("a", (0, 0)),
            _pt("b", (1, 0)),
            _pt("c", (5, 0)),
        )
    )
    result = compute_pinned_distance_support_profile(config)
    for entry in result.entries:
        dists = [dc.squared_distance.as_fraction() for dc in entry.distance_classes]
        assert dists == sorted(dists)


def test_result_preserves_source() -> None:
    config = PointConfiguration(
        points=(_pt("a", (0,)), _pt("b", (1,))),
    )
    result = compute_pinned_distance_support_profile(config)
    assert result.configuration == config


def test_result_sensitive_admission_accepts_large_but_representable_coordinates() -> (
    None
):
    """A small profile can use coordinates beyond the old fixed cap."""
    huge = CanonicalRational.from_fraction(Fraction(10**999))
    config = PointConfiguration(
        points=(
            LabelledRationalPoint(label="a", coordinates=(huge,)),
            _pt("b", (0,)),
        )
    )
    result = compute_pinned_distance_support_profile(config)
    assert (
        result.entries[0].distance_classes[0].squared_distance.as_fraction() == 10**1998
    )


def test_native_admission_rejects_distance_height_before_squaring() -> None:
    """Squared coordinates beyond the canonical height are rejected early."""
    huge = CanonicalRational.from_fraction(Fraction(10**16384))
    config = PointConfiguration(
        points=(
            LabelledRationalPoint(label="a", coordinates=(huge,)),
            _pt("b", (0,)),
        )
    )
    with pytest.raises(OperationDomainValidationError, match="squared-distance"):
        compute_pinned_distance_support_profile(config)


def test_overheight_denominator_is_rejected_at_the_first_pair() -> None:
    denominator = 10**16384 + 1
    tiny = CanonicalRational.from_fraction(Fraction(1, denominator))
    config = PointConfiguration(
        points=(
            _pt("origin", (0,)),
            LabelledRationalPoint(label="tiny", coordinates=(tiny,)),
        )
    )

    with pytest.raises(OperationDomainValidationError, match="squared-distance"):
        compute_pinned_distance_support_profile(config)


def test_coprime_coordinate_denominators_are_bounded_before_summing() -> None:
    first = CanonicalRational.from_fraction(Fraction(1, 2**30_000))
    second = CanonicalRational.from_fraction(Fraction(1, 3**18_800))
    config = PointConfiguration(
        points=(
            _pt("origin", (0, 0)),
            LabelledRationalPoint(
                label="coprime",
                coordinates=(first, second),
            ),
        )
    )

    with pytest.raises(OperationDomainValidationError, match="squared-distance"):
        compute_pinned_distance_support_profile(config)


def test_squared_coordinate_terms_can_cancel_to_a_small_distance() -> None:
    magnitude = 10**10_000
    denominator = magnitude**2 + 1
    config = PointConfiguration(
        points=(
            _pt("origin", (0, 0)),
            LabelledRationalPoint(
                label="cancelled",
                coordinates=(
                    CanonicalRational.from_fraction(
                        Fraction(magnitude**2 - 1, denominator)
                    ),
                    CanonicalRational.from_fraction(
                        Fraction(2 * magnitude, denominator)
                    ),
                ),
            ),
        )
    )

    result = compute_pinned_distance_support_profile(config)

    assert result.entries[0].distance_classes[0].squared_distance.as_fraction() == 1


def test_shared_denominators_are_admitted_from_exact_distances() -> None:
    denominator = 10**1000 + 1
    coordinate = CanonicalRational.from_fraction(Fraction(1, denominator))
    config = PointConfiguration(
        points=(
            _pt("origin", (0,) * 20),
            LabelledRationalPoint(label="diagonal", coordinates=(coordinate,) * 20),
        )
    )

    result = compute_pinned_distance_support_profile(config)

    assert result.entries[0].distance_classes[0].squared_distance.as_fraction() == (
        Fraction(20, denominator**2)
    )


def test_long_target_labels_are_counted_once_per_entry() -> None:
    labels = [chr(0x10000 + index) * 64 for index in range(64)]
    config = PointConfiguration(
        points=tuple(_pt(label, (index,)) for index, label in enumerate(labels))
    )

    result = compute_pinned_distance_support_profile(config)

    assert len(result.entries) == 64


def test_oversized_source_is_rejected_at_operation_boundary() -> None:
    huge = CanonicalRational.from_fraction(Fraction(10**32767))
    config = PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=f"p{index}",
                coordinates=(huge,) * 20,
            )
            for index in range(17)
        )
    )

    with pytest.raises(OperationDomainValidationError, match="JSON exceeds"):
        compute_pinned_distance_support_profile(config)


def test_aggregate_profile_size_is_rejected_before_result_construction() -> None:
    scale = 10**15999
    config = PointConfiguration(
        points=tuple(_pt(f"p{index}", (index * scale,)) for index in range(24))
    )

    with pytest.raises(
        OperationDomainValidationError, match="complete distance profile"
    ):
        compute_pinned_distance_support_profile(config)
