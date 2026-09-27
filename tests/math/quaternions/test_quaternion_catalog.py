"""Catalog admission contract for the rational unit-quaternion operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.quaternions import (
    RationalUnitQuaternion,
    conjugate_rational_unit_quaternion,
    inverse_rational_unit_quaternion,
    rational_unit_quaternion_scalar_part,
)
from jacobian.math.quaternions._tools import TOOLS

OPERATION_IDS = {tool.operation_id for tool in TOOLS}


def q(*coordinates: Fraction) -> RationalUnitQuaternion:
    return RationalUnitQuaternion(
        coordinates=tuple(
            CanonicalRational.from_fraction(value) for value in coordinates
        )
    )


def test_conjugation_and_inversion_publish_one_catalog_operation() -> None:
    published = {
        operation_id
        for operation_id in OPERATION_IDS
        if operation_id
        in {
            "quaternion.rational_unit.conjugate.compute",
            "quaternion.rational_unit.inverse.compute",
        }
    }
    assert len(published) == 1
    tool = next(tool for tool in TOOLS if tool.operation_id in published)
    vocabulary = " ".join(
        (tool.title, tool.description, *tool.tags, *tool.discovery_terms)
    ).casefold()
    # The omitted name stays reachable through discovery and the native alias.
    assert "inverse" in vocabulary or "conjugate" in vocabulary


def test_inverse_and_conjugate_agree_on_the_norm_one_domain() -> None:
    value = q(Fraction(3, 5), Fraction(4, 5), Fraction(0), Fraction(0))
    assert inverse_rational_unit_quaternion(
        value
    ) == conjugate_rational_unit_quaternion(value)
    assert tuple(
        coordinate.as_fraction()
        for coordinate in conjugate_rational_unit_quaternion(value).coordinates
    ) == (Fraction(3, 5), Fraction(-4, 5), Fraction(0), Fraction(0))


def test_scalar_part_projection_stays_out_of_the_catalog() -> None:
    assert "quaternion.rational_unit.scalar_part.compute" not in OPERATION_IDS
    value = q(Fraction(3, 5), Fraction(4, 5), Fraction(0), Fraction(0))
    assert rational_unit_quaternion_scalar_part(value).as_fraction() == Fraction(3, 5)
