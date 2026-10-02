"""Signed diagonal homology uses a proved Smith gcd/lcm envelope."""

from itertools import pairwise
from math import gcd
from typing import Any

import pytest
from sympy import Matrix

from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.chain_complexes.operations import homology_groups
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    HomologyResult,
    IntegralHomologyGroupValue,
)


def _complex(entries: tuple[tuple[int, ...], ...]) -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        degree_min=0,
        degree_max=1,
        basis_sizes=(len(entries), len(entries[0])),
        differential_matrices=(entries,),
    )


def _assert_homology(
    entries: tuple[tuple[int, ...], ...], torsion: tuple[int, ...], rank: int
) -> None:
    result = homology_groups(_complex(entries))
    decoded = HomologyResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    groups = decoded.homology_groups
    assert all(isinstance(group, IntegralHomologyGroupValue) for group in groups)
    zero, one = groups
    assert isinstance(zero, IntegralHomologyGroupValue)
    assert isinstance(one, IntegralHomologyGroupValue)
    assert zero.torsion_invariant_factors == torsion
    assert zero.free_rank == len(entries) - rank
    assert one.torsion_invariant_factors == ()
    assert one.free_rank == len(entries[0]) - rank
    boundary = Matrix(entries)
    for generator in zero.torsion_generators:
        assert boundary * Matrix(generator.bounding_chain.coefficients) == int(
            generator.order
        ) * Matrix(generator.cycle.coefficients)
    for free_generator in one.free_generators:
        assert boundary * Matrix(free_generator.cycle.coefficients) == Matrix.zeros(
            len(entries), 1
        )
    for group in groups:
        assert isinstance(group, IntegralHomologyGroupValue)
        for certificate in (
            group.outgoing_smith_certificate,
            group.incoming_smith_certificate,
        ):

            def matrix(value: IntegerMatrix) -> Any:
                return Matrix(
                    value.row_count,
                    value.column_count,
                    [entry for row in value.entries for entry in row],
                )

            source = matrix(certificate.source)
            diagonal = matrix(certificate.diagonal)
            left = matrix(certificate.left_transformation)
            right = matrix(certificate.right_transformation)
            assert left * source * right == diagonal
            assert left.det() == certificate.left_determinant
            assert right.det() == certificate.right_determinant
            assert abs(left.det()) == abs(right.det()) == 1
            factors = certificate.invariant_factors
            assert all(value > 0 for value in factors)
            assert all(b % a == 0 for a, b in pairwise(factors))


@pytest.mark.parametrize("p", (15, 16, 31, 32, 127, 10**31))
def test_consecutive_diagonal_homology(p: int) -> None:
    _assert_homology(((p, 0), (0, p + 1)), (p * (p + 1),), 2)


@pytest.mark.parametrize("p,q", ((32, 64), (6, 10), (15, 16)))
@pytest.mark.parametrize("permuted", (False, True))
def test_signed_permuted_and_noncoprime_diagonals(
    p: int, q: int, permuted: bool
) -> None:
    entries = ((0, -q), (p, 0)) if permuted else ((-p, 0), (0, q))
    d = gcd(p, q)
    torsion = tuple(value for value in (d, p * q // d) if value > 1)
    _assert_homology(entries, torsion, 2)


@pytest.mark.parametrize(
    "entries,torsion,rank",
    [
        (((15, 0, 0), (0, 16, 0)), (240,), 2),
        (((0, 0, 0), (0, 15, 0), (0, 0, 16)), (240,), 2),
        (((6, 0, 0), (0, 10, 0), (0, 0, 15)), (30, 30), 3),
    ],
)
def test_rectangular_zero_and_three_axis_diagonals(
    entries: tuple[tuple[int, ...], ...], torsion: tuple[int, ...], rank: int
) -> None:
    _assert_homology(entries, torsion, rank)


def test_unproved_general_height_refusal_is_resource_admission() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    coefficient = 10**32 - 1
    source = _complex(
        ((coefficient, coefficient - 1), (coefficient - 2, coefficient - 3))
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        homology_groups(source)
    assert (
        error.value.errors()[0]["type"]
        == "chain_complex.integral_homology_height_budget_exceeded"
    )
