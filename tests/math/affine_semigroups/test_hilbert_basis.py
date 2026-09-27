from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from math import gcd

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups import AffineConfiguration, hilbert_basis
from jacobian.math.affine_semigroups.semigroup import (
    MAX_HILBERT_BASIS_DETERMINANT,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineHilbertBasisRequest,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS


def _configuration(
    u: tuple[int, int], v: tuple[int, int], *, include_interior: bool = False
) -> AffineConfiguration:
    vectors = (u, v, (u[0] + v[0], u[1] + v[1])) if include_interior else (u, v)
    return AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=tuple(f"g{i}" for i in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(2)),
    )


def _cone_coordinates(
    p: tuple[int, int], u: tuple[int, int], v: tuple[int, int]
) -> tuple[int, int, int]:
    determinant = u[0] * v[1] - u[1] * v[0]
    return (
        determinant,
        p[0] * v[1] - p[1] * v[0],
        u[0] * p[1] - u[1] * p[0],
    )


def _direct_lattice_oracle(
    u: tuple[int, int], v: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Coordinate-box oracle with direct two-summand decomposability search."""
    determinant = u[0] * v[1] - u[1] * v[0]
    corners = ((0, 0), u, v, (u[0] + v[0], u[1] + v[1]))
    points: set[tuple[int, int]] = {u, v}
    for x in range(min(p[0] for p in corners), max(p[0] for p in corners) + 1):
        for y in range(min(p[1] for p in corners), max(p[1] for p in corners) + 1):
            point = (x, y)
            _, alpha_num, beta_num = _cone_coordinates(point, u, v)
            if (
                point != (0, 0)
                and 0 <= alpha_num < determinant
                and 0 <= beta_num < determinant
            ):
                points.add(point)

    indecomposable = []
    for point in points:
        if point in (u, v):
            indecomposable.append(point)
            continue
        _, alpha_num, beta_num = _cone_coordinates(point, u, v)
        alpha = Fraction(alpha_num, determinant)
        beta = Fraction(beta_num, determinant)
        q_corners = (
            (Fraction(0), Fraction(0)),
            (alpha * u[0], alpha * u[1]),
            (beta * v[0], beta * v[1]),
            (Fraction(point[0]), Fraction(point[1])),
        )
        qx_min = min(q[0] for q in q_corners).__floor__()
        qx_max = max(q[0] for q in q_corners).__ceil__()
        qy_min = min(q[1] for q in q_corners).__floor__()
        qy_max = max(q[1] for q in q_corners).__ceil__()
        decomposable = False
        for qx, qy in product(range(qx_min, qx_max + 1), range(qy_min, qy_max + 1)):
            q = (qx, qy)
            remainder = (point[0] - qx, point[1] - qy)
            if q == (0, 0) or remainder == (0, 0):
                continue
            _, q_alpha, q_beta = _cone_coordinates(q, u, v)
            _, r_alpha, r_beta = _cone_coordinates(remainder, u, v)
            if min(q_alpha, q_beta, r_alpha, r_beta) >= 0:
                decomposable = True
                break
        if not decomposable:
            indecomposable.append(point)
    return tuple(sorted(indecomposable))


@pytest.mark.parametrize(
    ("u", "v", "expected"),
    [
        ((1, 0), (0, 1), ((0, 1), (1, 0))),
        ((1, 0), (1, 3), ((1, 0), (1, 1), (1, 2), (1, 3))),
        ((1, 0), (3, 5), ((1, 0), (1, 1), (2, 3), (3, 5))),
        ((0, 1), (-3, 5), ((-3, 5), (-1, 2), (0, 1))),
    ],
)
def test_known_complete_hilbert_bases(
    u: tuple[int, int], v: tuple[int, int], expected: tuple[tuple[int, int], ...]
) -> None:
    result = hilbert_basis(_configuration(u, v, include_interior=True))
    assert result.basis == expected
    assert result.configuration.generator_labels == ("g0", "g1", "g2")


def test_small_pointed_cones_match_direct_lattice_decomposition_oracle() -> None:
    rays = tuple(
        (x, y)
        for x, y in product(range(-3, 4), repeat=2)
        if (x or y) and gcd(abs(x), abs(y)) == 1
    )
    for u in rays:
        for v in rays:
            determinant = u[0] * v[1] - u[1] * v[0]
            if not 0 < determinant <= 9:
                continue
            actual = hilbert_basis(_configuration(u, v)).basis
            assert actual == _direct_lattice_oracle(u, v), (u, v)


def test_full_admitted_determinant_boundary_and_resource_refusal() -> None:
    admitted = hilbert_basis(_configuration((1, 0), (1, MAX_HILBERT_BASIS_DETERMINANT)))
    assert len(admitted.basis) == MAX_HILBERT_BASIS_DETERMINANT + 1
    assert admitted.basis[0] == (1, 0)
    assert admitted.basis[-1] == (1, MAX_HILBERT_BASIS_DETERMINANT)
    with pytest.raises(OperationResourceAdmissionError):
        hilbert_basis(_configuration((1, 0), (1, MAX_HILBERT_BASIS_DETERMINANT + 1)))


@pytest.mark.parametrize(
    "vectors",
    [
        ((1, 0), (-1, 0)),  # nonpointed
        ((1, 0), (2, 0)),  # not full-dimensional
        ((0, 0), (1, 1)),  # zero generator
    ],
)
def test_nonpointed_or_degenerate_configurations_are_domain_errors(
    vectors: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    with pytest.raises(ValueError):
        hilbert_basis(_configuration(*vectors))


def test_tool_example_and_result_round_trip() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.hilbert_basis.compute"
    )
    request = AffineHilbertBasisRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    result = tool.run(request)
    decoded = tool.result_type.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.basis == ((1, 0), (1, 1), (1, 2), (1, 3))


def test_forged_configuration_does_not_bypass_admission() -> None:
    malformed = AffineConfiguration.model_construct(
        row_labels=("x", "y"),
        generator_labels=("a", "b"),
        entries=((1, 1),),
    )
    with pytest.raises(OperationDomainValidationError):
        hilbert_basis(malformed)
