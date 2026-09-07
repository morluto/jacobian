"""Exact source certificates for the covering regressions #3194 and #3192."""

import json
from fractions import Fraction
from itertools import combinations, permutations
from random import Random

import pytest
from tests.support.rationals import rational_payload as q

from jacobian.math.optimization import general_linear_program, linear_program
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._models import StandardFormRationalLinearProgram


def standard_program(
    rows: list[list[int]], rhs: list[int], objective: list[int]
) -> StandardFormRationalLinearProgram:
    return StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [f"x{i}" for i in range(len(objective))],
                "objective": [q(v) for v in objective],
                "coefficients": [[q(v) for v in row] for row in rows],
                "rhs": [q(v) for v in rhs],
            }
        )
    )


def assert_standard_certificate(program: StandardFormRationalLinearProgram) -> str:
    result = linear_program(program)
    rows = [[v.as_fraction() for v in row] for row in program.coefficients]
    rhs = [v.as_fraction() for v in program.rhs]
    c = [v.as_fraction() for v in program.objective]
    if result.status == "INFEASIBLE":
        assert result.farkas_candidate is not None
        y = [v.as_fraction() for v in result.farkas_candidate]
        assert sum(a * b for a, b in zip(rhs, y, strict=True)) < 0
        assert all(
            sum(row[j] * v for row, v in zip(rows, y, strict=True)) >= 0
            for j in range(len(c))
        )
        assert result.primal_candidate is None
        return result.status
    assert result.primal_candidate is not None
    x = [v.as_fraction() for v in result.primal_candidate]
    assert min(x) >= 0
    assert [sum(a * b for a, b in zip(row, x, strict=True)) for row in rows] == rhs
    objective = sum(a * b for a, b in zip(c, x, strict=True))
    assert (
        result.primal_objective is not None
        and result.primal_objective.as_fraction() == objective
    )
    if result.status == "UNBOUNDED":
        assert result.recession_direction is not None
        ray = [v.as_fraction() for v in result.recession_direction]
        assert min(ray) >= 0
        assert all(
            sum(a * b for a, b in zip(row, ray, strict=True)) == 0 for row in rows
        )
        assert sum(a * b for a, b in zip(c, ray, strict=True)) < 0
    else:
        assert result.status == "OPTIMAL" and result.dual_candidate is not None
        y = [v.as_fraction() for v in result.dual_candidate]
        assert all(
            sum(row[j] * v for row, v in zip(rows, y, strict=True)) <= c[j]
            for j in range(len(c))
        )
        assert sum(a * b for a, b in zip(rhs, y, strict=True)) == objective
    return result.status


def cover_program(rows: list[list[int]]) -> GeneralFormRationalLinearProgram:
    return GeneralFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [
                    {"name": f"x{i}", "lower_bound": q(0)} for i in range(len(rows[0]))
                ],
                "objective": {
                    "sense": "MINIMIZE",
                    "coefficients": [q(1)] * len(rows[0]),
                },
                "constraints": [
                    {
                        "label": f"row{i}",
                        "coefficients": [q(a) for a in row],
                        "relation": "GE",
                        "rhs": q(1),
                    }
                    for i, row in enumerate(rows)
                ],
            }
        )
    )


def assert_cover_certificate(result: GeneralRationalLinearProgramResult) -> None:
    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert result.constraint_dual is not None
    assert result.lower_bound_dual is not None
    x = [v.as_fraction() for v in result.primal_candidate]
    y = [v.as_fraction() for v in result.constraint_dual]
    lower = [v.as_fraction() for v in result.lower_bound_dual]
    rows = [
        [v.as_fraction() for v in row.coefficients]
        for row in result.program.constraints
    ]
    assert all(v >= 0 for v in (*x, *y, *lower))
    assert all(sum(a * b for a, b in zip(row, x, strict=True)) >= 1 for row in rows)
    assert all(
        sum(row[j] * v for row, v in zip(rows, y, strict=True)) + lower[j] == 1
        for j in range(len(x))
    )
    assert sum(x) == sum(y) == 2
    assert (
        result.primal_objective is not None
        and result.primal_objective.as_fraction() == 2
    )
    assert result.dual_objective == result.primal_objective


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_cover_row_order_preserves_exact_optimum(order: tuple[int, ...]) -> None:
    rows = [[1, 1], [1, 0], [0, 1]]
    assert_cover_certificate(
        general_linear_program(cover_program([rows[i] for i in order]))
    )


def test_standard_cover_has_valid_objective_two_certificate() -> None:
    rows = [[-1, -1, 1, 0, 0], [-1, 0, 0, 1, 0], [0, -1, 0, 0, 1]]
    program = StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [f"x{i}" for i in range(5)],
                "objective": [q(1), q(1), q(0), q(0), q(0)],
                "coefficients": [[q(a) for a in row] for row in rows],
                "rhs": [q(-1)] * 3,
            }
        )
    )
    result = linear_program(program)
    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None and result.dual_candidate is not None
    x = [v.as_fraction() for v in result.primal_candidate]
    y = [v.as_fraction() for v in result.dual_candidate]
    assert min(x) >= 0
    assert [sum(a * b for a, b in zip(row, x, strict=True)) for row in rows] == [-1] * 3
    assert all(
        sum(row[j] * v for row, v in zip(rows, y, strict=True)) <= int(j < 2)
        for j in range(5)
    )
    assert x[0] + x[1] == -sum(y) == Fraction(2)


def test_pinned_sympy_matrix_lp_defect() -> None:
    """Notify backend upgrades when the documented 1.14.0 defect changes.

    This is a defect sentinel, not evidence for Jacobian's mathematical answer.
    The public regressions independently prove the objective-two certificates.
    """
    from sympy.solvers.simplex import linprog

    objective, point = linprog([1, 1], [[-1, -1], [-1, 0], [0, -1]], [-1, -1, -1])
    assert objective == 1 and point == [0, 1]
    assert point[0] < 1  # Violates the second input row x >= 1.


@pytest.mark.parametrize("unused", [False, True])
def test_pair_cover_original_and_trimmed_encodings(unused: bool) -> None:
    triples = [(0, 2, 3), (0, 2, 4), (0, 2, 5), (0, 3, 5), (1, 3, 5), (2, 3, 5)]
    pairs = (
        list(combinations(range(6), 2))
        if unused
        else sorted({p for t in triples for p in combinations(t, 2)})
    )
    rows = [[int(i in t and j in t) for i, j in pairs] for t in triples]
    # Independent primal: pairs 02 and 35. Independent dual: triples 023 and
    # 135 share no pair. Both have value two, proving the expected optimum.
    assert_cover_certificate(general_linear_program(cover_program(rows)))


@pytest.mark.parametrize(
    ("rows", "rhs", "objective", "status"),
    [
        ([], [], [1, 0], "OPTIMAL"),
        ([[0, 0]], [0], [1, -1], "UNBOUNDED"),
        ([[0, 0]], [1], [1, -1], "INFEASIBLE"),
        ([[1, 1], [2, 2], [0, 0]], [1, 2, 0], [1, 1], "OPTIMAL"),
        ([[1, 1], [2, 2]], [1, 3], [1, 1], "INFEASIBLE"),
        ([[1, 0], [2, 0]], [-1, -2], [0, -1], "INFEASIBLE"),
        ([[1, -1], [2, -2]], [1, 2], [-1, 0], "UNBOUNDED"),
        ([[1, 0], [2, 0]], [1, 2], [0, -1], "UNBOUNDED"),
        ([[1, 1, 0], [0, 1, 1]], [-1, 1], [-1, 0, -1], "INFEASIBLE"),
    ],
)
def test_rank_deficiency_and_negative_certificates(
    rows: list[list[int]], rhs: list[int], objective: list[int], status: str
) -> None:
    assert assert_standard_certificate(standard_program(rows, rhs, objective)) == status


def test_varied_exact_certificates_and_redundant_row_invariance() -> None:
    random = Random(3194)
    statuses = set()
    for _ in range(80):
        n, m = random.randint(1, 5), random.randint(1, 4)
        rows = [[random.randint(-2, 2) for _ in range(n)] for _ in range(m)]
        rhs = [random.randint(-2, 2) for _ in range(m)]
        objective = [random.randint(-2, 2) for _ in range(n)]
        status = assert_standard_certificate(standard_program(rows, rhs, objective))
        statuses.add(status)
        assert (
            assert_standard_certificate(
                standard_program(
                    [*rows[::-1], rows[0]], [*rhs[::-1], rhs[0]], objective
                )
            )
            == status
        )
    assert statuses == {"OPTIMAL", "INFEASIBLE", "UNBOUNDED"}


def test_invalid_backend_negative_candidates_cannot_cross_trusted_boundary() -> None:
    from jacobian.math.optimization.operations import (
        _certify_infeasible,
        _certify_point,
    )

    # The actual invalid Farkas candidate from #3194 has negative A^T y.
    p = standard_program(
        [[-1, -1, 1, 0, 0], [-1, 0, 0, 1, 0], [0, -1, 0, 0, 1]],
        [-1, -1, -1],
        [1, 1, 0, 0, 0],
    )
    with pytest.raises(RuntimeError, match="produced no mathematical result"):
        _certify_infeasible(p, (Fraction(0), Fraction(1), Fraction(0)), 100)
    # Nonnegative improving directions still need Ad=0; homogeneous improving
    # vectors still need nonnegativity; and a ray alone needs a feasible point.
    p = standard_program([[1, -1]], [1], [-1, 0])
    for point, ray in [
        ((1, 0), (1, 0)),
        ((1, 0), (1, -1)),
        ((0, 0), (1, 1)),
        ((1, 0), (0, 0)),
    ]:
        with pytest.raises(RuntimeError, match="produced no mathematical result"):
            _certify_point(
                p, tuple(map(Fraction, point)), (), tuple(map(Fraction, ray)), 100
            )


def test_dense_full_rank_infeasibility_near_work_limit() -> None:
    # Every six-column Vandermonde minor is nonsingular. Negative first RHS
    # makes all C(17,6)=12376 original bases infeasible, forcing phase I.
    # The complete conservative reservation is 49,909,832 < 50,000,000 updates.
    rows = [[(j + 1) ** i for j in range(17)] for i in range(6)]
    assert (
        assert_standard_certificate(standard_program(rows, [-1] * 6, [1] * 17))
        == "INFEASIBLE"
    )


def test_coupled_farkas_maps_upper_and_lower_signs() -> None:
    program = GeneralFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [
                    {"name": name, "lower_bound": q(0), "upper_bound": q(1)}
                    for name in ("x", "y")
                ],
                "objective": {"sense": "MINIMIZE", "coefficients": [q(-1), q(-1)]},
                "constraints": [
                    {
                        "label": "sum",
                        "coefficients": [q(1), q(1)],
                        "relation": "GE",
                        "rhs": q(3),
                    }
                ],
            }
        )
    )
    result = general_linear_program(program)
    assert result.status == "INFEASIBLE"
    assert result.farkas_constraints is not None
    assert (
        result.farkas_lower_bounds is not None
        and result.farkas_upper_bounds is not None
    )
    y = result.farkas_constraints[0].as_fraction()
    lower = [v.as_fraction() for v in result.farkas_lower_bounds]
    upper = [v.as_fraction() for v in result.farkas_upper_bounds]
    assert y <= 0 and all(v <= 0 for v in lower) and all(v >= 0 for v in upper)
    assert all(y + lo + hi == 0 for lo, hi in zip(lower, upper, strict=True))
    assert 3 * y + sum(upper) < 0
