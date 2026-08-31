"""Exact lcm-lattice Betti-profile regressions for monomial ideals."""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.math.polynomials.ideals import (
    MonomialIdealBettiResult,
    monomial_ideal_graded_betti_table,
)
from jacobian.math.polynomials.ideals._models import MonomialIdealBettiRequest
from jacobian.math.polynomials.values import RationalPolynomialIdeal


def _monomial_ideal(
    variables: tuple[str, ...], *generators: tuple[int, ...]
) -> MonomialIdealBettiRequest:
    return MonomialIdealBettiRequest(
        ideal=RationalPolynomialIdeal.model_validate(
            {
                "variables": variables,
                "generators": [
                    {
                        "domain": "QQ",
                        "variables": list(variables),
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": list(generator),
                                }
                            ]
                        },
                    }
                    for generator in generators
                ],
            }
        )
    )


def _result(
    variables: tuple[str, ...], *generators: tuple[int, ...]
) -> MonomialIdealBettiResult:
    return monomial_ideal_graded_betti_table(
        _monomial_ideal(variables, *generators).ideal
    )


def _graded(result: MonomialIdealBettiResult) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (entry.homological_degree, entry.internal_degree, entry.value)
        for entry in result.graded_betti_numbers
    )


def test_two_quadrics_return_the_complete_lcm_homology_profile() -> None:
    result = _result(("x", "y"), (2, 0), (0, 2))

    assert _graded(result) == ((0, 2, 2), (1, 4, 1))
    assert result.regularity == 3
    assert result.has_linear_resolution is False
    assert tuple(entry.multidegree for entry in result.lcm_lattice_homology) == (
        (0, 2),
        (2, 0),
        (2, 2),
    )
    top = result.lcm_lattice_homology[-1]
    assert top.face_counts == (1, 2, 0)
    assert top.boundary_ranks == (1, 0)
    assert top.reduced_homology_dimensions == (0, 1)


def test_principal_monomial_ideal_has_only_its_generator_shift() -> None:
    result = _result(("x", "y"), (2, 1))

    assert _graded(result) == ((0, 3, 1),)
    assert result.regularity == 3
    assert result.has_linear_resolution is True
    assert result.lcm_lattice_homology[0].face_counts == (1, 0)
    assert result.lcm_lattice_homology[0].reduced_homology_dimensions == (1,)


def test_cover_ideal_obstruction_has_regularity_seven() -> None:
    result = _result(
        ("x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"),
        (1, 1, 1, 1, 1, 0, 1, 0),
        (1, 1, 1, 1, 0, 1, 0, 1),
    )

    assert _graded(result) == ((0, 6, 2), (1, 8, 1))
    assert result.regularity == 7
    assert result.has_linear_resolution is False


def test_alexander_dual_obstruction_has_a_degree_four_syzygy() -> None:
    result = _result(
        ("x1", "x2", "x3", "x4"),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
    )

    assert _graded(result) == ((0, 2, 2), (1, 4, 1))
    assert result.regularity == 3
    assert result.has_linear_resolution is False


def test_equigenerated_linear_ideal_is_detected_from_all_nonzero_shifts() -> None:
    result = _result(("x", "y"), (2, 0), (1, 1), (0, 2))

    assert _graded(result) == ((0, 2, 3), (1, 3, 2))
    assert result.regularity == 2
    assert result.has_linear_resolution is True


def test_eight_generator_boolean_lattice_boundary_is_complete() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    result = _result(
        variables,
        *(tuple(1 if row == column else 0 for column in range(8)) for row in range(8)),
    )

    assert _graded(result) == (
        (0, 1, 8),
        (1, 2, 28),
        (2, 3, 56),
        (3, 4, 70),
        (4, 5, 56),
        (5, 6, 28),
        (6, 7, 8),
        (7, 8, 1),
    )
    assert len(result.lcm_lattice_homology) == 255
    assert result.has_linear_resolution is True


def test_multigraded_entries_define_the_graded_table_and_regularity() -> None:
    result = _result(("x", "y", "z"), (2, 0, 0), (1, 1, 0), (0, 1, 2))
    totals: dict[tuple[int, int], int] = {}
    for entry in result.multigraded_betti_numbers:
        key = entry.homological_degree, sum(entry.multidegree)
        totals[key] = totals.get(key, 0) + entry.value

    assert _graded(result) == tuple(
        (homological_degree, internal_degree, value)
        for (homological_degree, internal_degree), value in sorted(totals.items())
    )
    assert result.regularity == max(
        entry.internal_degree - entry.homological_degree
        for entry in result.graded_betti_numbers
    )


def test_multigraded_table_has_the_taylor_resolution_euler_characteristic() -> None:
    result = _result(
        ("w", "x", "y", "z"),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
    )
    taylor_coefficients: dict[tuple[int, ...], int] = {}
    generators = [
        next(term.exponents for term in gen.polynomial.terms)
        for gen in result.ideal.generators
    ]
    for size in range(1, len(generators) + 1):
        for subset in combinations(generators, size):
            multidegree = tuple(map(max, zip(*subset, strict=True)))
            taylor_coefficients[multidegree] = taylor_coefficients.get(
                multidegree, 0
            ) + (-1) ** (size + 1)
    betti_coefficients: dict[tuple[int, ...], int] = {}
    for entry in result.multigraded_betti_numbers:
        betti_coefficients[entry.multidegree] = (
            betti_coefficients.get(entry.multidegree, 0)
            + (-1) ** entry.homological_degree * entry.value
        )

    assert betti_coefficients == {
        multidegree: value
        for multidegree, value in taylor_coefficients.items()
        if value
    }


def test_exact_result_round_trips_and_its_ideal_composes_unchanged() -> None:
    produced = _result(("x", "y"), (2, 0), (0, 2))
    decoded = MonomialIdealBettiResult.model_validate_json(produced.model_dump_json())
    consumed = monomial_ideal_graded_betti_table(decoded.ideal)

    assert decoded == produced
    assert consumed == produced

    # The ideal is a RationalPolynomialIdeal that composes with other operations.
    from jacobian.math.polynomials.values import RationalPolynomialIdeal

    assert isinstance(decoded.ideal, RationalPolynomialIdeal)


@pytest.mark.parametrize(
    ("generators", "message"),
    [
        (((2, 0), (1, 0)), "pairwise nondividing"),
        (((0, 2), (2, 0)), "descending lexicographic"),
        (((0, 0),), "unit ideal"),
    ],
)
def test_monomial_ideal_rejects_noncanonical_presentations(
    generators: tuple[tuple[int, int], ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _monomial_ideal(("x", "y"), *generators)


def test_generator_and_exponent_boundaries_reject_before_kernel_work() -> None:
    with pytest.raises(ValidationError, match="8-generator operation budget"):
        _monomial_ideal(
            tuple(f"x{index}" for index in range(8)),
            *tuple(
                tuple(index + 1 if slot == 0 else 0 for slot in range(8))
                for index in range(9)
            ),
        )
    with pytest.raises(ValidationError, match="single monomial term"):
        MonomialIdealBettiRequest(
            ideal=RationalPolynomialIdeal.model_validate(
                {
                    "variables": ("x",),
                    "generators": [
                        {
                            "domain": "QQ",
                            "variables": ["x"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [2],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [0],
                                    },
                                ]
                            },
                        },
                    ],
                }
            )
        )
