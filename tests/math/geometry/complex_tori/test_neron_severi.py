"""Exact Neron-Severi lattices of lattice-presented complex tori."""

import pytest
from tests.math.geometry.complex_tori._support import (
    index_six_alternating_form,
    quartic_index_six_torus,
    quartic_rank_one_torus,
    quartic_rank_zero_torus,
    rational,
    standard_alternating_form,
)

from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.complex_tori import (
    LatticeComplexStructure,
    compute_neron_severi_lattice,
    compute_riemann_form_profile,
)
from jacobian.math.geometry.complex_tori._models import RiemannFormProfileRequest
from jacobian.math.lattices.invariant_forms import InvariantBilinearFormLattice
from jacobian.math.matrices.values import RationalMatrix


def test_quartic_complex_torus_has_exact_rank_zero_neron_severi_lattice() -> None:
    torus = quartic_rank_zero_torus()

    result = compute_neron_severi_lattice(torus)

    assert result.kind == "ALTERNATING"
    assert result.coefficient_dimension == 6
    assert result.constraint_rank == 6
    assert result.rank == 0
    assert result.basis_forms == ()
    assert result.action.generators[0].matrix == torus.complex_structure


def test_quartic_complex_torus_has_primitive_rank_one_neron_severi_lattice() -> None:
    torus = quartic_rank_one_torus()

    result = compute_neron_severi_lattice(torus)

    assert result.constraint_rank == 5
    assert result.rank == 1
    assert result.basis_forms == (standard_alternating_form(torus),)

    replayed = InvariantBilinearFormLattice.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")),
        strict=True,
    )
    consumer_request = RiemannFormProfileRequest.model_validate_json(
        encode_strict_json(
            {
                "torus": torus.model_dump(mode="json"),
                "form": replayed.basis_forms[0].model_dump(mode="json"),
            }
        ),
        strict=True,
    )
    profile = compute_riemann_form_profile(
        consumer_request.torus,
        consumer_request.form,
    )
    assert replayed == result
    assert profile.outcome.status == "HODGE_TYPE_11"


def test_index_six_eta_is_the_primitive_hodge_generator() -> None:
    torus = quartic_index_six_torus()

    result = compute_neron_severi_lattice(torus)

    assert result.rank == 1
    assert result.basis_forms == (index_six_alternating_form(torus),)


@pytest.mark.parametrize("complex_dimension", (1, 2, 3))
def test_rational_complex_structure_has_neron_severi_rank_g_squared(
    complex_dimension: int,
) -> None:
    zero = rational(0)
    one = rational(1)
    negative_one = rational(-1)
    dimension = 2 * complex_dimension
    matrix = RationalMatrix(
        entries=tuple(
            tuple(
                one
                if column == row + 1 and row % 2 == 0
                else negative_one
                if column == row - 1 and row % 2 == 1
                else zero
                for column in range(dimension)
            )
            for row in range(dimension)
        )
    )
    torus = LatticeComplexStructure(
        coordinate_axis=tuple(f"e{index + 1}" for index in range(dimension)),
        complex_structure=matrix,
    )

    result = compute_neron_severi_lattice(torus)

    assert result.rank == complex_dimension**2
    assert result.constraint_rank == result.coefficient_dimension - result.rank
