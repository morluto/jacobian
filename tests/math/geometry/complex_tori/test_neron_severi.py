"""Exact Neron-Severi lattices of lattice-presented complex tori."""

import pytest
from sympy import QQ, Matrix
from sympy.polys.matrices import DomainMatrix
from tests.math.geometry.complex_tori._support import (
    index_six_alternating_form,
    nonmonic_quadratic_torus,
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
    verify_neron_severi_lattice,
)
from jacobian.math.geometry.complex_tori._models import RiemannFormProfileRequest
from jacobian.math.geometry.complex_tori._tools import (
    NERON_SEVERI_LATTICE_OPERATION,
)
from jacobian.math.lattices._lattice_ops import saturate_lattice
from jacobian.math.lattices.invariant_forms import (
    IntegralBilinearForm,
    InvariantBilinearFormLattice,
)
from jacobian.math.matrices._number_field import (
    domain_matrix_from_embedded,
    embedded_matrix_from_domain,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    IntegerMatrix,
    RationalMatrix,
)


def test_quartic_complex_torus_has_exact_rank_zero_neron_severi_lattice() -> None:
    torus = quartic_rank_zero_torus()

    request = NERON_SEVERI_LATTICE_OPERATION.request_type.model_validate_json(
        encode_strict_json({"torus": torus.model_dump(mode="json")}),
        strict=True,
    )
    result = NERON_SEVERI_LATTICE_OPERATION.run(request)

    assert result.kind == "ALTERNATING"
    assert result.coefficient_dimension == 6
    assert result.constraint_rank == 6
    assert result.rank == 0
    assert result.basis_forms == ()
    assert result.action.generators[0].matrix == torus.complex_structure
    assert verify_neron_severi_lattice(result) is True

    forged = result.model_dump(mode="json")
    forged["action"]["generators"][0]["matrix"]["entries"][0][0][
        "coefficients_ascending"
    ][0] = {"num": "1", "den": "1"}
    forged_claim = InvariantBilinearFormLattice.model_validate(forged)
    assert verify_neron_severi_lattice(forged_claim) is False


def test_quartic_complex_torus_has_primitive_rank_one_neron_severi_lattice() -> None:
    torus = quartic_rank_one_torus()

    result = compute_neron_severi_lattice(torus)

    assert result.constraint_rank == 5
    assert result.rank == 1
    assert result.basis_forms == (standard_alternating_form(torus),)

    round_tripped = InvariantBilinearFormLattice.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")),
        strict=True,
    )
    consumer_request = RiemannFormProfileRequest.model_validate_json(
        encode_strict_json(
            {
                "torus": torus.model_dump(mode="json"),
                "form": round_tripped.basis_forms[0].model_dump(mode="json"),
            }
        ),
        strict=True,
    )
    profile = compute_riemann_form_profile(
        consumer_request.torus,
        consumer_request.form,
    )
    assert round_tripped == result
    assert profile.outcome.status == "HODGE_NON_POSITIVE"


def test_index_six_eta_is_the_primitive_hodge_generator() -> None:
    torus = quartic_index_six_torus()

    result = compute_neron_severi_lattice(torus)

    assert result.rank == 1
    assert result.basis_forms == (index_six_alternating_form(torus),)


def test_nonmonic_public_torus_composes_neron_severi_with_riemann_profile() -> None:
    torus = nonmonic_quadratic_torus()

    lattice = compute_neron_severi_lattice(torus)
    positive_form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", "-1"), ("1", "0"))),
    )
    profile = compute_riemann_form_profile(torus, positive_form)

    assert lattice.rank == 1
    assert lattice.action.generators[0].matrix == torus.complex_structure
    assert profile.outcome.status == "RIEMANN_FORM"
    assert profile.outcome.polarization_type == ("1",)
    associated = profile.outcome.associated_form_inertia.matrix
    complex_structure = torus.complex_structure
    assert isinstance(associated, EmbeddedRealSimpleNumberFieldMatrix)
    assert isinstance(complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
    assert associated.embedding == complex_structure.embedding
    assert associated.embedding.presentation.coefficients_descending == (
        "2",
        "0",
        "-1",
    )
    assert associated.entries[0][0].coefficients_ascending == (
        rational(0),
        rational(2),
    )
    assert associated.entries[1][1].coefficients_ascending == (
        rational(0),
        rational(1),
    )


def test_nonmonic_algebraic_profiles_are_invariant_under_gl2z_coordinates() -> None:
    source_torus = nonmonic_quadratic_torus()
    source_structure = source_torus.complex_structure
    assert isinstance(source_structure, EmbeddedRealSimpleNumberFieldMatrix)
    recognized = recognize_real_simple_number_field(source_structure.embedding)
    change_rational = DomainMatrix([[1, 1], [0, 1]], (2, 2), QQ)
    change = change_rational.convert_to(recognized.field)
    source_j = domain_matrix_from_embedded(source_structure, recognized)
    target_j = change.inv().matmul(source_j).matmul(change).to_dense()
    target_torus = LatticeComplexStructure(
        coordinate_axis=source_torus.coordinate_axis,
        complex_structure=embedded_matrix_from_domain(target_j, recognized),
    )
    source_form_matrix = Matrix(((0, -1), (1, 0)))
    change_matrix = Matrix(((1, 1), (0, 1)))
    target_form_matrix = change_matrix.T * source_form_matrix * change_matrix

    def form(matrix: Matrix) -> IntegralBilinearForm:
        return IntegralBilinearForm(
            coordinate_axis=source_torus.coordinate_axis,
            kind="ALTERNATING",
            matrix=IntegerMatrix(
                entries=tuple(
                    tuple(str(int(matrix[row, column])) for column in range(2))
                    for row in range(2)
                )
            ),
        )

    source_lattice = compute_neron_severi_lattice(source_torus)
    target_lattice = compute_neron_severi_lattice(target_torus)
    transported_forms = [
        change_matrix.T
        * Matrix([[int(value) for value in row] for row in basis.matrix.entries])
        * change_matrix
        for basis in source_lattice.basis_forms
    ]
    transported_saturation, _, _ = saturate_lattice(
        [[int(value) for value in matrix] for matrix in transported_forms]
    )
    target_saturation, _, _ = saturate_lattice(
        [
            [int(value) for row in basis.matrix.entries for value in row]
            for basis in target_lattice.basis_forms
        ]
    )
    assert source_lattice.rank == target_lattice.rank == 1
    assert transported_saturation == target_saturation

    source_profile = compute_riemann_form_profile(
        source_torus,
        form(source_form_matrix),
    )
    target_profile = compute_riemann_form_profile(
        target_torus,
        form(target_form_matrix),
    )
    assert (
        source_profile.outcome.status
        == target_profile.outcome.status
        == ("RIEMANN_FORM")
    )
    assert source_profile.outcome.polarization_type == (
        target_profile.outcome.polarization_type
    )
    assert source_profile.outcome.hermitian_inertia == (
        target_profile.outcome.hermitian_inertia
    )
    source_associated = source_profile.outcome.associated_form_inertia.matrix
    target_associated = target_profile.outcome.associated_form_inertia.matrix
    assert isinstance(source_associated, EmbeddedRealSimpleNumberFieldMatrix)
    assert isinstance(target_associated, EmbeddedRealSimpleNumberFieldMatrix)
    assert domain_matrix_from_embedded(target_associated, recognized) == (
        change.transpose()
        .matmul(domain_matrix_from_embedded(source_associated, recognized))
        .matmul(change)
        .to_dense()
    )


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


def test_unimodular_coordinate_change_transports_the_public_neron_severi_lattice() -> (
    None
):
    axis = ("e1", "e2", "e3", "e4")
    complex_structure = Matrix(
        (
            (0, 1, 0, 0),
            (-1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, -1, 0),
        )
    )
    coordinate_change = Matrix(
        (
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 1, 1),
            (0, 0, 0, 1),
        )
    )
    assert coordinate_change.det() == 1
    transported_structure = (
        coordinate_change.inv() * complex_structure * coordinate_change
    )

    def torus(matrix: Matrix) -> LatticeComplexStructure:
        return LatticeComplexStructure(
            coordinate_axis=axis,
            complex_structure=RationalMatrix(
                entries=tuple(
                    tuple(rational(int(matrix[row, column])) for column in range(4))
                    for row in range(4)
                )
            ),
        )

    source = compute_neron_severi_lattice(torus(complex_structure))
    target = compute_neron_severi_lattice(torus(transported_structure))
    transported_forms = [
        coordinate_change.T
        * Matrix([[int(value) for value in row] for row in form.matrix.entries])
        * coordinate_change
        for form in source.basis_forms
    ]
    transported_saturation, _, _ = saturate_lattice(
        [[int(value) for value in matrix] for matrix in transported_forms]
    )
    target_saturation, _, _ = saturate_lattice(
        [
            [int(value) for row in form.matrix.entries for value in row]
            for form in target.basis_forms
        ]
    )

    assert source.rank == target.rank == 4
    assert transported_saturation == target_saturation
