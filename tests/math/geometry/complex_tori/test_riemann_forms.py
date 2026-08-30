"""Exact selected Riemann-form profiles on complex tori."""

from fractions import Fraction

from sympy import QQ
from sympy.polys.matrices import DomainMatrix
from tests.math.geometry.complex_tori._support import (
    index_six_alternating_form,
    quartic_index_six_torus,
    quartic_rank_one_torus,
    standard_alternating_form,
)

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.complex_tori import (
    LatticeComplexStructure,
    compute_riemann_form_profile,
)
from jacobian.math.geometry.complex_tori._models import RiemannFormProfile
from jacobian.math.lattices.invariant_forms import IntegralBilinearForm
from jacobian.math.matrices._number_field import (
    domain_matrix_from_embedded,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.values import IntegerMatrix, RationalMatrix


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def test_elliptic_degree_d_form_uses_standard_positive_sign_and_type() -> None:
    d = 6
    torus = LatticeComplexStructure(
        coordinate_axis=("e1", "e2"),
        complex_structure=RationalMatrix(
            entries=(
                (_rational(0), _rational(1)),
                (_rational(-1), _rational(0)),
            )
        ),
    )
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", str(d)), (str(-d), "0"))),
    )

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "HODGE_TYPE_11"
    assert result.outcome.associated_form_convention == "J_TRANSPOSE_TIMES_E"
    assert result.outcome.hermitian_form_convention == "G_PLUS_I_E_LINEAR_IN_FIRST"
    assert result.outcome.associated_form_inertia.matrix.entries == (
        (_rational(d), _rational(0)),
        (_rational(0), _rational(d)),
    )
    assert result.outcome.hermitian_inertia.n_positive == 1
    assert result.outcome.is_riemann_form is True
    assert result.alternating_elementary_divisors == (str(d),)
    assert result.outcome.polarization_type == (str(d),)
    assert (
        RiemannFormProfile.model_validate_json(
            encode_strict_json(result.model_dump(mode="json")),
            strict=True,
        )
        == result
    )


def test_quartic_rank_one_generator_has_hermitian_signature_one_one() -> None:
    torus = quartic_rank_one_torus()
    form = standard_alternating_form(torus)

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "HODGE_TYPE_11"
    assert result.outcome.associated_form_inertia.n_positive == 2
    assert result.outcome.associated_form_inertia.n_negative == 2
    assert result.outcome.associated_form_inertia.n_zero == 0
    assert result.outcome.hermitian_inertia.n_positive == 1
    assert result.outcome.hermitian_inertia.n_negative == 1
    assert result.outcome.hermitian_inertia.n_zero == 0
    assert result.outcome.hermitian_inertia.definiteness == "indefinite"
    assert result.outcome.is_riemann_form is False
    assert result.outcome.polarization_type is None


def test_selected_real_embedding_changes_exact_riemann_inertia() -> None:
    positive_root_torus = quartic_rank_one_torus(root_index=1)
    negative_root_torus = quartic_rank_one_torus(root_index=0)

    positive_result = compute_riemann_form_profile(
        positive_root_torus,
        standard_alternating_form(positive_root_torus),
    )
    negative_result = compute_riemann_form_profile(
        negative_root_torus,
        standard_alternating_form(negative_root_torus),
    )

    assert positive_result.outcome.status == "HODGE_TYPE_11"
    assert negative_result.outcome.status == "HODGE_TYPE_11"
    assert positive_result.outcome.hermitian_inertia.definiteness == "indefinite"
    assert negative_result.outcome.hermitian_inertia.n_positive == 0
    assert negative_result.outcome.hermitian_inertia.n_negative == 2
    assert negative_result.outcome.hermitian_inertia.definiteness == "negative_definite"
    assert (
        positive_result.outcome.associated_form_inertia.matrix.embedding
        == positive_root_torus.complex_structure.embedding
    )
    assert (
        negative_result.outcome.associated_form_inertia.matrix.embedding
        == negative_root_torus.complex_structure.embedding
    )


def test_basis_changed_index_six_fixture_has_exact_type_and_signature() -> None:
    # This is a separate algebraic torus built from the paper-shaped eta matrix;
    # it does not claim to specialize the paper's very-general symbolic family.
    torus = quartic_index_six_torus()
    form = index_six_alternating_form(torus)

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "HODGE_TYPE_11"
    assert result.alternating_elementary_divisors == ("1", "6")
    assert result.outcome.associated_form_inertia.n_positive == 2
    assert result.outcome.associated_form_inertia.n_negative == 2
    assert result.outcome.hermitian_inertia.n_positive == 1
    assert result.outcome.hermitian_inertia.n_negative == 1
    assert result.outcome.is_riemann_form is False


def test_index_six_fixture_is_an_explicit_change_of_lattice() -> None:
    source_torus = quartic_rank_one_torus()
    target_torus = quartic_index_six_torus()
    source_form = standard_alternating_form(source_torus)
    target_form = index_six_alternating_form(target_torus)
    # The columns of P generate an index-six sublattice. Thus J_source P =
    # P J_target and eta = P^T E P, but P is not a GL(4, ZZ) basis change;
    # this remains a separate algebraic torus rather than a paper-family point.
    change_rows = (
        (0, 1, 0, 0),
        (0, 0, 0, -1),
        (0, 0, 1, 0),
        (6, 0, 0, 0),
    )
    recognized = recognize_real_simple_number_field(
        source_torus.complex_structure.embedding
    )
    change_rational = DomainMatrix(
        [list(row) for row in change_rows],
        (4, 4),
        QQ,
    )
    change = change_rational.convert_to(recognized.field)
    source_j = domain_matrix_from_embedded(
        source_torus.complex_structure,
        recognized,
    )
    target_j = domain_matrix_from_embedded(
        target_torus.complex_structure,
        recognized,
    )
    source_e = DomainMatrix(
        [[int(value) for value in row] for row in source_form.matrix.entries],
        (4, 4),
        QQ,
    )
    target_e = DomainMatrix(
        [[int(value) for value in row] for row in target_form.matrix.entries],
        (4, 4),
        QQ,
    )

    assert change.det() == recognized.field.convert(-6)
    assert source_j.matmul(change) == change.matmul(target_j)
    assert change_rational.transpose().matmul(source_e).matmul(change_rational) == (
        target_e
    )


def test_non_hodge_alternating_form_is_a_discriminated_mathematical_outcome() -> None:
    torus = quartic_rank_one_torus()
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(
            entries=(
                ("0", "1", "0", "0"),
                ("-1", "0", "0", "0"),
                ("0", "0", "0", "0"),
                ("0", "0", "0", "0"),
            )
        ),
    )

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "NOT_HODGE_TYPE_11"
    assert result.outcome.is_riemann_form is False
    assert result.alternating_elementary_divisors == ("1",)
    assert result.is_degenerate is True


def test_zero_hodge_form_has_zero_hermitian_inertia_without_a_type() -> None:
    torus = LatticeComplexStructure(
        coordinate_axis=("e1", "e2"),
        complex_structure=RationalMatrix(
            entries=(
                (_rational(0), _rational(1)),
                (_rational(-1), _rational(0)),
            )
        ),
    )
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", "0"), ("0", "0"))),
    )

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "HODGE_TYPE_11"
    assert result.outcome.associated_form_inertia.n_positive == 0
    assert result.outcome.associated_form_inertia.n_negative == 0
    assert result.outcome.associated_form_inertia.n_zero == 2
    assert result.outcome.hermitian_inertia.n_positive == 0
    assert result.outcome.hermitian_inertia.n_negative == 0
    assert result.outcome.hermitian_inertia.n_zero == 1
    assert result.outcome.hermitian_inertia.definiteness == "zero"
    assert result.outcome.is_riemann_form is False
    assert result.outcome.polarization_type is None
    assert result.alternating_elementary_divisors == ()
    assert result.is_degenerate is True
