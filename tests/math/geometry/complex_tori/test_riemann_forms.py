"""Exact selected Riemann-form profiles on complex tori."""

import copy
from fractions import Fraction

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from sympy import QQ, Matrix
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
from jacobian.math.geometry.complex_tori._tools import RIEMANN_FORM_PROFILE_OPERATION
from jacobian.math.lattices.invariant_forms import IntegralBilinearForm
from jacobian.math.matrices._number_field import (
    domain_matrix_from_embedded,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    IntegerMatrix,
    RationalMatrix,
)


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

    assert result.outcome.status == "RIEMANN_FORM"
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

    request = RIEMANN_FORM_PROFILE_OPERATION.request_type.model_validate_json(
        encode_strict_json(
            {
                "torus": torus.model_dump(mode="json"),
                "form": form.model_dump(mode="json"),
            }
        ),
        strict=True,
    )
    result = RIEMANN_FORM_PROFILE_OPERATION.run(request)

    assert result.outcome.status == "HODGE_NON_POSITIVE"
    assert result.outcome.associated_form_inertia.n_positive == 2
    assert result.outcome.associated_form_inertia.n_negative == 2
    assert result.outcome.associated_form_inertia.n_zero == 0
    assert result.outcome.hermitian_inertia.n_positive == 1
    assert result.outcome.hermitian_inertia.n_negative == 1
    assert result.outcome.hermitian_inertia.n_zero == 0
    assert result.outcome.hermitian_inertia.definiteness == "indefinite"
    assert result.outcome.is_riemann_form is False
    assert "polarization_type" not in result.outcome.model_dump()


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

    assert positive_result.outcome.status == "HODGE_NON_POSITIVE"
    assert negative_result.outcome.status == "HODGE_NON_POSITIVE"
    positive_associated = positive_result.outcome.associated_form_inertia.matrix
    negative_associated = negative_result.outcome.associated_form_inertia.matrix
    positive_complex_structure = positive_root_torus.complex_structure
    negative_complex_structure = negative_root_torus.complex_structure
    assert isinstance(positive_associated, EmbeddedRealSimpleNumberFieldMatrix)
    assert isinstance(negative_associated, EmbeddedRealSimpleNumberFieldMatrix)
    assert isinstance(positive_complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
    assert isinstance(negative_complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
    assert positive_result.outcome.hermitian_inertia.definiteness == "indefinite"
    assert negative_result.outcome.hermitian_inertia.n_positive == 0
    assert negative_result.outcome.hermitian_inertia.n_negative == 2
    assert negative_result.outcome.hermitian_inertia.definiteness == "negative_definite"
    assert positive_associated.embedding == positive_complex_structure.embedding
    assert negative_associated.embedding == negative_complex_structure.embedding


def test_basis_changed_index_six_fixture_has_exact_type_and_signature() -> None:
    # This is a separate algebraic torus built from the paper-shaped eta matrix;
    # it does not claim to specialize the paper's very-general symbolic family.
    torus = quartic_index_six_torus()
    form = index_six_alternating_form(torus)

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "HODGE_NON_POSITIVE"
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
    source_complex_structure = source_torus.complex_structure
    target_complex_structure = target_torus.complex_structure
    assert isinstance(source_complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
    assert isinstance(target_complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
    recognized = recognize_real_simple_number_field(source_complex_structure.embedding)
    change_rational = DomainMatrix(
        [list(row) for row in change_rows],
        (4, 4),
        QQ,
    )
    change = change_rational.convert_to(recognized.field)
    source_j = domain_matrix_from_embedded(
        source_complex_structure,
        recognized,
    )
    target_j = domain_matrix_from_embedded(
        target_complex_structure,
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


def test_unimodular_coordinate_change_transports_the_public_riemann_profile() -> None:
    axis = ("e1", "e2", "e3", "e4")
    complex_structure = Matrix(
        (
            (0, 1, 0, 0),
            (-1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, -1, 0),
        )
    )
    source_form = Matrix(
        (
            (0, 1, 0, 0),
            (-1, 0, 0, 0),
            (0, 0, 0, 3),
            (0, 0, -3, 0),
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
    target_structure = coordinate_change.inv() * complex_structure * coordinate_change
    target_form = coordinate_change.T * source_form * coordinate_change

    def torus(matrix: Matrix) -> LatticeComplexStructure:
        return LatticeComplexStructure(
            coordinate_axis=axis,
            complex_structure=RationalMatrix(
                entries=tuple(
                    tuple(_rational(int(matrix[row, column])) for column in range(4))
                    for row in range(4)
                )
            ),
        )

    def alternating_form(matrix: Matrix) -> IntegralBilinearForm:
        return IntegralBilinearForm(
            coordinate_axis=axis,
            kind="ALTERNATING",
            matrix=IntegerMatrix(
                entries=tuple(
                    tuple(str(int(matrix[row, column])) for column in range(4))
                    for row in range(4)
                )
            ),
        )

    source = compute_riemann_form_profile(
        torus(complex_structure), alternating_form(source_form)
    )
    target = compute_riemann_form_profile(
        torus(target_structure), alternating_form(target_form)
    )
    assert source.outcome.status == target.outcome.status == "RIEMANN_FORM"
    assert (
        source.outcome.polarization_type
        == target.outcome.polarization_type
        == (
            "1",
            "3",
        )
    )
    assert source.outcome.hermitian_inertia == target.outcome.hermitian_inertia
    source_matrix = source.outcome.associated_form_inertia.matrix
    target_matrix = target.outcome.associated_form_inertia.matrix
    assert isinstance(source_matrix, RationalMatrix)
    assert isinstance(target_matrix, RationalMatrix)
    source_associated = Matrix(
        [[value.as_fraction() for value in row] for row in source_matrix.entries]
    )
    target_associated = Matrix(
        [[value.as_fraction() for value in row] for row in target_matrix.entries]
    )
    assert (
        target_associated == coordinate_change.T * source_associated * coordinate_change
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

    assert result.outcome.status == "NOT_HODGE"
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

    assert result.outcome.status == "HODGE_NON_POSITIVE"
    assert result.outcome.associated_form_inertia.n_positive == 0
    assert result.outcome.associated_form_inertia.n_negative == 0
    assert result.outcome.associated_form_inertia.n_zero == 2
    assert result.outcome.associated_form_inertia.definiteness == "zero"
    assert result.outcome.hermitian_inertia.n_positive == 0
    assert result.outcome.hermitian_inertia.n_negative == 0
    assert result.outcome.hermitian_inertia.n_zero == 1
    assert result.outcome.hermitian_inertia.definiteness == "zero"
    assert result.outcome.is_riemann_form is False
    assert "polarization_type" not in result.outcome.model_dump()
    assert result.alternating_elementary_divisors == ()
    assert result.is_degenerate is True


def test_rank_fifty_two_zero_hodge_form_returns_a_typed_profile() -> None:
    dimension = 52
    zero = _rational(0)
    one = _rational(1)
    negative_one = _rational(-1)
    torus = LatticeComplexStructure(
        coordinate_axis=tuple(f"e{index + 1}" for index in range(dimension)),
        complex_structure=RationalMatrix(
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
        ),
    )
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(
            entries=tuple(
                tuple("0" for _ in range(dimension)) for _ in range(dimension)
            )
        ),
    )

    result = compute_riemann_form_profile(torus, form)

    assert result.outcome.status == "HODGE_NON_POSITIVE"
    assert result.outcome.associated_form_inertia.n_zero == dimension
    assert result.outcome.hermitian_inertia.n_zero == dimension // 2
    assert result.is_degenerate is True


def test_outcome_schema_requires_only_positive_polarization_data() -> None:
    schema = RiemannFormProfile.model_json_schema()
    outcome_schema = schema["properties"]["outcome"]
    assert set(outcome_schema["discriminator"]["mapping"]) == {
        "NOT_HODGE",
        "HODGE_NON_POSITIVE",
        "RIEMANN_FORM",
    }
    positive = schema["$defs"]["RiemannFormPositive"]
    nonpositive = schema["$defs"]["RiemannFormHodgeNonPositive"]
    not_hodge = schema["$defs"]["RiemannFormNotHodge"]
    assert all(
        "status" in branch["required"] for branch in (positive, nonpositive, not_hodge)
    )
    assert "polarization_type" in positive["required"]
    assert "polarization_type" not in nonpositive["properties"]


@pytest.mark.parametrize(
    ("form_scale", "expected_status"),
    ((1, "RIEMANN_FORM"), (-1, "HODGE_NON_POSITIVE")),
)
def test_hodge_outcome_schema_requires_every_exact_real_discriminator(
    form_scale: int,
    expected_status: str,
) -> None:
    torus = LatticeComplexStructure(
        coordinate_axis=("e1", "e2"),
        complex_structure=RationalMatrix(
            entries=((_rational(0), _rational(1)), (_rational(-1), _rational(0)))
        ),
    )
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", str(form_scale)), (str(-form_scale), "0"))),
    )
    payload = compute_riemann_form_profile(torus, form).model_dump(mode="json")
    assert payload["outcome"]["status"] == expected_status

    validator = Draft202012Validator(RiemannFormProfile.model_json_schema())
    assert not list(validator.iter_errors(payload))
    missing_torus_domain = copy.deepcopy(payload)
    del missing_torus_domain["torus"]["complex_structure"]["domain"]
    missing_inertia_domain = copy.deepcopy(payload)
    del missing_inertia_domain["outcome"]["associated_form_inertia"]["matrix"]["domain"]
    for invalid in (missing_torus_domain, missing_inertia_domain):
        assert list(validator.iter_errors(invalid))
        with pytest.raises(ValidationError):
            RiemannFormProfile.model_validate(invalid)
