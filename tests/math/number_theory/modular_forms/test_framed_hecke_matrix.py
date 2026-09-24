from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    modular_form_basis_frame,
    modular_form_coordinates_hecke,
    modular_form_coordinates_to_frame,
    modular_form_hecke_matrix_in_frame,
)
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormBasisFrameRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormFramedHeckeMatrix,
)


def _rat(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _frame():
    space = ModularFormSpace(level=1, weight=12, kind="M")
    return modular_form_basis_frame(
        ModularFormBasisFrameRequest(
            space=space,
            source_basis_id="level-one-e4-e6-monomials-v1",
            source_labels=("E4^3", "E6^2"),
            labels=("c1", "c2"),
            # c1=E4^3+E6^2 and c2=E6^2.
            entries=((_rat(1), _rat(0)), (_rat(1), _rat(1))),
        )
    )


def test_framed_hecke_matrix_is_exact_conjugate_and_matches_coordinate_action():
    frame = _frame()

    result = modular_form_hecke_matrix_in_frame(frame, 2)

    assert result.row_labels == result.column_labels == ("c1", "c2")
    entries = tuple(
        tuple(value.as_fraction() for value in row) for row in result.entries
    )
    assert entries == (
        (Fraction(2622), Fraction(1323)),
        (Fraction(-1146), Fraction(-597)),
    )
    assert (
        ModularFormFramedHeckeMatrix.model_validate_json(result.model_dump_json())
        == result
    )

    for column, canonical_coordinates in enumerate(((1, 1), (0, 1))):
        canonical_form = ModularFormCoordinates(
            space=frame.space,
            basis_id=frame.source_basis_id,
            coordinates=tuple(_rat(value) for value in canonical_coordinates),
        )
        canonical_image = modular_form_coordinates_hecke(canonical_form, 2)
        framed_image = modular_form_coordinates_to_frame(frame, canonical_image)
        assert tuple(
            value.as_fraction() for value in framed_image.coordinates
        ) == tuple(entries[row][column] for row in range(2))


def test_zero_dimensional_cusp_space_has_empty_framed_hecke_matrix():
    space = ModularFormSpace(level=1, weight=4, kind="S")
    frame = modular_form_basis_frame(
        ModularFormBasisFrameRequest(
            space=space,
            source_basis_id="level-one-e4-e6-monomials-v1",
            source_labels=(),
            labels=(),
            entries=(),
        )
    )

    result = modular_form_hecke_matrix_in_frame(frame, 1)

    assert result.entries == result.row_labels == result.column_labels == ()
