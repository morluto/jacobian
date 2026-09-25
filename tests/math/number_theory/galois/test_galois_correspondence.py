import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.galois._models import (
    GaloisFixedFieldRequest,
    GaloisSubgroupRequest,
    IntermediateFieldStabilizerRequest,
)
from jacobian.math.number_theory.galois.operations import (
    automorphisms,
    galois_fixed_field,
    galois_subgroup,
    intermediate_field_stabilizer,
    splitting_field,
)
from jacobian.math.number_theory.number_fields._field_embedding import (
    SimpleNumberFieldEmbedding,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _polynomial(coefficients: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": value, "den": 1},
                        "exponents": [power],
                    }
                    for power, value in reversed(tuple(enumerate(coefficients)))
                    if value
                ]
            },
        }
    )


def _split(coefficients: tuple[int, ...]):
    return splitting_field(_polynomial(coefficients))


def _subgroup(field, maps):
    return galois_subgroup(GaloisSubgroupRequest(field=field, elements=tuple(maps)))


def _element(presentation, *coordinates: int):
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            CanonicalRational(num=value, den=1) for value in coordinates
        ),
    )


def test_quadratic_galois_correspondence_binds_both_directions_exactly() -> None:
    field = _split((-2, 0, 1)).field
    identity, conjugation = automorphisms(field).automorphisms

    trivial = _subgroup(field, (identity,))
    whole = _subgroup(field, (identity, conjugation))

    trivial_fixed = galois_fixed_field(GaloisFixedFieldRequest(subgroup=trivial))
    whole_fixed = galois_fixed_field(GaloisFixedFieldRequest(subgroup=whole))
    assert trivial_fixed.fixed_field == field.extension
    assert trivial_fixed.inclusion.generator_image == _element(field.extension, 0, 1)
    assert whole_fixed.fixed_field == SimpleNumberFieldPresentation(
        coefficients_descending=(1, 0)
    )
    assert whole_fixed.inclusion.generator_image == _element(field.extension, 0, 0)

    rational_stabilizer = intermediate_field_stabilizer(
        IntermediateFieldStabilizerRequest(field=field, inclusion=whole_fixed.inclusion)
    ).subgroup
    extension_stabilizer = intermediate_field_stabilizer(
        IntermediateFieldStabilizerRequest(
            field=field, inclusion=trivial_fixed.inclusion
        )
    ).subgroup
    assert rational_stabilizer == whole
    assert extension_stabilizer == trivial


def test_split_quadratic_carrier_has_only_the_trivial_correspondence() -> None:
    field = _split((-1, 0, 1)).field
    (identity,) = automorphisms(field).automorphisms
    subgroup = _subgroup(field, (identity,))
    fixed = galois_fixed_field(GaloisFixedFieldRequest(subgroup=subgroup))
    assert fixed.fixed_field == field.extension
    assert fixed.inclusion.source == fixed.inclusion.target

    stabilizer = intermediate_field_stabilizer(
        IntermediateFieldStabilizerRequest(field=field, inclusion=fixed.inclusion)
    )
    assert stabilizer.subgroup == subgroup


def test_subgroup_admission_rejects_non_subsets_of_the_exact_parent() -> None:
    field = _split((-2, 0, 1)).field
    identity, conjugation = automorphisms(field).automorphisms
    with pytest.raises(OperationDomainValidationError, match="identity"):
        _subgroup(field, (conjugation,))

    other_field = _split((-3, 0, 1)).field
    other_conjugation = next(
        auto
        for auto in automorphisms(other_field).automorphisms
        if auto.root_permutation == (1, 0)
    )
    with pytest.raises(OperationDomainValidationError):
        _subgroup(field, (identity, other_conjugation))


def test_intermediate_field_embedding_must_target_the_exact_extension() -> None:
    field = _split((-2, 0, 1)).field
    rational = SimpleNumberFieldPresentation(coefficients_descending=(1, 0))
    wrong_target = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -3))
    inclusion = SimpleNumberFieldEmbedding(
        source=rational,
        target=wrong_target,
        generator_image=_element(wrong_target, 0, 0),
    )
    with pytest.raises(OperationDomainValidationError, match="retained extension"):
        intermediate_field_stabilizer(
            IntermediateFieldStabilizerRequest(field=field, inclusion=inclusion)
        )


def test_stabilizer_uses_the_exact_embedded_image_not_just_field_degree() -> None:
    field = _split((-2, 0, 1)).field
    source = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -8))
    inclusion = SimpleNumberFieldEmbedding(
        source=source,
        target=field.extension,
        generator_image=_element(field.extension, 0, 2),
    )

    stabilizer = intermediate_field_stabilizer(
        IntermediateFieldStabilizerRequest(field=field, inclusion=inclusion)
    )
    assert stabilizer.inclusion == inclusion
    assert len(stabilizer.subgroup.elements) == 1
    assert stabilizer.subgroup.elements[0].root_permutation == (0, 1)


def test_stabilizer_rejects_a_generator_image_that_breaks_the_field_relation() -> None:
    field = _split((-2, 0, 1)).field
    source = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -3))
    inclusion = SimpleNumberFieldEmbedding(
        source=source,
        target=field.extension,
        generator_image=_element(field.extension, 0, 1),
    )

    with pytest.raises(OperationDomainValidationError, match="injective QQ-field map"):
        intermediate_field_stabilizer(
            IntermediateFieldStabilizerRequest(field=field, inclusion=inclusion)
        )
