from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FiniteCellularSheaf,
    SheafField,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafMorphismResult,
    compose_morphisms,
    morphism,
)
from jacobian.math.topology.cellular_sheaves.morphism_add import add_morphisms


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def _one_point_sheaf(field: SheafField = SheafField.RATIONAL) -> FiniteCellularSheaf:
    from jacobian.math.topology._models import canonical_complex

    complex_ = canonical_complex(("v",), (("v",),))
    return FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=field,
        prime=3 if field is SheafField.PRIME_FIELD else None,
        stalks=(SheafStalk(simplex=("v",), basis=("e",)),),
        cover_restrictions=(),
        derived_restrictions=(),
        diamonds=0,
        comparable_pairs=0,
    )


def _map(
    sheaf: FiniteCellularSheaf, value: CanonicalRational | int
) -> SheafMorphismResult:
    return morphism(sheaf, sheaf, ((("v",), ((value,),)),))


def test_addition_is_exact_and_result_composes_unchanged() -> None:
    sheaf = _one_point_sheaf()
    left = _map(sheaf, _q(1, 2))
    right = _map(sheaf, _q(1, 3))

    result = add_morphisms(left, right)

    assert result.natural
    assert result.components == (
        (
            ("v",),
            ((_q(5, 6),),),
        ),
    )
    assert compose_morphisms(result, _map(sheaf, _q(1))).components == result.components


def test_prime_field_addition_reduces_in_the_declared_field() -> None:
    sheaf = _one_point_sheaf(SheafField.PRIME_FIELD)

    result = add_morphisms(_map(sheaf, 2), _map(sheaf, 2))

    assert result.components == ((("v",), ((1,),)),)


def test_empty_stalk_component_addition_preserves_empty_axes() -> None:
    sheaf = _one_point_sheaf()
    zero_sheaf = sheaf.model_copy(
        update={"stalks": (SheafStalk(simplex=("v",), basis=()),)}
    )
    left = morphism(zero_sheaf, zero_sheaf, ((("v",), ()),))

    result = add_morphisms(left, left)

    assert result.components == ((("v",), ()),)


def test_naturality_claim_is_rechecked_and_common_parents_are_required() -> None:
    sheaf = _one_point_sheaf()
    natural_but_untrusted = _map(sheaf, _q(1)).model_copy(update={"natural": False})
    # The serialized flag is not authority; the component map is.
    assert add_morphisms(natural_but_untrusted, natural_but_untrusted).natural

    other = _one_point_sheaf().model_copy(
        update={"stalks": (SheafStalk(simplex=("v",), basis=("f",)),)}
    )
    with pytest.raises(OperationDomainValidationError, match="exactly equal"):
        add_morphisms(
            _map(sheaf, _q(1)),
            _map(other, _q(1)),
        )

    from jacobian.math.topology._models import canonical_complex
    from jacobian.math.topology.cellular_sheaves.operations import from_cover_maps

    interval = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in interval.faces_by_dimension for face in group.faces)
    sheaf_result = from_cover_maps(
        interval,
        SheafField.RATIONAL,
        None,
        tuple(SheafStalk(simplex=face, basis=("e",)) for face in faces),
        tuple(
            CoverRestrictionMatrix(
                source=vertex,
                target=("a", "b"),
                entries=((_q(1),),),
            )
            for vertex in (("a",), ("b",))
        ),
    )
    assert sheaf_result.sheaf is not None
    interval_sheaf = sheaf_result.sheaf
    non_natural = morphism(
        interval_sheaf,
        interval_sheaf,
        tuple(
            (face, ((_q(0 if len(face) == 2 else 1),),))
            for face in interval_sheaf.canonical_face_order
        ),
    ).model_copy(update={"natural": True})
    with pytest.raises(OperationDomainValidationError, match="natural"):
        add_morphisms(non_natural, non_natural)


def test_growth_above_composable_scalar_envelope_is_typed_refusal() -> None:
    sheaf = _one_point_sheaf()
    boundary = _q(10**63)
    at_boundary = add_morphisms(_map(sheaf, boundary), _map(sheaf, _q(0)))
    assert at_boundary.components == ((("v",), ((boundary,),)),)

    large = _q(int("9" * 64))

    with pytest.raises(OperationResourceAdmissionError, match="scalar"):
        add_morphisms(_map(sheaf, large), _map(sheaf, large))
