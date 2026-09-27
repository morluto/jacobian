from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafStalk,
    from_cover_maps,
    identity_morphism,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix


def _sheaf(field: SheafField, prime: int | None):
    complex_ = canonical_complex(("a", "b"), (("a",), ("b",)))
    sheaf = from_cover_maps(
        complex_,
        field,
        prime,
        (
            SheafStalk(simplex=("a",), basis=("x", "y")),
            SheafStalk(simplex=("b",), basis=()),
        ),
        (),
    ).sheaf
    assert sheaf is not None
    return sheaf


def _triangle_sheaf():
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        SheafField.PRIME_FIELD,
        3,
        tuple(SheafStalk(simplex=face, basis=("x",)) for face in faces),
        tuple(
            CoverRestrictionMatrix(source=face, target=coface, entries=((1,),))
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def test_identity_morphism_retains_exact_stalk_axes_and_round_trips():
    sheaf = _sheaf(SheafField.RATIONAL, None)

    result = identity_morphism(sheaf)

    assert result.source == result.target == sheaf
    assert result.natural and result.obstruction is None
    assert result.components == (
        (
            ("a",),
            (
                (
                    CanonicalRational.from_fraction(Fraction(1)),
                    CanonicalRational.from_fraction(Fraction(0)),
                ),
                (
                    CanonicalRational.from_fraction(Fraction(0)),
                    CanonicalRational.from_fraction(Fraction(1)),
                ),
            ),
        ),
        (("b",), ()),
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_identity_morphism_uses_prime_field_scalars():
    sheaf = _sheaf(SheafField.PRIME_FIELD, 3)

    result = identity_morphism(sheaf)

    assert result.components == ((("a",), ((1, 0), (0, 1))), (("b",), ()))


def test_identity_morphism_reports_identity_owned_domain_errors():
    with pytest.raises(OperationDomainValidationError) as non_sheaf:
        identity_morphism(object())  # type: ignore[arg-type]
    assert non_sheaf.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.morphism_identity.parent_type_invalid"
    )
    assert non_sheaf.value.errors()[0]["loc"] == ("sheaf",)

    sheaf = _triangle_sheaf()
    assert sheaf.derived_restrictions
    forged = sheaf.model_copy(
        update={
            "derived_restrictions": (
                sheaf.derived_restrictions[0].model_copy(update={"entries": ((9,),)}),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as forged_error:
        identity_morphism(forged)
    assert forged_error.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.morphism_identity.parent_diagram_not_admitted"
    )
    assert forged_error.value.errors()[0]["loc"] == ("sheaf",)

    malformed_stalk = SheafStalk.model_construct(
        simplex=("a",), basis=("duplicate", "duplicate")
    )
    malformed = sheaf.model_copy(
        update={"stalks": (malformed_stalk, *sheaf.stalks[1:])}
    )
    with pytest.raises(OperationDomainValidationError) as malformed_error:
        identity_morphism(malformed)
    assert malformed_error.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.morphism_identity.stalks_not_admitted"
    )

    oversized_stalk = SheafStalk.model_construct(
        simplex=("a",), basis=tuple(f"x{index}" for index in range(1000))
    )
    oversized = sheaf.model_copy(
        update={"stalks": (oversized_stalk, *sheaf.stalks[1:])}
    )
    with pytest.raises(OperationResourceAdmissionError, match="stalk-rank"):
        identity_morphism(oversized)

    too_many_stalks = sheaf.model_copy(update={"stalks": sheaf.stalks * 33})
    with pytest.raises(OperationResourceAdmissionError, match="stalk count"):
        identity_morphism(too_many_stalks)
