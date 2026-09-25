from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafMorphismIdentityRequest,
    SheafStalk,
    from_cover_maps,
    identity_morphism,
)


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


def test_identity_morphism_uses_prime_field_scalars_and_is_in_catalog():
    sheaf = _sheaf(SheafField.PRIME_FIELD, 3)

    result = identity_morphism(sheaf)

    assert result.components == ((("a",), ((1, 0), (0, 1))), (("b",), ()))
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "cellular_sheaf.morphism.identity.compute"
    )
    assert tool.run(SheafMorphismIdentityRequest(sheaf=sheaf)) == result
