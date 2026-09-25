from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.homomorphism._models import (
    FreeAlgebraHomomorphism,
    FreeAlgebraHomomorphismApplyRequest,
)
from jacobian.math.free_algebras.homomorphism.operations import apply


def polynomial(alphabet, *terms):
    return FreeAlgebraPolynomial.model_validate(
        {
            "alphabet": alphabet,
            "terms": [
                {"coefficient": CanonicalRational.from_fraction(c), "word": list(word)}
                for word, c in terms
            ],
        }
    )


def test_apply_preserves_order_and_matches_independent_matrix_evaluation():
    target = ("u", "v")
    hom = FreeAlgebraHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=target,
        generator_images=(
            polynomial(target, (("v",), Fraction(1)), (("u",), Fraction(1))),
            polynomial(target, (("u", "v"), Fraction(1))),
        ),
    )
    source = polynomial(
        ("x", "y"), (("y", "x"), Fraction(-1)), (("x", "y"), Fraction(1))
    )
    result = apply(
        FreeAlgebraHomomorphismApplyRequest(homomorphism=hom, polynomial=source)
    )
    assert [
        (term.word, term.coefficient.as_fraction()) for term in result.image.terms
    ] == [
        (("v", "u", "v"), Fraction(1)),
        (("u", "v", "v"), Fraction(-1)),
        (("u", "v", "u"), Fraction(-1)),
        (("u", "u", "v"), Fraction(1)),
    ]

    # Evaluate both source and result by a separate representation in M_2(Q).
    matrices = {
        "u": ((Fraction(1), Fraction(1)), (Fraction(0), Fraction(1))),
        "v": ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0))),
    }

    def multiply(a, b):
        return tuple(
            tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
            for i in range(2)
        )

    def evaluate(poly, images):
        total = ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0)))
        for term in poly.terms:
            value = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
            for letter in term.word:
                value = multiply(value, images[letter])
            total = tuple(
                tuple(
                    total[i][j] + term.coefficient.as_fraction() * value[i][j]
                    for j in range(2)
                )
                for i in range(2)
            )
        return total

    assert evaluate(result.image, matrices) == evaluate(
        source,
        {
            "x": tuple(
                tuple(matrices["u"][i][j] + matrices["v"][i][j] for j in range(2))
                for i in range(2)
            ),
            "y": multiply(matrices["u"], matrices["v"]),
        },
    )


def test_empty_axes_zero_and_unit_are_preserved():
    hom = FreeAlgebraHomomorphism(
        source_alphabet=(), target_alphabet=("z",), generator_images=()
    )
    zero = polynomial(
        (),
    )
    result = apply(
        FreeAlgebraHomomorphismApplyRequest(homomorphism=hom, polynomial=zero)
    )
    assert result.image.alphabet == ("z",)
    assert result.image.terms == ()

    unit_hom = FreeAlgebraHomomorphism(
        source_alphabet=("x",),
        target_alphabet=(),
        generator_images=(polynomial((), ((), Fraction(2))),),
    )
    unit = polynomial(("x",), ((), Fraction(3)))
    image = apply(
        FreeAlgebraHomomorphismApplyRequest(homomorphism=unit_hom, polynomial=unit)
    ).image
    assert [(term.word, term.coefficient.as_fraction()) for term in image.terms] == [
        ((), Fraction(3))
    ]


def test_equal_generator_images_are_collected_exactly():
    target = ("u",)
    u = polynomial(target, (("u",), Fraction(1)))
    hom = FreeAlgebraHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=target,
        generator_images=(u, u),
    )
    source = polynomial(("x", "y"), (("y",), Fraction(-1)), (("x",), Fraction(1)))
    image = apply(
        FreeAlgebraHomomorphismApplyRequest(homomorphism=hom, polynomial=source)
    ).image
    assert image.terms == ()


def test_map_requires_one_image_per_source_generator_and_matching_axes():
    with pytest.raises(ValidationError, match="homomorphism_image_count"):
        FreeAlgebraHomomorphism(
            source_alphabet=("x",), target_alphabet=(), generator_images=()
        )
    with pytest.raises(ValidationError, match="homomorphism_target_axis"):
        FreeAlgebraHomomorphism(
            source_alphabet=("x",),
            target_alphabet=("u",),
            generator_images=(polynomial(("v",), (("v",), Fraction(1))),),
        )
    with pytest.raises(ValidationError, match="homomorphism_source_axis"):
        FreeAlgebraHomomorphismApplyRequest(
            homomorphism=FreeAlgebraHomomorphism(
                source_alphabet=("x",),
                target_alphabet=(),
                generator_images=(polynomial(()),),
            ),
            polynomial=polynomial(("y",)),
        )
