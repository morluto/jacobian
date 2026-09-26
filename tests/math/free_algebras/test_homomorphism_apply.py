from fractions import Fraction
from itertools import product as cartesian_product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialHomomorphism,
)
from jacobian.math.free_algebras.homomorphism._models import (
    FreeAlgebraHomomorphismApplyRequest,
)
from jacobian.math.free_algebras.homomorphism.operations import apply
from jacobian.math.free_algebras.operations import multiply


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
    hom = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=target,
        images=(
            polynomial(target, (("v",), Fraction(1)), (("u",), Fraction(1))),
            polynomial(target, (("u", "v"), Fraction(1))),
        ),
    )
    source = polynomial(
        ("x", "y"), (("y", "x"), Fraction(-1)), (("x", "y"), Fraction(1))
    )
    result = apply(hom, source)
    assert [(term.word, term.coefficient.as_fraction()) for term in result.terms] == [
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

    assert evaluate(result, matrices) == evaluate(
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
    hom = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=(), target_alphabet=("z",), images=()
    )
    zero = polynomial(
        (),
    )
    result = apply(hom, zero)
    assert result.alphabet == ("z",)
    assert result.terms == ()

    unit_hom = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x",),
        target_alphabet=(),
        images=(polynomial((), ((), Fraction(2))),),
    )
    unit = polynomial(("x",), ((), Fraction(3)))
    image = apply(unit_hom, unit)
    assert [(term.word, term.coefficient.as_fraction()) for term in image.terms] == [
        ((), Fraction(3))
    ]


def test_equal_generator_images_are_collected_exactly():
    target = ("u",)
    u = polynomial(target, (("u",), Fraction(1)))
    hom = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=target,
        images=(u, u),
    )
    source = polynomial(("x", "y"), (("y",), Fraction(-1)), (("x",), Fraction(1)))
    image = apply(hom, source)
    assert image.terms == ()


def test_map_requires_one_image_per_source_generator_and_matching_axes():
    with pytest.raises(ValidationError, match="homomorphism_source_axis"):
        FreeAlgebraHomomorphismApplyRequest(
            homomorphism=FreeAlgebraPolynomialHomomorphism(
                source_alphabet=("x",),
                target_alphabet=(),
                images=(polynomial(()),),
            ),
            polynomial=polynomial(("y",)),
        )


def test_application_roundtrips_and_composes_through_typed_json_values():
    source_alphabet = ("x",)
    target_alphabet = ("u", "v")
    homomorphism = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=source_alphabet,
        target_alphabet=target_alphabet,
        images=(
            polynomial(
                target_alphabet,
                (("v",), Fraction(1)),
                (("u",), Fraction(1)),
            ),
        ),
    )
    request = FreeAlgebraHomomorphismApplyRequest(
        homomorphism=homomorphism,
        polynomial=polynomial(source_alphabet, (("x", "x"), Fraction(1))),
    )
    decoded_request = FreeAlgebraHomomorphismApplyRequest.model_validate_json(
        request.model_dump_json()
    )
    result = apply(decoded_request.homomorphism, decoded_request.polynomial)
    decoded_result = FreeAlgebraPolynomial.model_validate_json(result.model_dump_json())

    assert decoded_result == polynomial(
        target_alphabet,
        (("v", "v"), Fraction(1)),
        (("v", "u"), Fraction(1)),
        (("u", "v"), Fraction(1)),
        (("u", "u"), Fraction(1)),
    )
    squared = multiply(decoded_result, decoded_result).product
    assert {term.word for term in squared.terms} == set(
        cartesian_product(target_alphabet, repeat=4)
    )
    assert all(term.coefficient.as_fraction() == 1 for term in squared.terms)
