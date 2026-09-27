from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.galois._models import (
    AutomorphismElementApplyRequest,
    QQFieldAutomorphism,
    QQSplittingField,
    SplittingFieldRequest,
    SplittingFieldResult,
)
from jacobian.math.number_theory.galois._tools import TOOLS
from jacobian.math.number_theory.galois.operations import (
    apply_automorphism,
    apply_automorphism_to_element,
    automorphisms,
    compose_automorphisms,
    inverse_automorphism,
    splitting_field,
)
from jacobian.math.number_theory.number_fields.values import SimpleNumberFieldElement
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


def _split(coefficients: tuple[int, ...]) -> SplittingFieldResult:
    return splitting_field(_polynomial(coefficients))


def _coordinates(element) -> tuple[Fraction, ...]:
    return tuple(
        coefficient.as_fraction() for coefficient in element.coefficients_ascending
    )


def _element(presentation, coordinates: tuple[Fraction, ...]):
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(value) for value in coordinates
        ),
    )


def _multiply(left, right, extension) -> tuple[Fraction, ...]:
    a = _coordinates(left)
    b = _coordinates(right)
    if extension.degree == 1:
        return (a[0] * b[0],)
    leading, linear, constant = map(Fraction, extension.coefficients_descending)
    product = [Fraction(0), Fraction(0), Fraction(0)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] += x * y
    product[0] -= product[2] * constant / leading
    product[1] -= product[2] * linear / leading
    return tuple(product[:2])


def test_quadratic_extension_has_exact_roots_factorization_and_full_group() -> None:
    result = _split((-2, 0, 1))
    assert result.field.extension.coefficients_descending == (1, 0, -2)
    assert tuple(_coordinates(root.value) for root in result.roots) == (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
    )
    assert tuple(_coordinates(value) for value in result.factor_reconstruction) == (
        (Fraction(-2), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
    )
    # Independent field arithmetic: each exact coordinate root annihilates x^2-2.
    for root in result.roots:
        assert _multiply(root.value, root.value, result.field.extension) == (
            Fraction(2),
            Fraction(0),
        )

    group = automorphisms(result.field).automorphisms
    assert len(group) == 2
    assert {auto.root_permutation for auto in group} == {(0, 1), (1, 0)}
    assert all(len(auto.basis_images) == 2 for auto in group)
    conjugation = next(auto for auto in group if auto.root_permutation == (1, 0))
    image = apply_automorphism(conjugation, result.roots[0])
    assert _coordinates(image.value) == (Fraction(0), Fraction(-1))
    identity = compose_automorphisms(conjugation, conjugation)
    assert identity.root_permutation == (0, 1)
    assert identity.basis_images == next(
        auto.basis_images for auto in group if auto.root_permutation == (0, 1)
    )


def test_automorphism_applies_to_arbitrary_field_elements_with_exact_oracle() -> None:
    split = _split((-2, 0, 1))
    conjugation = next(
        auto
        for auto in automorphisms(split.field).automorphisms
        if auto.root_permutation == (1, 0)
    )
    element = _element(split.field.extension, (Fraction(3), Fraction(2)))
    # Independent coordinate oracle uses alpha -> -alpha for alpha^2=2.
    image = apply_automorphism_to_element(conjugation, element)
    assert _coordinates(image) == (Fraction(3), Fraction(-2))
    assert image.presentation == split.field.extension

    other = _element(split.field.extension, (Fraction(1), Fraction(1)))
    # Verify multiplicativity against an independent polynomial-reduction oracle.
    source_product = _multiply(element, other, split.field.extension)
    source_product_value = _element(split.field.extension, source_product)
    mapped_product = apply_automorphism_to_element(conjugation, source_product_value)
    mapped_other = apply_automorphism_to_element(conjugation, other)
    assert _coordinates(mapped_product) == _multiply(
        image, mapped_other, split.field.extension
    )

    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "number_field.automorphism.apply_element.compute"
    )
    catalog_result = operation.run(
        AutomorphismElementApplyRequest(automorphism=conjugation, element=element)
    )
    assert _coordinates(catalog_result) == (Fraction(3), Fraction(-2))


def test_automorphism_element_application_rejects_a_different_parent() -> None:
    split = _split((-2, 0, 1))
    identity = automorphisms(split.field).automorphisms[0]
    rational_field = _split((-1, 1)).field
    rational_element = rational_field.root_values[0]
    with pytest.raises(OperationDomainValidationError, match="automorphism field"):
        apply_automorphism_to_element(identity, rational_element)


def test_automorphism_inverse_composes_to_identity_on_both_sides() -> None:
    split = _split((-2, 0, 1))
    maps = automorphisms(split.field).automorphisms
    identity = next(auto for auto in maps if auto.root_permutation == (0, 1))
    for automorphism in maps:
        inverse = inverse_automorphism(automorphism)
        assert inverse.field == automorphism.field
        assert compose_automorphisms(inverse, automorphism) == identity
        assert compose_automorphisms(automorphism, inverse) == identity
        assert (
            tuple(
                inverse.root_permutation[automorphism.root_permutation[index]]
                for index in range(len(split.roots))
            )
            == identity.root_permutation
        )
        assert (
            tuple(
                automorphism.root_permutation[inverse.root_permutation[index]]
                for index in range(len(split.roots))
            )
            == identity.root_permutation
        )
        for root in split.roots:
            assert (
                apply_automorphism(inverse, apply_automorphism(automorphism, root))
                == root
            )
            assert (
                apply_automorphism(automorphism, apply_automorphism(inverse, root))
                == root
            )


def test_degree_one_automorphism_inverse_is_identity() -> None:
    split = _split((-1, 1))
    automorphism = automorphisms(split.field).automorphisms[0]
    assert inverse_automorphism(automorphism) == automorphism


def test_automorphism_inverse_on_imaginary_quadratic_root_axis() -> None:
    split = _split((1, 0, 1))
    maps = automorphisms(split.field).automorphisms
    assert len(maps) == 2
    identity = next(auto for auto in maps if auto.root_permutation == (0, 1))
    for automorphism in maps:
        inverse = inverse_automorphism(automorphism)
        assert compose_automorphisms(inverse, automorphism) == identity
        assert compose_automorphisms(automorphism, inverse) == identity
        for root in split.roots:
            assert (
                apply_automorphism(inverse, apply_automorphism(automorphism, root))
                == root
            )


def test_split_quadratic_and_repeated_root_are_in_qq_with_multiplicity() -> None:
    split = _split((-1, 0, 1))
    assert split.field.extension.coefficients_descending == (1, 0)
    assert tuple(_coordinates(root.value) for root in split.roots) == (
        (Fraction(-1),),
        (Fraction(1),),
    )
    assert tuple(root.multiplicity for root in split.roots) == (1, 1)
    assert len(automorphisms(split.field).automorphisms) == 1

    repeated = _split((1, -2, 1))
    assert tuple(_coordinates(root.value) for root in repeated.roots) == (
        (Fraction(1),),
    )
    assert tuple(root.multiplicity for root in repeated.roots) == (2,)
    assert tuple(_coordinates(value) for value in repeated.factor_reconstruction) == (
        (Fraction(1),),
        (Fraction(-2),),
        (Fraction(1),),
    )


def test_linear_polynomial_root_is_exact_rational() -> None:
    result = _split((4, -2))
    assert result.field.extension.degree == 1
    assert _coordinates(result.roots[0].value) == (Fraction(2),)
    assert result.roots[0].multiplicity == 1


def test_exact_split_field_and_root_values_round_trip_through_json() -> None:
    result = _split((1, 0, 1))
    decoded = SplittingFieldResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    group = automorphisms(decoded.field).automorphisms
    assert len(group) == 2
    assert tuple(_coordinates(root.value) for root in decoded.roots) == (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
    )
    conjugation = next(auto for auto in group if auto.root_permutation == (1, 0))
    assert _coordinates(apply_automorphism(conjugation, decoded.roots[0]).value) == (
        Fraction(0),
        Fraction(-1),
    )


def test_nonmonic_quadratic_retains_exact_extension_and_reconstruction() -> None:
    result = _split((-3, 0, 2))
    assert result.field.extension.coefficients_descending == (2, 0, -3)
    assert tuple(_coordinates(value) for value in result.factor_reconstruction) == (
        (Fraction(-3, 2), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
    )
    for root in result.roots:
        square = _multiply(root.value, root.value, result.field.extension)
        assert tuple(2 * value for value in square) == (Fraction(3), Fraction(0))


def test_cubic_is_rejected_before_exact_extension_construction(monkeypatch) -> None:
    from jacobian.math.number_theory.galois import operations

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported cubic reached field construction")

    monkeypatch.setattr(operations, "_construct_splitting_field", forbidden)
    with pytest.raises(OperationDomainValidationError) as error:
        splitting_field(_polynomial((0, 0, 0, 1)))
    assert (
        error.value.errors()[0]["type"] == "galois_theory.splitting_field_degree_bound"
    )


def test_model_constructed_request_cannot_reach_field_construction(
    monkeypatch,
) -> None:
    from jacobian.math.number_theory.galois import operations

    forged = SplittingFieldRequest.model_construct(polynomial=_polynomial((0, 0, 0, 1)))

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported cubic reached field construction")

    monkeypatch.setattr(operations, "_construct_splitting_field", forbidden)
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "number_field.polynomial.splitting_field.compute"
    )
    with pytest.raises(OperationDomainValidationError) as error:
        operation.run(forged)
    assert (
        error.value.errors()[0]["type"] == "galois_theory.splitting_field_degree_bound"
    )


def test_nonhomomorphic_basis_image_is_rejected() -> None:
    field = _split((-2, 0, 1)).field
    one = field.extension
    from jacobian._exact import CanonicalRational
    from jacobian.math.number_theory.number_fields.values import (
        SimpleNumberFieldElement,
    )

    one_element = SimpleNumberFieldElement(
        presentation=one,
        coefficients_ascending=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    zero_element = SimpleNumberFieldElement(
        presentation=one,
        coefficients_ascending=(
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    forged = QQFieldAutomorphism.model_construct(
        field=field,
        root_permutation=(0, 1),
        basis_images=(one_element, zero_element),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        apply_automorphism(forged, _split((-2, 0, 1)).roots[0])
    assert (
        error.value.errors()[0]["type"] == "galois_theory.automorphism_not_homomorphism"
    )


def test_consumer_rejects_structurally_missing_field_presentation() -> None:
    source = _split((-2, 0, 1)).field.source
    forged = QQSplittingField.model_construct(source=source)
    with pytest.raises(OperationDomainValidationError):
        automorphisms(forged)
