"""Contract tests for the canonical rational base inclusion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldBaseEmbedding,
    FunctionFieldBaseEmbeddingApplyRequest,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_base_embedding,
    function_field_base_embedding_apply,
    function_field_element_multiply,
)


def _rf(
    numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=2, coefficients=numerator),
        denominator=PrimeFieldPolynomial(characteristic=2, coefficients=denominator),
    )


def _field() -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((0, 1)), _rf((1,)), _rf((1,))),
    )


def test_rational_base_inclusion_is_bound_to_exact_extension() -> None:
    target = _field()
    embedding = function_field_base_embedding(target).embedding
    assert embedding.source.characteristic == target.characteristic
    assert embedding.source.variable == target.variable
    assert embedding.source.degree == 1
    assert embedding.target == target
    assert embedding.variable_image.field == target
    assert embedding.variable_image.coordinates == (_rf((0, 1)), _rf((0,)))
    assert (
        FunctionFieldBaseEmbedding.model_validate(embedding.model_dump(mode="json"))
        == embedding
    )


def test_base_inclusion_rejects_rational_target() -> None:
    rational = FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((1,)),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_base_embedding(rational)
    assert error.value.errors()[0]["type"] == (
        "function_field.base_embedding_requires_extension"
    )


def test_base_embedding_value_rejects_different_rational_variable() -> None:
    target = _field()
    other_source = FiniteFunctionField(
        characteristic=2,
        variable="t",
        generator="z",
        defining_polynomial=(_rf((1,)),),
    )
    y = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=2, coefficients=(0, 1)),
        denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
    )
    element = FiniteFunctionFieldElement(field=target, coordinates=(y, _rf((0,))))
    with pytest.raises(ValidationError):
        FunctionFieldBaseEmbedding(
            source=other_source, target=target, variable_image=element
        )


def test_base_embedding_value_rejects_wrong_variable_image() -> None:
    target = _field()
    embedding = function_field_base_embedding(target).embedding
    y = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
        denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
    )
    wrong_image = FiniteFunctionFieldElement(field=target, coordinates=(_rf((0,)), y))
    with pytest.raises(ValidationError):
        FunctionFieldBaseEmbedding(
            source=embedding.source, target=target, variable_image=wrong_image
        )


def test_base_embedding_value_rejects_linear_degree_one_source() -> None:
    target = _field()
    linear_source = FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="z",
        defining_polynomial=(_rf((1,)), _rf((1,))),
    )
    element = FiniteFunctionFieldElement(
        field=target, coordinates=(_rf((0, 1)), _rf((0,)))
    )
    with pytest.raises(ValidationError) as error:
        FunctionFieldBaseEmbedding(
            source=linear_source, target=target, variable_image=element
        )
    assert error.value.errors()[0]["type"] == ("function_field.base_embedding_parent")


def test_base_embedding_value_rejects_sentinel_source_with_other_generator() -> None:
    target = _field()
    other_sentinel_source = FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="q",
        defining_polynomial=(_rf((1,)),),
    )
    element = FiniteFunctionFieldElement(
        field=target, coordinates=(_rf((0, 1)), _rf((0,)))
    )
    with pytest.raises(ValidationError) as error:
        FunctionFieldBaseEmbedding(
            source=other_sentinel_source, target=target, variable_image=element
        )
    assert error.value.errors()[0]["type"] == ("function_field.base_embedding_parent")


def test_apply_rejects_malformed_native_embedding() -> None:
    target = _field()
    embedding = function_field_base_embedding(target).embedding
    malformed_source = FiniteFunctionField.model_construct(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(
            PrimeFieldRationalFunction.model_construct(
                numerator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(0,)),
            ),
        ),
    )
    forged = FunctionFieldBaseEmbedding.model_construct(
        source=malformed_source,
        target=embedding.target,
        variable_image=embedding.variable_image,
    )
    element = FiniteFunctionFieldElement(
        field=embedding.source, coordinates=(_rf((1,)),)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_base_embedding_apply(forged, element)
    assert error.value.errors()[0]["type"] == ("function_field.invalid_base_embedding")


def test_apply_rejects_malformed_native_element() -> None:
    target = _field()
    embedding = function_field_base_embedding(target).embedding
    malformed_coordinate = PrimeFieldRationalFunction.model_construct(
        numerator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
        denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(0,)),
    )
    forged_element = FiniteFunctionFieldElement.model_construct(
        field=embedding.source, coordinates=(malformed_coordinate,)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_base_embedding_apply(embedding, forged_element)
    assert error.value.errors()[0]["type"] == "function_field.invalid_element"


def test_apply_inclusion_preserves_rational_function_and_composes() -> None:
    target = _field()
    embedding = function_field_base_embedding(target).embedding
    x_plus_one_over_x = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=2, coefficients=(1, 1)),
        denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(0, 1)),
    )
    source_element = FiniteFunctionFieldElement(
        field=embedding.source, coordinates=(x_plus_one_over_x,)
    )
    result = function_field_base_embedding_apply(embedding, source_element)
    assert result.image.field == target
    assert result.image.coordinates == (x_plus_one_over_x, _rf((0,)))
    product = function_field_element_multiply(result.image, result.image)
    assert product.product.field == target


def test_apply_inclusion_rejects_an_element_from_another_source() -> None:
    target = _field()
    embedding = function_field_base_embedding(target).embedding
    other_source = FiniteFunctionField(
        characteristic=2,
        variable="t",
        generator="z",
        defining_polynomial=(_rf((1,)),),
    )
    with pytest.raises(ValidationError):
        FunctionFieldBaseEmbeddingApplyRequest(
            embedding=embedding,
            element=FiniteFunctionFieldElement(
                field=other_source, coordinates=(_rf((1,)),)
            ),
        )


def _cubic_chain_field() -> FiniteFunctionField:
    """``f(y) = y^3 + x^12*y^2 + x`` over GF(2)(x)."""

    x12 = tuple([0] * 12 + [1])
    return FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rf((1,)),
            _rf((0,)),
            _rf(x12),
            _rf((1,)),
        ),
    )


def test_chained_generator_reduction_is_rejected_with_a_typed_error() -> None:
    """Squaring ``y^2`` reduces to a degree-24 coordinate.

    The cheap admission estimate charges the defining-coefficient degree once,
    so this product passed it at 12 and then reduced past the published
    13-coefficient schema. The result is assembled with ``model_construct``,
    so an out-of-schema value would be returned as canonical and could neither
    round-trip nor compose.
    """

    field = _cubic_chain_field()
    y_squared = FiniteFunctionFieldElement(
        field=field,
        coordinates=(_rf((0,)), _rf((0,)), _rf((1,))),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_element_multiply(y_squared, y_squared)

    assert error.value.errors()[0]["type"] == (
        "function_field.coefficient_growth_exceeds_envelope"
    )


def test_chained_denominator_reduction_is_rejected_with_a_typed_error() -> None:
    """Repeated reductions reach denominator degree 14 over GF(257)(x)."""

    characteristic = 257

    def rational_function(
        numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
    ) -> PrimeFieldRationalFunction:
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(
                characteristic=characteristic, coefficients=numerator
            ),
            denominator=PrimeFieldPolynomial(
                characteristic=characteristic, coefficients=denominator
            ),
        )

    field = FiniteFunctionField(
        characteristic=characteristic,
        variable="x",
        generator="y",
        defining_polynomial=(
            *(
                rational_function((1,), (1, (index + 8) % characteristic, 1))
                for index in range(5)
            ),
            rational_function((1,)),
        ),
    )
    y_fourth = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            *(rational_function((0,)) for _ in range(4)),
            rational_function((1,)),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_element_multiply(y_fourth, y_fourth)

    assert error.value.errors()[0]["type"] == (
        "function_field.coefficient_growth_exceeds_envelope"
    )
