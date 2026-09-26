import json

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    FiniteCartanDatum,
    RootLatticeVector,
    WeylElement,
    WeylElementRootActionRequest,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_datum,
    root_lattice_vector,
    weyl_element_compose,
    weyl_element_from_word,
    weyl_element_inverse,
)
from jacobian.math.groups.root_systems.root_actions import weyl_element_act_on_root
from jacobian.math.matrices.values import IntegerMatrix

CartanRows = tuple[tuple[int, ...], ...]
_A2: CartanRows = ((2, -1), (-1, 2))
_B2: CartanRows = ((2, -2), (-1, 2))


def _datum(cartan: CartanRows) -> FiniteCartanDatum:
    return cartan_datum(CartanMatrix.model_validate(cartan))


def _reflect_root_vector(
    vector: tuple[int, ...], cartan: CartanRows, index: int
) -> tuple[int, ...]:
    """Independent application of s_i(v)=v-<alpha_i^vee,v> alpha_i."""
    pairing = sum(
        cartan[index][column] * vector[column] for column in range(len(vector))
    )
    return tuple(
        coordinate - pairing if coordinate_index == index else coordinate
        for coordinate_index, coordinate in enumerate(vector)
    )


def _apply_word(
    vector: tuple[int, ...], cartan: CartanRows, word: tuple[int, ...]
) -> tuple[int, ...]:
    image = vector
    for index in word:
        image = _reflect_root_vector(image, cartan, index)
    return image


def _act(element: WeylElement, vector: RootLatticeVector) -> RootLatticeVector:
    return weyl_element_act_on_root(element, vector)


@pytest.mark.parametrize(
    ("cartan", "word", "coordinates"),
    (
        (_A2, (0, 1, 0), (2, -1)),
        (_B2, (1, 0, 1), (1, 2)),
    ),
)
def test_element_action_matches_independent_simple_reflection_formula(
    cartan: CartanRows, word: tuple[int, ...], coordinates: tuple[int, ...]
) -> None:
    element = weyl_element_from_word(cartan, word)
    source = root_lattice_vector(cartan, coordinates)
    request = WeylElementRootActionRequest(element=element, vector=source)

    result = weyl_element_act_on_root(request.element, request.vector)

    assert result.coordinates == _apply_word(coordinates, cartan, word)
    assert result.datum == _datum(cartan)
    decoded_request = WeylElementRootActionRequest.model_validate_json(
        request.model_dump_json()
    )
    decoded_result = RootLatticeVector.model_validate_json(result.model_dump_json())
    assert (
        weyl_element_act_on_root(decoded_request.element, decoded_request.vector)
        == decoded_result
    )


def test_identity_composition_and_inverse_obey_the_action_law() -> None:
    source = root_lattice_vector(_A2, (3, -2))
    first = weyl_element_from_word(_A2, (0, 1))
    then = weyl_element_from_word(_A2, (1, 0))
    identity = weyl_element_from_word(_A2, ())

    assert _act(identity, source) == source
    composed = weyl_element_compose(first, then)
    sequential = _act(then, _act(first, source))
    assert _act(composed, source) == sequential
    assert _act(weyl_element_inverse(first), _act(first, source)) == source


def test_same_rank_different_ordered_cartan_parent_is_rejected() -> None:
    request = WeylElementRootActionRequest(
        element=weyl_element_from_word(_A2, (0,)),
        vector=root_lattice_vector(_B2, (1, 0)),
    )
    with pytest.raises(
        OperationDomainValidationError, match="same ordered Cartan datum"
    ):
        weyl_element_act_on_root(request.element, request.vector)


def test_non_weyl_root_system_automorphism_is_rejected() -> None:
    swap = WeylElement.model_construct(
        matrix=_datum(_A2).cartan_matrix,
        root_action=IntegerMatrix(
            row_count=2,
            column_count=2,
            entries=((0, 1), (1, 0)),
        ),
    )
    request = WeylElementRootActionRequest.model_construct(
        element=swap,
        vector=root_lattice_vector(_A2, (1, 0)),
    )
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_root(request.element, request.vector)


def test_carrier_boundary_and_cancellation_are_admitted_before_result_creation() -> (
    None
):
    element = weyl_element_from_word(_A2, (0,))
    datum = _datum(_A2)
    large = 1 << 132  # The 133-bit canonical vector carrier boundary.
    source = RootLatticeVector.model_construct(datum=datum, coordinates=(large, large))

    # The first coordinate cancels exactly; admission uses the bounded exact
    # intermediate size and checks the actual output coordinates.
    assert _act(element, source).coordinates == (0, large)

    overflowing = RootLatticeVector.model_construct(
        datum=datum, coordinates=(large, -large)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        _act(element, overflowing)
    assert error.value.errors()[0]["type"] == "root_system.weyl_root_action_image_bound"


def test_oversized_caller_constructed_vector_is_resource_rejected() -> None:
    vector = RootLatticeVector.model_construct(
        datum=_datum(_A2), coordinates=(1 << 200, 0)
    )
    request = WeylElementRootActionRequest.model_construct(
        element=weyl_element_from_word(_A2, ()), vector=vector
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        weyl_element_act_on_root(request.element, request.vector)
    assert (
        error.value.errors()[0]["type"]
        == "root_system.lattice_coordinates_over_envelope"
    )


@pytest.mark.parametrize("num, den", [(True, 1), (1, True)])
def test_caller_constructed_vector_with_malformed_symmetrizer_is_rejected(num, den):
    datum = _datum(_A2)
    malformed = FiniteCartanDatum.model_construct(
        cartan_matrix=datum.cartan_matrix,
        symmetrizer=(
            CanonicalRational.model_construct(num=num, den=den),
            datum.symmetrizer[1],
        ),
        root_to_weight=datum.root_to_weight,
        coroot_to_coweight=datum.coroot_to_coweight,
    )
    vector = RootLatticeVector.model_construct(datum=malformed, coordinates=(1, 0))
    with pytest.raises(OperationDomainValidationError) as error:
        weyl_element_act_on_root(weyl_element_from_word(_A2, ()), vector)
    assert error.value.errors()[0]["type"] == "root_system.invalid_lattice_vector_datum"


def test_caller_constructed_vector_with_noncanonical_cartan_axis_is_rejected() -> None:
    datum = _datum(_A2)
    malformed_cartan = CartanMatrix.model_construct(
        matrix=datum.cartan_matrix.matrix, simple_root_axis=(1, 0)
    )
    malformed_datum = FiniteCartanDatum.model_construct(
        cartan_matrix=malformed_cartan,
        symmetrizer=datum.symmetrizer,
        root_to_weight=datum.root_to_weight,
        coroot_to_coweight=datum.coroot_to_coweight,
    )
    vector = RootLatticeVector.model_construct(
        datum=malformed_datum, coordinates=(1, 0)
    )
    request = WeylElementRootActionRequest.model_construct(
        element=weyl_element_from_word(_A2, ()), vector=vector
    )

    with pytest.raises(
        OperationDomainValidationError, match="canonical ordered Cartan axis"
    ):
        weyl_element_act_on_root(request.element, request.vector)


def test_caller_constructed_vector_with_wrong_axis_length_is_rejected() -> None:
    vector = RootLatticeVector.model_construct(datum=_datum(_A2), coordinates=(1,))
    request = WeylElementRootActionRequest.model_construct(
        element=weyl_element_from_word(_A2, ()), vector=vector
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="root-lattice coordinates must be a bounded integer tuple",
    ):
        weyl_element_act_on_root(request.element, request.vector)


def test_public_catalog_operation_returns_the_canonical_vector_value() -> None:
    tool = Catalog.open().operation("weyl_group.element.act_on_root.compute")
    assert tool is not None
    example = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(example)
    assert isinstance(result, RootLatticeVector)
    assert result.coordinates == (-1, 0)
    assert result.datum == example.vector.datum
