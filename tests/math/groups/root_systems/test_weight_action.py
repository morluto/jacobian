import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    FiniteCartanDatum,
    WeightLatticeVector,
    WeylElement,
    WeylElementWeightActionRequest,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_datum,
    weight_lattice_vector,
    weyl_element_from_word,
)
from jacobian.math.groups.root_systems.weight_actions import weyl_element_act_on_weight
from jacobian.math.matrices.values import IntegerMatrix

_A2 = ((2, -1), (-1, 2))
_B2 = ((2, -2), (-1, 2))


def _apply_simple_reflection(weight: tuple[int, ...], cartan, index: int):
    """Independent fundamental-weight formula used as a small exact oracle."""
    root_in_fundamental_basis = tuple(cartan[row][index] for row in range(len(cartan)))
    return tuple(
        weight[row] - weight[index] * root_in_fundamental_basis[row]
        for row in range(len(cartan))
    )


def _weight_action(cartan, word, coordinates):
    element = weyl_element_from_word(cartan, word)
    vector = weight_lattice_vector(cartan, coordinates)
    return weyl_element_act_on_weight(
        WeylElementWeightActionRequest(element=element, weight=vector)
    )


def test_a2_action_matches_exact_reflection_formula_and_serializes():
    element = weyl_element_from_word(_A2, (0, 1))
    source = weight_lattice_vector(_A2, (1, 0))
    request = WeylElementWeightActionRequest(element=element, weight=source)
    result = weyl_element_act_on_weight(request)

    # Apply the defining reflection formula independently, one factor at a time.
    expected = _apply_simple_reflection((1, 0), _A2, 0)
    expected = _apply_simple_reflection(expected, _A2, 1)
    assert expected == (0, -1)
    assert result.coordinates == expected
    assert result.datum == cartan_datum(_A2)

    decoded_request = WeylElementWeightActionRequest.model_validate_json(
        request.model_dump_json()
    )
    decoded_result = WeightLatticeVector.model_validate_json(result.model_dump_json())
    assert weyl_element_act_on_weight(decoded_request) == decoded_result


def test_b2_unequal_root_lengths_use_the_exact_weight_basis_map():
    assert _apply_simple_reflection((1, 0), _B2, 0) == (-1, 1)
    assert _apply_simple_reflection((0, 1), _B2, 1) == (2, -1)
    assert _weight_action(_B2, (0,), (1, 0)).coordinates == (-1, 1)
    assert _weight_action(_B2, (1,), (0, 1)).coordinates == (2, -1)


def test_identity_reflection_composition_and_inverse_are_exact():
    source = weight_lattice_vector(_A2, (3, -2))
    identity = weyl_element_from_word(_A2, ())
    reflection = weyl_element_from_word(_A2, (0,))

    assert (
        weyl_element_act_on_weight(
            WeylElementWeightActionRequest(element=identity, weight=source)
        )
        == source
    )
    once = weyl_element_act_on_weight(
        WeylElementWeightActionRequest(element=reflection, weight=source)
    )
    twice = weyl_element_act_on_weight(
        WeylElementWeightActionRequest(element=reflection, weight=once)
    )
    assert twice == source


def test_same_rank_but_different_cartan_parent_is_rejected():
    request = WeylElementWeightActionRequest(
        element=weyl_element_from_word(_A2, (0,)),
        weight=weight_lattice_vector(_B2, (1, 0)),
    )
    with pytest.raises(
        OperationDomainValidationError, match="same ordered Cartan datum"
    ):
        weyl_element_act_on_weight(request)


def test_caller_constructed_invalid_weyl_element_is_re_admitted():
    invalid = WeylElement.model_construct(
        matrix=cartan_datum(_A2).cartan_matrix,
        root_action=IntegerMatrix(
            row_count=2,
            column_count=2,
            entries=((1, 1), (0, 1)),
        ),
    )
    request = WeylElementWeightActionRequest(
        element=invalid,
        weight=weight_lattice_vector(_A2, (1, 0)),
    )
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(request)


@pytest.mark.parametrize(
    "root_action",
    (
        object(),
        IntegerMatrix.model_construct(row_count=2, column_count=2, entries=object()),
    ),
)
def test_malformed_nested_weyl_action_is_a_domain_error(root_action):
    invalid = WeylElement.model_construct(
        matrix=cartan_datum(_A2).cartan_matrix,
        root_action=root_action,
    )
    request = WeylElementWeightActionRequest.model_construct(
        element=invalid,
        weight=weight_lattice_vector(_A2, (1, 0)),
    )
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(request)


@pytest.mark.parametrize(
    "weight",
    (
        WeightLatticeVector.model_construct(datum=object(), coordinates=(1, 0)),
        WeightLatticeVector.model_construct(
            datum=cartan_datum(_A2), coordinates=object()
        ),
        WeightLatticeVector.model_construct(
            datum=FiniteCartanDatum.model_construct(
                cartan_matrix=object(),
                symmetrizer=(),
                root_to_weight=object(),
                coroot_to_coweight=object(),
            ),
            coordinates=(1, 0),
        ),
        WeightLatticeVector.model_construct(
            datum=FiniteCartanDatum.model_construct(
                cartan_matrix=CartanMatrix.model_construct(
                    matrix=object(), simple_root_axis=(0, 1)
                ),
                symmetrizer=cartan_datum(_A2).symmetrizer,
                root_to_weight=cartan_datum(_A2).root_to_weight,
                coroot_to_coweight=cartan_datum(_A2).coroot_to_coweight,
            ),
            coordinates=(1, 0),
        ),
    ),
)
def test_malformed_nested_weight_shapes_are_domain_errors(weight):
    request = WeylElementWeightActionRequest.model_construct(
        element=weyl_element_from_word(_A2, (0,)),
        weight=weight,
    )
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(request)


def test_oversized_input_weight_preserves_resource_admission_code():
    request = WeylElementWeightActionRequest.model_construct(
        element=weyl_element_from_word(_A2, (0,)),
        weight=WeightLatticeVector.model_construct(
            datum=cartan_datum(_A2), coordinates=(1 << 200, 0)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        weyl_element_act_on_weight(request)
    assert (
        error.value.errors()[0]["type"]
        == "root_system.lattice_coordinates_over_envelope"
    )


def test_output_coordinate_growth_is_rejected_before_vector_construction(monkeypatch):
    element = weyl_element_from_word(_A2, (0,))
    source = WeightLatticeVector.model_construct(
        datum=cartan_datum(_A2),
        coordinates=((1 << 132), (1 << 132)),
    )
    request = WeylElementWeightActionRequest(element=element, weight=source)

    def construction_is_too_late(*args, **kwargs):
        pytest.fail("weight output must be admitted before result construction")

    monkeypatch.setattr(
        WeightLatticeVector, "model_construct", construction_is_too_late
    )
    with pytest.raises(OperationResourceAdmissionError):
        weyl_element_act_on_weight(request)


def test_public_catalog_operation_uses_weight_lattice_value():
    tool = Catalog.open().operation("weyl_group.element.act_on_weight.compute")
    assert tool is not None
    example_request = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    example_result = tool.run(example_request)
    assert example_result.coordinates == (-1, 1)

    result = tool.run(
        WeylElementWeightActionRequest(
            element=weyl_element_from_word(_A2, (0,)),
            weight=weight_lattice_vector(_A2, (1, 0)),
        )
    )
    assert isinstance(result, WeightLatticeVector)
    assert result.coordinates == (-1, 1)
