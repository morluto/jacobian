import pytest

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
    cartan_matrix_from_type,
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
    return weyl_element_act_on_weight(element, vector)


def test_a2_action_matches_exact_reflection_formula_and_serializes():
    element = weyl_element_from_word(_A2, (0, 1))
    source = weight_lattice_vector(_A2, (1, 0))
    request = WeylElementWeightActionRequest(element=element, weight=source)
    result = weyl_element_act_on_weight(element, source)

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
    assert (
        weyl_element_act_on_weight(decoded_request.element, decoded_request.weight)
        == decoded_result
    )


def test_b2_unequal_root_lengths_use_the_exact_weight_basis_map():
    assert _apply_simple_reflection((1, 0), _B2, 0) == (-1, 1)
    assert _apply_simple_reflection((0, 1), _B2, 1) == (2, -1)
    assert _weight_action(_B2, (0,), (1, 0)).coordinates == (-1, 1)
    assert _weight_action(_B2, (1,), (0, 1)).coordinates == (2, -1)


def test_identity_reflection_composition_and_inverse_are_exact():
    source = weight_lattice_vector(_A2, (3, -2))
    identity = weyl_element_from_word(_A2, ())
    reflection = weyl_element_from_word(_A2, (0,))

    assert weyl_element_act_on_weight(identity, source) == source
    once = weyl_element_act_on_weight(reflection, source)
    twice = weyl_element_act_on_weight(reflection, once)
    assert twice == source


def test_same_rank_but_different_cartan_parent_is_rejected():
    element = weyl_element_from_word(_A2, (0,))
    weight = weight_lattice_vector(_B2, (1, 0))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        weyl_element_act_on_weight(element, weight)
    assert (
        exc_info.value.errors()[0]["type"] == "root_system.weyl_weight_parent_mismatch"
    )


def test_caller_constructed_invalid_weyl_element_is_re_admitted():
    invalid = WeylElement.model_construct(
        matrix=cartan_datum(_A2).cartan_matrix,
        root_action=IntegerMatrix(
            row_count=2,
            column_count=2,
            entries=((1, 1), (0, 1)),
        ),
    )
    weight = weight_lattice_vector(_A2, (1, 0))
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(invalid, weight)


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
    weight = weight_lattice_vector(_A2, (1, 0))
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(invalid, weight)


def test_huge_caller_constructed_cartan_entry_is_rejected_before_rendering():
    huge = 1 << 1_000_000
    matrix = IntegerMatrix.model_construct(
        row_count=2, column_count=2, entries=((2, huge), (-1, 2))
    )
    invalid = WeylElement.model_construct(
        matrix=CartanMatrix.model_construct(matrix=matrix, simple_root_axis=(0, 1)),
        root_action=weyl_element_from_word(_A2, ()).root_action,
    )

    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(invalid, weight_lattice_vector(_A2, (1, 0)))


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
    element = weyl_element_from_word(_A2, (0,))
    with pytest.raises(OperationDomainValidationError):
        weyl_element_act_on_weight(element, weight)


def test_oversized_input_weight_preserves_resource_admission_code():
    element = weyl_element_from_word(_A2, (0,))
    weight = WeightLatticeVector.model_construct(
        datum=cartan_datum(_A2), coordinates=(1 << 200, 0)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        weyl_element_act_on_weight(element, weight)
    assert (
        error.value.errors()[0]["type"]
        == "root_system.lattice_coordinates_over_envelope"
    )


def test_cancellation_keeps_admissible_output_coordinates():
    element = weyl_element_from_word(_A2, (0,))
    source = WeightLatticeVector.model_construct(
        datum=cartan_datum(_A2),
        coordinates=((1 << 132), -(1 << 132)),
    )
    result = weyl_element_act_on_weight(element, source)
    assert result.coordinates == (-(1 << 132), 0)


def test_rank_eight_fraction_preflight_accepts_identity_action():
    e8 = cartan_matrix_from_type("E", 8).matrix
    source = weight_lattice_vector(e8, (1, 0, 0, 0, 0, 0, 0, 0))
    result = weyl_element_act_on_weight(weyl_element_from_word(e8, ()), source)
    assert result == source
