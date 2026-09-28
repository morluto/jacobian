"""Catalog dispatch coverage relocated out of the owner-local math tests.

The math test lanes must not boot complete product boundaries, so operations
that callers reach through the published catalog are asserted here instead of
from a domain test module.
"""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from math import gcd

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation, parse_operation_input
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryMatrixResult,
    BinarySymmetricMatrix,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import (
    polytopal_complex_affine_transform,
    polytopal_complex_closure,
)
from jacobian.math.geometry.polytopes.complexes import (
    polytopal_complex_affine_transform as exported_affine_transform,
)
from jacobian.math.geometry.polytopes.complexes import spline_dimension
from jacobian.math.geometry.polytopes.complexes._models import (
    PiecewisePolynomialResult,
    SplineDimensionRequest,
    SplineDimensionResult,
)
from jacobian.math.logic.automata.transducers.values import SubsequentialTransducer
from jacobian.math.matrices.cyclic_linear import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms import (
    cyclotomic,
    modular_character_coordinates_u_prime,
    modular_form_space_inclusion,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterUPrimeRequest,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.character_basis import (
    _character_sturm_precision,
    modular_character_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
    ModularFormSpaceInclusion,
)
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.quantum import (
    CheckSpaceValue,
    ExactQubitPauli,
    LogicalPauliFrame,
    PhaseFreeQubitPauli,
    QubitRegister,
    css_exact_distance,
    css_logical_pauli_frame,
)
from jacobian.math.quantum._models import ExactStabilizerGroup


_FIELD = RationalCyclotomicField(order=6)


_GENERIC_BASIS = "gamma0-cyclotomic-character-sturm-rref-v1"


_EXPECTED = {
    (2, 26, 2): (((0, 0), (0, -2)), ((1, 0), (-1, -1))),
    (10, 26, 2): (((0, 0), (-2, 2)), ((1, 0), (-2, 1))),
    (2, 26, 13): (((-1, -3), (0, 0)), ((0, 0), (-1, -3))),
    (10, 26, 13): (((-4, 3), (0, 0)), ((0, 0), (-4, 3))),
    (2, 39, 3): (
        ((0, 0), (2, -1), (0, -3)),
        ((0, 0), (1, -1), (-2, -2)),
        ((1, 0), (-1, -1), (-2, 2)),
    ),
    (10, 39, 3): (
        ((0, 0), (1, 1), (-3, 3)),
        ((0, 0), (0, 1), (-4, 2)),
        ((1, 0), (-2, 1), (0, -2)),
    ),
    (2, 39, 13): (
        ((4, -1), (0, 0), (7, -5)),
        ((4, -1), (-1, -3), (3, -4)),
        ((0, 0), (0, 0), (-1, -3)),
    ),
    (10, 39, 13): (
        ((3, 1), (0, 0), (2, 5)),
        ((3, 1), (-4, 3), (-1, 4)),
        ((0, 0), (0, 0), (-4, 3)),
    ),
}


def _space(coordinate: int, level: int) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind="S",
        character=_inflated_character(coordinate, level),
        coefficient_domain=_FIELD,
    )


def _unit(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            {"num": 1, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


def _zero(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            {"num": 0, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


def _coordinates(space: ModularFormSpace, index: int) -> ModularFormCoordinates:
    basis = modular_character_basis_q_expansions(space)
    return ModularFormCoordinates(
        space=space,
        basis_id=_GENERIC_BASIS,
        coordinates=tuple(
            _unit(_FIELD) if coordinate == index else _zero(_FIELD)
            for coordinate in range(len(basis.elements))
        ),
    )


def _inflated_character(coordinate: int, level: int):
    source = dirichlet_character(character_group(13), (coordinate,))
    group = character_group(level)
    for candidate_coordinates in product(
        *(range(order) for order in group.generator_orders)
    ):
        candidate = dirichlet_character(group, candidate_coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source, residue).value
            for residue in range(level)
            if gcd(residue, level) == 1
        ):
            return candidate
    raise AssertionError("the exact character inflation must exist")


# --- relocated from tests/math/combinatorics/matroids/delta/test_binary_matrix_twist.py


def test_zero_matrix_twist_example_composes_through_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("delta_matroid.binary.from_matrix_twist.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    decoded = BinaryMatrixResult.model_validate_json(json.dumps(result.output))
    assert decoded.matrix.entries == ((0, 0), (0, 0))
    assert decoded.twist == (0, 1)
    assert decoded.delta_matroid.feasible == ((0, 1),)


# --- relocated from tests/math/combinatorics/matroids/delta/test_distance_profile.py


def test_published_operation_example_runs_through_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("delta_matroid.distance_profile.compute")
    assert operation is not None

    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    assert result.output["distance_by_mask"] == [0, 0, 0, 0]
    assert result.output["nearest_feasible_count_by_mask"] == [1, 1, 1, 1]
    assert result.output["distance_histogram"] == [4, 0, 0]


# --- relocated from tests/math/geometry/polytopes/test_common_refinement.py


def test_common_refinement_catalog_example_is_composable():
    catalog = Catalog.open()
    operation = catalog.operation("polytopal_complex.common_refinement.compute")
    example = operation.examples[0]
    result = invoke_operation(operation.operation_id, example.input, catalog)
    assert result.output["cell_pairs"][0]["refined_cell_id"] == "M0"


# --- relocated from tests/math/geometry/polytopes/test_complex_affine_transform.py


def test_affine_transform_catalog_example_roundtrips():
    catalog = Catalog.open()
    operation = catalog.operation("polytopal_complex.affine_transform.compute")
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    target = result.output["target"]
    assert target["maximal_cells"][0]["vertices"] == [
        {"coordinates": [{"num": "3", "den": "1"}]},
        {"coordinates": [{"num": "5", "den": "1"}]},
    ]
    assert len(result.output["cell_transport"]) == 1
    assert exported_affine_transform is polytopal_complex_affine_transform


# --- relocated from tests/math/geometry/polytopes/test_face_lattice.py


_OPERATION_ID = "polytope.face_lattice.compute"



def test_face_lattice_operation_is_catalogued_and_runs_its_example() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_OPERATION_ID)
    assert operation is not None
    result = invoke_operation(_OPERATION_ID, operation.examples[0].input, catalog)
    assert result.output["faces"]
    assert len(result.output["covers"]) == 32


# --- relocated from tests/math/geometry/polytopes/test_piecewise_polynomial_addition.py


def _coefficient_map(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }



def test_catalog_addition_example_executes_through_typed_contract():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.add.compute")
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert _coefficient_map(validated.pieces[0].polynomial) == {(0,): Fraction(3)}


# --- relocated from tests/math/geometry/polytopes/test_piecewise_polynomial_multiply.py


def _coefficient_map(polynomial: RationalPolynomial):
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }



def test_catalog_piecewise_multiplication_example_executes():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.multiply.compute")
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert _coefficient_map(validated.pieces[0].polynomial) == {
        (2,): Fraction(1),
        (1,): Fraction(1),
    }


# --- relocated from tests/math/geometry/polytopes/test_spline_dimension.py


def _interval(left: int, right: int, prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=(CanonicalRational(num=value, den=1),),
            )
            for index, value in enumerate((left, right))
        ),
    )



def test_one_cell_zero_row_dimension_roundtrips_and_catalog_invokes():
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    request = SplineDimensionRequest(complex=complex_value, degree=5, smoothness=1)

    result = spline_dimension(request)
    replayed = SplineDimensionResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )

    assert result.compatibility_matrix.row_count == 0
    assert result.compatibility_matrix.column_count == 6
    assert result.rank == 0 and result.nullity == 6
    assert replayed == result
    invoked = invoke_operation(
        "polyhedral_complex.spline_dimension.compute",
        {
            "complex": complex_value.model_dump(mode="json"),
            "degree": 0,
            "smoothness": 0,
        },
        Catalog.open(),
    )
    assert invoked.output["nullity"] == 1 and invoked.output["rank"] == 0


# --- relocated from tests/math/graphs/test_anonymous_graph_card_multiset.py


def test_forged_catalog_request_missing_card_order_is_a_domain_error() -> None:
    tool = Catalog.open().operation("graph.deck.from_cards.construct")
    assert tool is not None
    forged = tool.request_type.model_construct(cards=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tool.run(forged)
    assert exc_info.value.errors()[0]["type"] == "graph_deck.anonymous_order_invalid"



def test_catalog_publishes_anonymous_cards_as_distinct_from_realizable_decks() -> None:
    tool = Catalog.open().operation("graph.deck.from_cards.construct")
    assert tool is not None
    result = tool.run(
        tool.request_type.model_validate(
            {"card_order": 1, "cards": [{"vertices": ["x"], "edges": []}]}
        )
    )
    assert result.card_order == 1
    assert result.classes[0].multiplicity == 1


# --- relocated from tests/math/graphs/test_source_bound_anonymous_deck.py


def test_operation_is_published_with_typed_output_and_equality_example() -> None:
    declaration = Catalog.open().operation("graph.deck.vertex.anonymous.compute")
    assert declaration is not None
    assert declaration.result_type.__name__ == "AnonymousGraphCardMultiset"
    assert declaration.examples


# --- relocated from tests/math/logic/automata/transducers/test_domain_restriction.py


def test_tool_is_discoverable_and_has_a_valid_example() -> None:
    operation_id = "transducer.subsequential.restrict_domain.compute"
    catalog = Catalog.open()
    tool = catalog.operation(operation_id)
    assert tool is not None
    assert len(tool.examples) == 1
    result = invoke_operation(operation_id, tool.examples[0].input, catalog)
    restricted = tool.result_type.model_validate_json(json.dumps(result.output))
    assert isinstance(restricted, SubsequentialTransducer)
    assert restricted.state_count == 2
    assert len(restricted.final_outputs) == 1
    assert restricted.final_outputs[0].state == 0
    assert restricted.final_outputs[0].output == ()


# --- relocated from tests/math/number_theory/modular_forms/test_character_u_prime.py


_GENERIC_BASIS = "gamma0-cyclotomic-character-sturm-rref-v1"



_EXPECTED = {
    (2, 26, 2): (((0, 0), (0, -2)), ((1, 0), (-1, -1))),
    (10, 26, 2): (((0, 0), (-2, 2)), ((1, 0), (-2, 1))),
    (2, 26, 13): (((-1, -3), (0, 0)), ((0, 0), (-1, -3))),
    (10, 26, 13): (((-4, 3), (0, 0)), ((0, 0), (-4, 3))),
    (2, 39, 3): (
        ((0, 0), (2, -1), (0, -3)),
        ((0, 0), (1, -1), (-2, -2)),
        ((1, 0), (-1, -1), (-2, 2)),
    ),
    (10, 39, 3): (
        ((0, 0), (1, 1), (-3, 3)),
        ((0, 0), (0, 1), (-4, 2)),
        ((1, 0), (-2, 1), (0, -2)),
    ),
    (2, 39, 13): (
        ((4, -1), (0, 0), (7, -5)),
        ((4, -1), (-1, -3), (3, -4)),
        ((0, 0), (0, 0), (-1, -3)),
    ),
    (10, 39, 13): (
        ((3, 1), (0, 0), (2, 5)),
        ((3, 1), (-4, 3), (-1, 4)),
        ((0, 0), (0, 0), (-4, 3)),
    ),
}



def _space(coordinate: int, level: int) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind="S",
        character=_inflated_character(coordinate, level),
        coefficient_domain=_FIELD,
    )



def _numerators(value: RationalCyclotomicElement) -> tuple[int, ...]:
    return tuple(int(coefficient.num) for coefficient in value.coefficients_ascending)



@pytest.mark.parametrize("character_coordinate,level,prime", tuple(_EXPECTED))
def test_u_prime_matches_independent_q_prefix_action_and_target_coordinates(
    character_coordinate: int, level: int, prime: int
) -> None:
    space = _space(character_coordinate, level)
    sturm_precision = _character_sturm_precision(space)
    source_precision = prime * (sturm_precision - 1) + 1
    source_basis = modular_character_basis_q_expansions(space, source_precision)
    target_basis = modular_character_basis_q_expansions(space)
    result_columns = []
    for basis_index in range(len(target_basis.elements)):
        form = _coordinates(space, basis_index)
        result = modular_character_coordinates_u_prime(form, prime)
        assert result.space == space
        assert result.basis_id == _GENERIC_BASIS
        assert (
            TypeAdapter(ModularFormCoordinates).validate_json(result.model_dump_json())
            == result
        )
        result_columns.append(tuple(_numerators(value) for value in result.coordinates))

        source_coefficients = source_basis.elements[basis_index].expansion.coefficients
        expected_prefix = tuple(
            source_coefficients[prime * exponent] for exponent in range(sturm_precision)
        )
        # Expand the returned coordinates using the public exact target basis.
        target_prefix = tuple(
            _sum(
                cyclotomic.multiply(
                    result.coordinates[row],
                    target_basis.elements[row].expansion.coefficients[exponent],
                )
                for row in range(len(result.coordinates))
            )
            for exponent in range(sturm_precision)
        )
        assert target_prefix == expected_prefix

    assert tuple(result_columns) == _EXPECTED[(character_coordinate, level, prime)]
    tool = Catalog.open().operation("modular_form.character_coordinates.u_prime.apply")
    assert "order-six" in tool.description
    request = ModularCharacterUPrimeRequest(form=_coordinates(space, 0), prime=prime)
    assert tool.run(request).space == space



def _sum(values):
    result = _zero(_FIELD)
    for value in values:
        result = cyclotomic.add(result, value)
    return result


# --- relocated from tests/math/number_theory/modular_forms/test_coordinate_transport.py


def _transport_coordinates(
    level: int, weight: int, kind: str, basis_id: str, values: tuple[int, ...]
) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=level, weight=weight, kind=kind),  # type: ignore[arg-type]
        basis_id=basis_id,  # type: ignore[arg-type]
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in values),
    )



def test_transport_catalog_operation_round_trips_target_coordinates() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.transport.compute"
    )
    source = _transport_coordinates(1, 4, "M", "level-one-e4-e6-monomials-v1", (1,))
    inclusion_tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.space.inclusion.compute"
    )
    catalog = Catalog.open()
    inclusion_result = invoke_operation(
        inclusion_tool.operation_id,
        {
            "source_space": source.space.model_dump(mode="json"),
            "target_space": {"level": 2, "weight": 4, "kind": "M"},
        },
        catalog,
    )
    decoded = ModularFormSpaceInclusion.model_validate_json(
        encode_strict_json(inclusion_result.output)
    )
    assert decoded == modular_form_space_inclusion(
        source.space, ModularFormSpace(level=2, weight=4, kind="M")
    )

    result = invoke_operation(
        tool.operation_id,
        {"form": source.model_dump(mode="json"), "inclusion": inclusion_result.output},
        catalog,
    )

    assert result.output["space"]["level"] == 2
    assert (
        ModularFormCoordinates.model_validate_json(
            encode_strict_json(result.output)
        ).space.level
        == 2
    )


# --- relocated from tests/math/polynomials/test_local_series_newton_polygon.py


def test_newton_edge_roots_declared_catalog_example_executes() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    operation = Catalog.open().operation(
        "local_series.polynomial.newton_edge_characteristic_roots.compute"
    )
    assert operation is not None
    example = operation.examples[0]
    result = invoke_operation(operation.operation_id, example.input, Catalog.open())
    assert result.output["characteristic"]["characteristic_polynomial"]["polynomial"]
    roots = result.output["roots"]
    assert len(roots) == 2
    assert all(root["value"]["polynomial"] == ["1", "0", "-2"] for root in roots)


# --- relocated from tests/math/quantum/test_css_check_space.py


def test_catalog_example_serializes_into_existing_stabilizer_consumers() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.quantum import (
        CheckSpaceValue,
        PhaseFreeQubitPauli,
        stabilizer_error_equivalence,
        stabilizer_syndrome,
    )

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.css_check_space.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["witness"] is None
    from jacobian.math.quantum import CSSCheckSpaceValue

    css_value = CSSCheckSpaceValue.model_validate(result.output["css_check_space"])
    check_space = CheckSpaceValue.model_validate(css_value.check_space.model_dump())
    error = PhaseFreeQubitPauli(
        register=check_space.qubit_register,
        x_bits=(1, 0, 0),
        z_bits=(0, 0, 0),
    )
    assert stabilizer_syndrome(check_space, error).syndrome == (0, 1)
    assert stabilizer_error_equivalence(
        check_space, error, error
    ).equivalent_mod_stabilizers



def test_catalog_css_logical_frame_and_steane_parameter_fixture() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.quantum import (
        CSSCheckSpaceValue,
        CSSLogicalPauliFrame,
        css_check_space,
    )

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.css_logical_frame.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    frame = CSSLogicalPauliFrame.model_validate(result.output)
    assert frame.logical_qubits == 2
    distance_operation = catalog.operation("quantum.stabilizer.css_distance.compute")
    assert distance_operation is not None
    distance_result = invoke_operation(
        distance_operation.operation_id,
        distance_operation.examples[0].input,
        catalog,
    )
    from jacobian.math.quantum import CSSDistanceResult

    distance = CSSDistanceResult.model_validate(distance_result.output)
    assert distance.logical_qubits == 2
    assert distance.x_distance == distance.z_distance == 2

    # The standard seven-qubit Hamming CSS checks have one logical qubit.
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(7)))
    hamming_rows = (
        (1, 0, 1, 0, 1, 0, 1),
        (0, 1, 1, 0, 0, 1, 1),
        (0, 0, 0, 1, 1, 1, 1),
    )
    css = css_check_space(register, hamming_rows, hamming_rows).css_check_space
    assert css is not None
    steane_frame = css_logical_pauli_frame(
        CSSCheckSpaceValue.model_validate(css.model_dump())
    )
    assert steane_frame.logical_qubits == 1
    assert (
        sum(
            x * z
            for x, z in zip(
                steane_frame.x_logical_basis[0].x_bits,
                steane_frame.z_logical_basis[0].z_bits,
                strict=True,
            )
        )
        % 2
        == 1
    )
    distance = css_exact_distance(steane_frame.css_check_space)
    assert distance.x_distance == distance.z_distance == 3
    assert distance.x_representative is not None
    assert distance.z_representative is not None
    assert distance.x_representative.weight == distance.z_representative.weight == 3


# --- relocated from tests/math/quantum/test_exact_stabilizer_group.py


def _pauli(
    register: QubitRegister,
    x_bits: tuple[int, ...],
    z_bits: tuple[int, ...],
    phase: int,
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x_bits, z_bits=z_bits),
        phase=phase,
    )



def test_exact_group_is_available_through_catalog_dispatch() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation_id = "quantum.stabilizer.exact_group.from_generators.compute"
    assert catalog.operation(operation_id) is not None
    operation_result = invoke_operation(
        operation_id,
        {
            "register": {"qubit_ids": ["q"]},
            "generators": [
                {
                    "phase_free": {
                        "register": {"qubit_ids": ["q"]},
                        "x_bits": [0],
                        "z_bits": [1],
                    },
                    "phase": 0,
                }
            ],
        },
        catalog,
    )
    result = ExactStabilizerGroup.model_validate(operation_result.output)
    assert result.generators == (
        _pauli(QubitRegister(qubit_ids=("q",)), (0,), (1,), 0),
    )


# --- relocated from tests/math/quantum/test_logical_frame.py


def test_catalog_publishes_generic_mixed_pauli_logical_frame() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.logical_frame.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    frame = LogicalPauliFrame.model_validate(result.output)
    assert frame.logical_qubits == 1
    assert frame.x_logical_basis[0].qubit_register == frame.check_space.qubit_register
    assert frame.z_logical_basis[0].qubit_register == frame.check_space.qubit_register
    assert LogicalPauliFrame.model_validate(frame.model_dump(mode="json")) == frame

    # Deserialization is structural only: the kernel establishes S-perp
    # membership and canonical pairings once, so model_validate must not replay
    # that GF(2) work. Structural violations are still rejected here.
    bad_register = frame.model_dump(mode="json")
    bad_register["x_logical_basis"][0]["qubit_register"] = {"qubit_ids": ["other"]}
    with pytest.raises(ValidationError, match="register"):
        LogicalPauliFrame.model_validate(bad_register)

    bad_dimension = frame.model_dump(mode="json")
    bad_dimension["x_logical_basis"] = []
    with pytest.raises(ValidationError, match="size k"):
        LogicalPauliFrame.model_validate(bad_dimension)


# --- relocated from tests/math/quantum/test_pauli_labels.py


def test_label_conversion_catalog_round_trip_preserves_y_scalar() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    from_labels = catalog.operation("quantum.pauli.qubit.from_labels.compute")
    to_labels = catalog.operation("quantum.pauli.qubit.to_labels.compute")
    assert from_labels is not None and to_labels is not None
    constructed = invoke_operation(
        from_labels.operation_id,
        {"register": {"qubit_ids": ["a", "b"]}, "labels": ["Y", "Z"], "phase": 2},
        catalog,
    )
    decoded = invoke_operation(
        to_labels.operation_id, {"pauli": constructed.output["pauli"]}, catalog
    ).output
    assert decoded["labels"] == ["Y", "Z"]
    assert decoded["phase"] == 2


# --- relocated from tests/math/quantum/test_stabilizer_code.py


def test_code_value_is_published_and_roundtrips_through_catalog() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation_id = "quantum.stabilizer.code.compute"
    assert catalog.operation(operation_id) is not None
    raw = invoke_operation(
        operation_id,
        {
            "group": {
                "register": {"qubit_ids": ["q"]},
                "generators": [
                    {
                        "phase_free": {
                            "register": {"qubit_ids": ["q"]},
                            "x_bits": [0],
                            "z_bits": [1],
                        },
                        "phase": 0,
                    }
                ],
            },
            "generator_eigenvalues": [-1],
        },
        catalog,
    )
    from jacobian.math.quantum._models import StabilizerCodeValue

    value = StabilizerCodeValue.model_validate(raw.output)
    assert value.logical_qubits == 0
    assert value.group.generators[0].phase == 2


# --- relocated from tests/math/quantum/test_stabilizer_distance.py


def test_catalog_example_runs_and_returns_source_bound_result() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.distance.compute")
    assert operation is not None
    output = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    ).output
    assert output["logical_qubits"] == 1
    assert output["distance"] == 1
    assert (
        output["representative"]["qubit_register"]
        == output["check_space"]["qubit_register"]
    )



# --- relocated from tests/math/geometry/crystallographic/extensions/test_polytope_facet_pairings.py


def test_pairing_operation_is_published_with_square_example() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "crystallographic.extension.polytope_facet_pairings.compute"
    )

    assert tool.examples[0].name == "unit_square_translation_pairings"
    example_request = parse_operation_input(tool.request_type, tool.examples[0].input)
    assert tool.run(example_request).facet_profile.dimension == 2
