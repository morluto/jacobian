"""Exact primitive-positive formula semantics on finite structures."""

from __future__ import annotations

import itertools
import json

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    PPDefinedRelation,
    PPFormulaEvaluationRequest,
    PrimitivePositiveFormula,
    evaluate_pp_formula,
    search_homomorphism,
)


def _structure(
    carrier_size: int,
    edges: tuple[tuple[int, int], ...],
    *,
    proposition: bool = False,
) -> FiniteRelationalStructure:
    signature = [FiniteRelationSymbol(symbol_id="E", arity=2)]
    tables: list[tuple[tuple[int, ...], ...]] = [edges]
    if proposition:
        signature.append(FiniteRelationSymbol(symbol_id="P", arity=0))
        tables.append(((),))
    return FiniteRelationalStructure(
        carrier_size=carrier_size,
        signature=tuple(signature),
        relation_tables=tuple(tables),
    )


def test_two_step_formula_returns_exact_ordered_relation_and_roundtrips() -> None:
    structure = _structure(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
    formula = PrimitivePositiveFormula(
        variable_count=3,
        free_variables=(2, 0),
        atoms=(
            {"kind": "relation", "symbol_id": "E", "variables": (0, 1)},
            {"kind": "relation", "symbol_id": "E", "variables": (1, 2)},
        ),
    )

    result = evaluate_pp_formula(structure, formula)

    assert result.tuples == ((2, 0), (3, 1))
    restored = PPDefinedRelation.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.formula.free_variables == (2, 0)


def test_formula_atoms_normalize_but_free_variable_axis_order_is_preserved() -> None:
    structure = _structure(4, ((0, 1), (1, 2), (2, 3), (3, 0)))
    first = PrimitivePositiveFormula(
        variable_count=3,
        free_variables=(2, 0),
        atoms=(
            {"kind": "relation", "symbol_id": "E", "variables": (0, 1)},
            {"kind": "equality", "left": 2, "right": 1},
            {"kind": "relation", "symbol_id": "E", "variables": (0, 1)},
            {"kind": "equality", "left": 1, "right": 2},
        ),
    )
    equivalent = PrimitivePositiveFormula(
        variable_count=3,
        free_variables=(2, 0),
        atoms=(
            {"kind": "equality", "left": 1, "right": 2},
            {"kind": "relation", "symbol_id": "E", "variables": (0, 1)},
        ),
    )

    assert first == equivalent
    assert first.model_dump_json() == equivalent.model_dump_json()
    assert len(first.atoms) == 2
    assert first.atoms[0].kind == "equality"
    assert first.atoms[0].left == 1
    assert first.atoms[0].right == 2
    assert first.free_variables == (2, 0)
    assert (
        evaluate_pp_formula(structure, first).tuples
        == evaluate_pp_formula(structure, equivalent).tuples
    )


def test_equality_nullary_atoms_and_empty_carrier_semantics() -> None:
    structure = _structure(2, ((0, 1),), proposition=True)
    same_endpoint = PrimitivePositiveFormula(
        variable_count=2,
        free_variables=(0,),
        atoms=(
            {"kind": "equality", "left": 0, "right": 1},
            {"kind": "relation", "symbol_id": "P", "variables": ()},
        ),
    )
    assert evaluate_pp_formula(structure, same_endpoint).tuples == ((0,), (1,))

    empty = FiniteRelationalStructure(carrier_size=0)
    true_sentence = PrimitivePositiveFormula(
        variable_count=0, free_variables=(), atoms=()
    )
    assert evaluate_pp_formula(empty, true_sentence).tuples == ((),)
    existential = PrimitivePositiveFormula(
        variable_count=1, free_variables=(), atoms=()
    )
    assert evaluate_pp_formula(empty, existential).tuples == ()
    free_axis = PrimitivePositiveFormula(
        variable_count=1, free_variables=(0,), atoms=()
    )
    assert evaluate_pp_formula(empty, free_axis).tuples == ()


def test_pp_evaluator_matches_independent_assignment_oracle() -> None:
    structure = _structure(3, ((0, 0), (0, 1), (1, 2), (2, 0)), proposition=True)
    formulas = (
        PrimitivePositiveFormula(
            variable_count=2,
            free_variables=(1,),
            atoms=({"kind": "relation", "symbol_id": "E", "variables": (0, 1)},),
        ),
        PrimitivePositiveFormula(
            variable_count=3,
            free_variables=(2, 0),
            atoms=(
                {"kind": "relation", "symbol_id": "E", "variables": (0, 1)},
                {"kind": "relation", "symbol_id": "E", "variables": (1, 2)},
                {"kind": "equality", "left": 0, "right": 2},
            ),
        ),
    )
    relation = {
        symbol.symbol_id: set(table)
        for symbol, table in zip(
            structure.signature, structure.relation_tables, strict=True
        )
    }
    for formula in formulas:
        expected = set()
        for assignment in itertools.product(
            range(structure.carrier_size), repeat=formula.variable_count
        ):
            satisfies = True
            for atom in formula.atoms:
                if atom.kind == "relation":
                    satisfies &= (
                        tuple(assignment[i] for i in atom.variables)
                        in relation[atom.symbol_id]
                    )
                else:
                    satisfies &= assignment[atom.left] == assignment[atom.right]
            if satisfies:
                expected.add(tuple(assignment[i] for i in formula.free_variables))
        assert evaluate_pp_formula(structure, formula).tuples == tuple(sorted(expected))


def test_closed_pp_sentence_matches_canonical_structure_homomorphism() -> None:
    target = FiniteRelationalStructure(
        carrier_size=3,
        signature=(
            FiniteRelationSymbol(symbol_id="E", arity=2),
            FiniteRelationSymbol(symbol_id="P", arity=1),
        ),
        relation_tables=(((0, 1), (1, 2)), ((0,), (1,))),
    )
    # The canonical database of E(x,y) and P(x) has two variables and one
    # tuple in each relation; maps into the target exactly when the sentence
    # has a satisfying assignment.
    canonical_structure = FiniteRelationalStructure(
        carrier_size=2,
        signature=target.signature,
        relation_tables=(((0, 1),), ((0,),)),
    )
    sentence = PrimitivePositiveFormula(
        variable_count=2,
        free_variables=(),
        atoms=(
            {"kind": "relation", "symbol_id": "E", "variables": (0, 1)},
            {"kind": "relation", "symbol_id": "P", "variables": (0,)},
        ),
    )

    has_solution = evaluate_pp_formula(target, sentence).tuples == ((),)
    has_homomorphism = (
        search_homomorphism(canonical_structure, target).status.value == "FOUND"
    )
    assert has_solution is has_homomorphism is True


def test_evaluation_rejects_signature_mismatch_and_admits_output_before_expansion() -> (
    None
):
    structure = FiniteRelationalStructure(carrier_size=64)
    unknown = PrimitivePositiveFormula(
        variable_count=1,
        free_variables=(0,),
        atoms=({"kind": "relation", "symbol_id": "R", "variables": (0,)},),
    )
    with pytest.raises(OperationDomainValidationError, match="name a symbol"):
        evaluate_pp_formula(structure, unknown)

    broad_relation = PrimitivePositiveFormula(
        variable_count=3, free_variables=(0, 1, 2), atoms=()
    )
    with pytest.raises(OperationResourceAdmissionError, match="defined relation"):
        evaluate_pp_formula(structure, broad_relation)


def test_equality_output_bound_uses_identified_coordinate_count() -> None:
    structure = FiniteRelationalStructure(carrier_size=64)
    formula = PrimitivePositiveFormula(
        variable_count=3,
        free_variables=(0, 1, 2),
        atoms=({"kind": "equality", "left": 0, "right": 1},),
    )
    result = evaluate_pp_formula(structure, formula)
    assert len(result.tuples) == 64**2


def test_defined_value_rejects_formula_incompatible_with_retained_structure() -> None:
    structure = FiniteRelationalStructure(carrier_size=2)
    formula = PrimitivePositiveFormula(
        variable_count=1,
        free_variables=(0,),
        atoms=({"kind": "relation", "symbol_id": "R", "variables": (0,)},),
    )
    with pytest.raises(ValueError, match="names no symbol"):
        PPDefinedRelation(structure=structure, formula=formula, tuples=((0,),))


def test_evaluation_admits_coordinate_work_separately_from_atom_checks() -> None:
    structure = FiniteRelationalStructure(
        carrier_size=16,
        signature=(FiniteRelationSymbol(symbol_id="Q", arity=4),),
        relation_tables=((),),
    )
    formula = PrimitivePositiveFormula(
        variable_count=5,
        free_variables=(),
        atoms=tuple(
            {
                "kind": "relation",
                "symbol_id": "Q",
                "variables": variables,
            }
            for variables in (
                (0, 1, 2, 3),
                (0, 1, 3, 2),
                (0, 2, 1, 3),
                (0, 2, 3, 1),
                (0, 3, 1, 2),
                (0, 3, 2, 1),
                (1, 0, 2, 3),
                (1, 0, 3, 2),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="coordinate steps"):
        evaluate_pp_formula(structure, formula)


def test_operation_is_discoverable_and_serialized_consumer_value_is_exact() -> None:
    operation_id = "pp_formula.evaluate_relation.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = PPFormulaEvaluationRequest.model_validate(tool.examples[0].input)
    direct = tool.run(request)
    assert direct.tuples == ((0, 2),)

    wire_input = json.loads(request.model_dump_json())
    output = invoke_operation(operation_id, wire_input, Catalog.open()).output
    assert output["tuples"] == [[0, 2]]
    assert PPDefinedRelation.model_validate(output).tuples == direct.tuples
