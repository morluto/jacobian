import json
from fractions import Fraction
from itertools import product

from jacobian._exact import CanonicalRational
from jacobian.math.affine_semigroups import (
    AffineConfiguration,
    construct,
    fiber_graph,
    markov_basis,
)
from jacobian.math.affine_semigroups._tools import TOOLS
from jacobian.math.affine_semigroups.graver import graver_basis
from jacobian.math.affine_semigroups.graver_models import (
    IntegerConfigurationMarkovBasisRequest,
)
from jacobian.math.matrices.values import IntegerMatrix


def _configuration(weights: tuple[int, ...]) -> IntegerMatrix:
    return IntegerMatrix.model_validate({"entries": [list(weights)]})


def test_markov_basis_has_distinct_global_contract_and_exact_moves() -> None:
    configuration = _configuration((1, 2, 3))
    result = markov_basis(configuration)
    assert result.moves == graver_basis(configuration).vectors
    assert (
        result.property == "GENERATES_TORIC_IDEAL_AND_CONNECTS_EVERY_NONNEGATIVE_FIBER"
    )
    assert result.construction == "COMPLETE_GRAVER_BASIS"


def test_returned_moves_connect_every_small_materialized_fiber() -> None:
    configuration = _configuration((1, 2, 3))
    labeled = AffineConfiguration(
        row_labels=("degree",),
        generator_labels=("a", "b", "c"),
        entries=((1, 2, 3),),
    )
    semigroup = construct(labeled, (CanonicalRational.from_fraction(Fraction(1)),))
    moves = markov_basis(configuration).moves
    for target in range(16):
        graph = fiber_graph(semigroup, (target,), moves)
        assert len(graph.components) <= 1
        assert all(
            sum(
                weight * count
                for weight, count in zip((1, 2, 3), factorization, strict=True)
            )
            == target
            for factorization in graph.vertices
        )


def test_markov_moves_match_independent_complete_fiber_graphs() -> None:
    moves = markov_basis(_configuration((1, 2, 3))).moves
    vectors = tuple(product(range(16), repeat=3))
    for target in range(16):
        fiber = {
            vector
            for vector in vectors
            if vector[0] + 2 * vector[1] + 3 * vector[2] == target
        }
        if not fiber:
            continue
        reached = {min(fiber)}
        frontier = list(reached)
        while frontier:
            current = frontier.pop()
            for move in moves:
                for sign in (-1, 1):
                    neighbour = tuple(
                        value + sign * delta
                        for value, delta in zip(current, move, strict=True)
                    )
                    if neighbour in fiber and neighbour not in reached:
                        reached.add(neighbour)
                        frontier.append(neighbour)
        assert reached == fiber


def test_markov_operation_example_round_trips_through_serialization() -> None:
    tool = next(
        candidate
        for candidate in TOOLS
        if candidate.operation_id == "integer_configuration.markov_basis.compute"
    )
    request = IntegerConfigurationMarkovBasisRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    result = tool.run(request)
    restored = tool.result_type.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.moves == markov_basis(_configuration((1, 2, 3))).moves
