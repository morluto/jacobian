"""The removed projection cannot publish an arbitrary family as a presentation."""

import json
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.numerical_semigroups import relation_binomials
from jacobian.math.number_theory.numerical_semigroups._presentation_models import (
    MinimalPresentationResult,
)


@pytest.mark.parametrize(
    "relations",
    ([], [{"first": [10, 0], "second": [0, 6]}], [{"first": [5, 0], "second": [0, 3]}]),
)
def test_public_relation_projection_is_no_longer_invocable(
    relations: list[Any],
) -> None:
    with pytest.raises(ValueError, match="unknown operation"):
        invoke_operation(
            "number_theory.numerical_semigroup.presentation_binomials.compute",
            {"generators": ["3", "5"], "relations": relations},
            Catalog.open(),
        )


def test_actual_public_presentation_still_feeds_the_native_encoding() -> None:
    output = invoke_operation(
        "number_theory.numerical_semigroup.minimal_presentation.compute",
        {"generators": ["8", "5", "3"]},
        Catalog.open(),
    ).output
    assert output == {
        "minimal_generators": ["3", "5"],
        "betti_elements": ["15"],
        "relations": [{"first": [5, 0], "second": [0, 3]}],
    }
    presentation = MinimalPresentationResult.model_validate_json(
        json.dumps(output), strict=True
    )
    result = relation_binomials(presentation.minimal_generators, presentation.relations)
    assert result.minimal_generators == presentation.minimal_generators
    assert len(result.binomials) == 1
    binomial = result.binomials[0]
    assert (binomial.left_exponents, binomial.right_exponents) == ((5, 0), (0, 3))
    assert (binomial.left_coefficient, binomial.right_coefficient) == (1, -1)
