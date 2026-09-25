"""Exact ground-label relabeling for finite delta-matroids."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.matroids.delta import (
    FiniteDeltaMatroid,
    relabel,
)
from jacobian.math.combinatorics.matroids.delta.relabel import (
    DeltaMatroidRelabelRequest,
)


def test_relabel_preserves_feasible_membership_under_a_ground_permutation() -> None:
    source = FiniteDeltaMatroid(
        ground=("a", "b", "c"),
        feasible=((0,), (1,), (2,)),
    )
    target_ground = ("c", "a", "b")

    renamed = relabel(source, target_ground)

    old_to_new = dict(zip(source.ground, target_ground, strict=True))
    source_family = {
        frozenset(source.ground[index] for index in row) for row in source.feasible
    }
    transported_family = {
        frozenset(old_to_new[label] for label in feasible) for feasible in source_family
    }
    result_family = {
        frozenset(renamed.ground[index] for index in row) for row in renamed.feasible
    }

    assert renamed.ground == target_ground
    assert result_family == transported_family
    assert relabel(renamed, source.ground) == source


def test_empty_ground_relabels_and_catalog_example_round_trips() -> None:
    empty = FiniteDeltaMatroid(ground=(), feasible=((),))
    assert relabel(empty, ()) == empty

    catalog = Catalog.open()
    operation = catalog.operation("delta_matroid.relabel.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output == {
        "ground": ["x", "y"],
        "feasible": [[], [0], [0, 1], [1]],
    }
    decoded = FiniteDeltaMatroid.model_validate_json(json.dumps(result.output))
    assert decoded.ground == ("x", "y")
    assert decoded.feasible == ((), (0,), (0, 1), (1,))


def test_target_labels_must_be_a_bounded_bijection() -> None:
    assert DeltaMatroidRelabelRequest.model_json_schema()["admission_limits"] == {
        "max_source_feasible_set_memberships": 16_384,
        "max_source_ground_label_utf8_bytes": 2_048,
        "max_symmetric_exchange_candidate_checks_per_pass": 250_000,
        "max_target_ground_label_utf8_bytes": 2_048,
    }
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((),))
    with pytest.raises(ValidationError, match="cover the source ground"):
        DeltaMatroidRelabelRequest(delta_matroid=source, target_ground=("x",))
    with pytest.raises(ValidationError, match="must be unique"):
        DeltaMatroidRelabelRequest(delta_matroid=source, target_ground=("x", "x"))

    at_limit = ("a" * 1024, "b" * 1024)
    assert (
        len(
            DeltaMatroidRelabelRequest(
                delta_matroid=source, target_ground=at_limit
            ).target_ground
        )
        == 2
    )
    with pytest.raises(ValidationError, match="2048-byte envelope"):
        DeltaMatroidRelabelRequest(
            delta_matroid=source, target_ground=("a" * 1024, "b" * 1025)
        )


def test_relabel_rejects_a_forged_source_that_fails_symmetric_exchange() -> None:
    forged = FiniteDeltaMatroid.model_construct(
        ground=("a", "b", "c"),
        feasible=((), (0, 1), (0, 1, 2)),
    )
    request = DeltaMatroidRelabelRequest(
        delta_matroid=forged,
        target_ground=("x", "y", "z"),
    )
    with pytest.raises(ValueError, match="fails symmetric exchange"):
        relabel(request.delta_matroid, request.target_ground)
