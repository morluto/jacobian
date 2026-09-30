"""Ordering and growth regressions for the homology-coordinate projection.

The projection envelope must be decided before either Smith backend runs, and
the torsion witness comparisons must be charged to the same envelope as the
matrix-vector products around them.
"""

from __future__ import annotations

from typing import Any

import pytest

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes._integral_homology import (
    IntegralHomologyExecutionPlan,
    admit_integral_homology,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainMapValue,
    HomologyGroupValue,
    IntegralHomologyGroupValue,
)
from jacobian.math.topology.simplicial_sets import (
    FiniteTruncatedSimplicialSet,
    TruncatedSimplicialMap,
    compose_simplicial_homology_maps,
    induced_normalized_chain_map,
    induced_normalized_homology_map,
)
from jacobian.math.topology.simplicial_sets import maps as simplicial_maps
from jacobian.math.topology.simplicial_sets.maps import NormalizedHomologyResult
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

AdmitProjection = Any
NormalizedEndpoint = Any


def _cyclic_group_two_nerve_prefix() -> FiniteTruncatedSimplicialSet:
    result = from_tables(
        2,
        (("v",), ("id", "g"), ("id_id", "id_g", "g_id", "g_g")),
        (
            ((0, 0), (0, 0)),
            ((0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 1, 1)),
        ),
        (((0,),), ((0, 1), (0, 2))),
    )
    assert result.simplicial_set is not None
    return result.simplicial_set


def _identity() -> TruncatedSimplicialMap:
    source = _cyclic_group_two_nerve_prefix()
    return TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )


def _collapse() -> TruncatedSimplicialMap:
    source = _cyclic_group_two_nerve_prefix()
    point = standard_simplex(0, 2)
    return TruncatedSimplicialMap(
        source=source,
        target=point,
        maps=tuple((0,) * len(level) for level in source.sets),
    )


def _count_smith_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every dispatched Smith kernel call."""
    calls: list[str] = []
    original = simplicial_maps.__dict__["compute_integral_homology"]

    def counting_backend(*args: Any, **kwargs: Any) -> Any:
        calls.append("compute_integral_homology")
        return original(*args, **kwargs)

    monkeypatch.setattr(simplicial_maps, "compute_integral_homology", counting_backend)
    return calls


def test_projection_envelope_is_decided_before_the_smith_backend_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused projection must not pay for the endpoint reductions.

    ``compute_integral_homology`` is the dispatched Smith kernel, and it runs
    on a child worker for the reductions that need one. When the coordinate
    projection is already outside its envelope, the operation should say so
    from the admitted endpoint plans instead of reducing first.
    """
    backend_calls = _count_smith_backend(monkeypatch)
    monkeypatch.setattr(simplicial_maps, "MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK", 0)

    with pytest.raises(OperationResourceAdmissionError) as error:
        induced_normalized_homology_map(_collapse())

    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_work_exceeded"
    )
    assert backend_calls == []


def test_composition_projection_envelope_is_decided_before_either_backend_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    homology_map = induced_normalized_homology_map(_identity())
    backend_calls = _count_smith_backend(monkeypatch)
    monkeypatch.setattr(simplicial_maps, "MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK", 0)

    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_simplicial_homology_maps(homology_map, homology_map)

    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_work_exceeded"
    )
    assert backend_calls == []


def test_torsion_witness_products_are_charged_to_the_projection_envelope() -> None:
    """The witness comparisons double the height of their inputs.

    The retained torsion order and cycle coordinates here each sit at 32,001
    digits, inside every value-layer bound. The surrounding matrix-vector
    products stay within the height limit, but the witness comparison
    materializes ``torsion_order * value`` entries of 64,001 digits, so the
    projection preflight must refuse it.
    """
    source = _cyclic_group_two_nerve_prefix()
    identity = _identity()
    result = induced_normalized_homology_map(identity)
    chain_map = induced_normalized_chain_map(identity)
    plan: IntegralHomologyExecutionPlan = admit_integral_homology(chain_map.source)
    endpoint: NormalizedEndpoint = simplicial_maps.__dict__[
        "_normalized_homology_endpoint"
    ](source, plan, [], _canonical=True)

    huge = 10**32_000
    admit_projection: AdmitProjection = simplicial_maps.__dict__[
        "_admit_homology_projection"
    ]

    with pytest.raises(OperationResourceAdmissionError) as error:
        admit_projection(
            ChainMapValue(
                source=chain_map.source,
                target=chain_map.target,
                map_matrices=chain_map.map_matrices,
            ),
            _groups_at_limit(result.source, huge),
            _groups_at_limit(result.target, huge),
            endpoint.right_inverses,
        )

    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_height_exceeded"
    )
    assert MAX_CANONICAL_INTEGER_DIGITS < 2 * 32_001


def _groups_at_limit(
    endpoint: NormalizedHomologyResult, value: int
) -> NormalizedHomologyResult:
    """Move the degree-one torsion generator to the retained height limit."""
    groups: list[IntegralHomologyGroupValue | HomologyGroupValue] = []
    for group in endpoint.homology_groups:
        if (
            not isinstance(group, IntegralHomologyGroupValue)
            or not group.torsion_generators
        ):
            groups.append(group)
            continue
        generator = group.torsion_generators[0]
        groups.append(
            group.model_copy(
                update={
                    "torsion_invariant_factors": (value,),
                    "torsion_generators": (
                        generator.model_copy(
                            update={
                                "order": value,
                                "cycle": generator.cycle.model_copy(
                                    update={"coefficients": (value,)}
                                ),
                                "bounding_chain": generator.bounding_chain.model_copy(
                                    update={"coefficients": (value,)}
                                ),
                            }
                        ),
                    ),
                }
            )
        )
    return endpoint.model_copy(update={"homology_groups": tuple(groups)})
