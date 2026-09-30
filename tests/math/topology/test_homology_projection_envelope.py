"""Ordering and growth regressions for the homology-coordinate projection.

The projection envelope must be decided before either Smith backend runs, and
the torsion witness comparisons must be charged to the same envelope as the
matrix-vector products around them.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
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
from jacobian.math.topology.simplicial_sets.standard import (
    simplex_boundary,
    standard_simplex,
)

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


def _identity_on(source: FiniteTruncatedSimplicialSet) -> TruncatedSimplicialMap:
    return TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )


def _discrete_identity(size: int) -> TruncatedSimplicialMap:
    indices = tuple(range(size))
    labels = tuple(map(str, indices))
    result = from_tables(1, (labels, labels), ((indices, indices),), ((indices,),))
    assert result.simplicial_set is not None
    return _identity_on(result.simplicial_set)


def _torsion_inclusion_into_contractible_nerve() -> TruncatedSimplicialMap:
    """Include C2 into the monoid obtained by adjoining an absorbing zero.

    Its normalized d_2 has columns (2,0), (1,0), (1,0), (0,1), so the
    target H_1 vanishes while C_1 still has two coordinates.
    """
    pairs = tuple(product(range(3), repeat=2))
    indices = {pair: index for index, pair in enumerate(pairs)}
    result = from_tables(
        2,
        (("v",), ("id", "g", "zero"), tuple(map(str, pairs))),
        (
            ((0, 0, 0), (0, 0, 0)),
            (
                tuple(b for a, b in pairs),
                tuple(2 if a == 2 or b == 2 else (a + b) % 2 for a, b in pairs),
                tuple(a for a, b in pairs),
            ),
        ),
        (
            ((0,),),
            (
                tuple(indices[(0, a)] for a in range(3)),
                tuple(indices[(a, 0)] for a in range(3)),
            ),
        ),
    )
    assert result.simplicial_set is not None
    return TruncatedSimplicialMap(
        source=_cyclic_group_two_nerve_prefix(),
        target=result.simplicial_set,
        maps=((0,), (0, 1), (0, 1, 3, 4)),
    )


def _projection_work(map_value: TruncatedSimplicialMap) -> tuple[int, int]:
    chain_map = induced_normalized_chain_map(map_value)
    source_plan = admit_integral_homology(chain_map.source)
    target_plan = admit_integral_homology(chain_map.target)
    bound = simplicial_maps.__dict__["_admit_homology_projection_plan"](
        chain_map, source_plan, target_plan
    )
    cache: list[NormalizedEndpoint] = []
    endpoint = simplicial_maps.__dict__["_normalized_homology_endpoint"]
    source = endpoint(map_value.source, source_plan, cache, _canonical=True)
    target = endpoint(map_value.target, target_plan, cache, _canonical=True)
    realized = simplicial_maps.__dict__["_admit_homology_projection"](
        chain_map, source.homology, target.homology, target.right_inverses
    )[-1]
    return bound, realized


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


@pytest.mark.parametrize(
    "map_value",
    [
        _discrete_identity(32),
        _identity_on(standard_simplex(3, 2)),
        _identity_on(simplex_boundary(2, 2)),
        _identity(),
        _collapse(),
        _torsion_inclusion_into_contractible_nerve(),
    ],
    ids=("discrete", "simplex", "circle", "torsion", "collapse", "torsion-to-zero"),
)
def test_projection_plan_bounds_the_realized_ledger(
    map_value: TruncatedSimplicialMap,
) -> None:
    bound, realized = _projection_work(map_value)
    assert 0 < realized <= bound


def test_many_generators_are_refused_before_endpoint_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # There are 32 independent H_0 generators, each projected through three
    # 32-square matrices. A single mat-vec reservation is not an upper bound.
    identity = _discrete_identity(32)
    bound, realized = _projection_work(identity)
    assert bound == realized == 199_680
    calls = _count_smith_backend(monkeypatch)
    monkeypatch.setattr(
        simplicial_maps, "MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK", 100_000
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        induced_normalized_homology_map(identity)
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_work_exceeded"
    )
    assert calls == []


def test_combined_generator_work_is_refused_before_endpoint_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = induced_normalized_homology_map(_discrete_identity(10))
    calls = _count_smith_backend(monkeypatch)
    # Each map reserves 6,300 units, so only the combined reservation fails.
    monkeypatch.setattr(
        simplicial_maps, "MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK", 10_000
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_simplicial_homology_maps(identity, identity)
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_work_exceeded"
    )
    assert calls == []


def test_generator_output_axes_are_reserved_before_endpoint_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_smith_backend(monkeypatch)
    # C_1 is zero, but the identity on H_0=Z^2 still returns four coordinates.
    chain_map = induced_normalized_chain_map(_discrete_identity(2))
    plan = admit_integral_homology(chain_map.source)
    monkeypatch.setattr(simplicial_maps, "MAX_INDUCED_CHAIN_MAP_OUTPUT_CELLS", 3)
    with pytest.raises(OperationResourceAdmissionError) as error:
        simplicial_maps.__dict__["_admit_homology_projection_plan"](
            chain_map, plan, plan
        )
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_output_exceeded"
    )
    assert calls == []


@pytest.mark.parametrize("dimension, degree, work", [(2, 2, 126), (0, 0, 2)])
def test_presolved_cases_keep_their_inexpensive_projection_boundary(
    monkeypatch: pytest.MonkeyPatch, dimension: int, degree: int, work: int
) -> None:
    identity = _identity_on(standard_simplex(dimension, degree))
    assert _projection_work(identity) == (work, work)
    monkeypatch.setattr(
        simplicial_maps, "MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK", work
    )
    result = induced_normalized_homology_map(identity)
    assert len(result.degree_maps) == degree
    if degree:
        assert result.degree_maps[0].free_generator_images[0].free == (1,)
        assert result.degree_maps[1].free_generator_images == ()


def test_torsion_witnesses_use_source_orders_and_target_chain_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_value = _torsion_inclusion_into_contractible_nerve()
    scalar_products: list[tuple[int, int, int]] = []
    original: Callable[[int, int, int], tuple[int, int]] = simplicial_maps.__dict__[
        "_admit_projection_scalar_product"
    ]

    def record_product(left_bits: int, right_bits: int, count: int) -> tuple[int, int]:
        scalar_products.append((left_bits, right_bits, count))
        return original(left_bits, right_bits, count)

    monkeypatch.setattr(
        simplicial_maps, "_admit_projection_scalar_product", record_product
    )
    bound, realized = _projection_work(map_value)
    assert bound >= realized
    # The first pair is preflight, the second is the realized check. Both
    # retain the source order 2, one source coordinate, and two target chain
    # coordinates, although the target has no H_1 coordinates at all.
    assert scalar_products == [(2, 1, 1), (2, 2, 2)] * 2
    result = induced_normalized_homology_map(map_value)
    target = result.target.homology_groups[1]
    assert isinstance(target, IntegralHomologyGroupValue)
    assert target.chain_rank == 2
    assert target.free_rank == 0 and target.torsion_invariant_factors == ()
    assert result.degree_maps[1].torsion_generator_images[0].torsion == ()


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
