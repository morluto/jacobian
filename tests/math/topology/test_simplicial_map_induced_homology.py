"""Induced homology maps on exact finite simplicial-set prefixes."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes.values import IntegralHomologyGroupValue
from jacobian.math.topology.simplicial_sets import (
    FiniteTruncatedSimplicialSet,
    TruncatedSimplicialMap,
    compose_simplicial_homology_maps,
    induced_normalized_homology_map,
    normalized_chains,
)
from jacobian.math.topology.simplicial_sets import maps as simplicial_maps
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _cyclic_group_two_nerve_prefix() -> FiniteTruncatedSimplicialSet:
    # The one-object category with endomorphism group C2. The normalized
    # degree-1 generator g has boundary zero and the nondegenerate 2-simplex
    # (g,g) has boundary 2g, so an independent cellular calculation gives
    # H_1 = Z/2.
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


def test_identity_preserves_free_and_torsion_classes_and_serializes() -> None:
    source = _cyclic_group_two_nerve_prefix()
    chain_complex = normalized_chains(source).chain_complex
    # Independent normalized-chain calculation: ker(d_1)=Z and im(d_2)=2Z.
    assert chain_complex.differential_matrices == (((0,),), ((2,),))
    identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )

    result = induced_normalized_homology_map(identity)

    h0, h1 = result.source.homology_groups
    assert isinstance(h0, IntegralHomologyGroupValue)
    assert isinstance(h1, IntegralHomologyGroupValue)
    assert h0.free_rank == 1
    assert h1.free_rank == 0
    assert h1.torsion_invariant_factors == (2,)
    assert result.degree_maps[0].free_generator_images[0].free == (1,)
    assert result.degree_maps[1].torsion_generator_images[0].torsion == (1,)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_zero_rank_target_and_functorial_composition() -> None:
    source = _cyclic_group_two_nerve_prefix()
    point = standard_simplex(0, 2)
    collapse = TruncatedSimplicialMap(
        source=source,
        target=point,
        maps=tuple((0,) * len(level) for level in source.sets),
    )
    identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )
    identity_map = induced_normalized_homology_map(identity)
    collapse_map = induced_normalized_homology_map(collapse)

    h1 = collapse_map.target.homology_groups[1]
    assert isinstance(h1, IntegralHomologyGroupValue)
    assert h1.free_rank == 0
    assert h1.torsion_invariant_factors == ()
    assert collapse_map.degree_maps[1].torsion_generator_images[0].free == ()
    assert collapse_map.degree_maps[1].torsion_generator_images[0].torsion == ()
    assert compose_simplicial_homology_maps(identity_map, identity_map) == identity_map
    composed = compose_simplicial_homology_maps(identity_map, collapse_map)
    assert composed == collapse_map


def test_induced_homology_executes_each_admitted_endpoint_plan_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cyclic_group_two_nerve_prefix()
    point = standard_simplex(0, 2)
    collapse = TruncatedSimplicialMap(
        source=source,
        target=point,
        maps=tuple((0,) * len(level) for level in source.sets),
    )
    original = simplicial_maps.admit_integral_homology
    calls = 0
    original_from_tables = simplicial_maps.from_tables
    table_validations = 0

    def count_admissions(complex_value):
        nonlocal calls
        calls += 1
        return original(complex_value)

    def count_table_validations(*args, **kwargs):
        nonlocal table_validations
        table_validations += 1
        return original_from_tables(*args, **kwargs)

    monkeypatch.setattr(simplicial_maps, "admit_integral_homology", count_admissions)
    monkeypatch.setattr(simplicial_maps, "from_tables", count_table_validations)
    induced_normalized_homology_map(collapse)

    assert calls == 2
    assert table_validations == 2


def test_composition_admits_both_coordinate_projections_before_either_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cyclic_group_two_nerve_prefix()
    identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )
    homology_map = induced_normalized_homology_map(identity)
    original_admit = simplicial_maps._admit_homology_projection
    admitted_work: list[int] = []

    def record_admission(*args, **kwargs):
        result = original_admit(*args, **kwargs)
        admitted_work.append(result[-1])
        if len(admitted_work) == 1:
            monkeypatch.setattr(
                simplicial_maps,
                "MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK",
                result[-1],
            )
        return result

    def projection_must_not_run(*args, **kwargs):
        pytest.fail("coordinate projection ran before combined work admission")

    monkeypatch.setattr(
        simplicial_maps, "_admit_homology_projection", record_admission
    )
    monkeypatch.setattr(
        simplicial_maps, "_map_homology_generator", projection_must_not_run
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_simplicial_homology_maps(homology_map, homology_map)

    assert len(admitted_work) == 2
    assert admitted_work[0] > 0
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_projection_work_exceeded"
    )


def test_composition_canonicalizes_authored_homology_endpoints() -> None:
    source = _cyclic_group_two_nerve_prefix()
    identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )
    canonical = induced_normalized_homology_map(identity)
    stale_set = source.model_copy(
        update={"checked_identities": source.checked_identities + 1}
    )
    stale_simplicial_map = canonical.simplicial_map.model_copy(
        update={"source": stale_set, "target": stale_set}
    )
    stale_endpoint = canonical.source.model_copy(update={"simplicial_set": stale_set})
    stale = canonical.model_copy(
        update={
            "simplicial_map": stale_simplicial_map,
            "source": stale_endpoint,
            "target": stale_endpoint,
        }
    )

    assert compose_simplicial_homology_maps(stale, canonical) == canonical


def test_composition_admits_each_distinct_carrier_with_shared_chain_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_carrier = standard_simplex(0, 2)
    second_carrier = first_carrier.model_copy(
        update={"sets": (("other-vertex",), *first_carrier.sets[1:])}
    )
    third_carrier = first_carrier.model_copy(
        update={"sets": (("third-vertex",), *first_carrier.sets[1:])}
    )

    def map_between(source, target):
        return TruncatedSimplicialMap(
            source=source,
            target=target,
            maps=tuple(tuple(range(len(level))) for level in source.sets),
        )

    first = induced_normalized_homology_map(
        map_between(first_carrier, second_carrier)
    )
    second = induced_normalized_homology_map(
        map_between(second_carrier, third_carrier)
    )
    chain_complex = normalized_chains(first_carrier).chain_complex
    one_endpoint_work = simplicial_maps.admit_integral_homology(
        chain_complex
    ).total_work
    monkeypatch.setattr(
        simplicial_maps, "MAX_INTEGRAL_HOMOLOGY_WORK_UNITS", one_endpoint_work
    )

    def endpoint_must_not_run(*args, **kwargs):
        pytest.fail("endpoint homology ran before the aggregate work admission")

    monkeypatch.setattr(simplicial_maps, "normalized_homology", endpoint_must_not_run)
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_simplicial_homology_maps(first, second)

    assert error.value.errors()[0]["type"] == (
        "simplicial_set.induced_homology_endpoint_work_budget_exceeded"
    )


def test_composition_checkpoints_while_composing_generator_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cyclic_group_two_nerve_prefix()
    identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )
    homology_map = induced_normalized_homology_map(identity)
    phases: list[str] = []
    monkeypatch.setattr(
        simplicial_maps, "request_checkpoint", lambda phase: phases.append(phase)
    )

    compose_simplicial_homology_maps(homology_map, homology_map)

    assert "during homology-coordinate composition" in phases


def test_identity_of_contractible_simplex_keeps_zero_rank_map_rows() -> None:
    source = standard_simplex(1, 2)
    identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )

    result = induced_normalized_homology_map(identity)

    assert len(result.degree_maps) == 2
    group = result.source.homology_groups[1]
    assert isinstance(group, IntegralHomologyGroupValue)
    assert group.free_rank == 0
    assert result.degree_maps[1].free_generator_images == ()
    assert result.degree_maps[1].torsion_generator_images == ()


def test_rejects_noncomposable_homology_maps() -> None:
    source = _cyclic_group_two_nerve_prefix()
    c2_identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )
    source = standard_simplex(1, 2)
    simplex_identity = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )

    with pytest.raises(ValueError, match="first target homology"):
        compose_simplicial_homology_maps(
            induced_normalized_homology_map(c2_identity),
            induced_normalized_homology_map(simplex_identity),
        )
