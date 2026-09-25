"""Induced homology maps on exact finite simplicial-set prefixes."""

import pytest

from jacobian.math.topology.chain_complexes.values import IntegralHomologyGroupValue
from jacobian.math.topology.simplicial_sets import (
    FiniteTruncatedSimplicialSet,
    TruncatedSimplicialMap,
    compose_simplicial_homology_maps,
    induced_normalized_homology_map,
    normalized_chains,
)
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
