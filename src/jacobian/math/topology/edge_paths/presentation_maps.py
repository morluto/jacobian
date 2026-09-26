"""Exact bounded maps between finite presentation carriers."""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, cast

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology._request_admission import run_topology_admission
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    MAX_INDUCED_MAP_EDGE_LETTERS,
    MAX_PRESENTATION_RELATOR_LETTERS,
    MAX_WORD,
    EdgeWordEntry,
    FiniteGroupPresentation,
    FiniteGroupWord,
    FundamentalGroupMapRequest,
    FundamentalGroupMapResult,
    PresentationRelatorImage,
    WordLetter,
)
from jacobian.math.topology.edge_paths.operations import fundamental_group_presentation


class DirectRelatorMatchRequest(StrictModel):
    """Check only the finite direct-relator witness, not normal-closure membership."""

    source: FiniteGroupPresentation
    target: FiniteGroupPresentation
    generator_images: tuple[FiniteGroupWord, ...] = Field(min_length=0, max_length=64)


class DirectRelatorMatchResult(StrictModel):
    source: FiniteGroupPresentation
    target: FiniteGroupPresentation
    generator_images: tuple[FiniteGroupWord, ...]
    relator_images: tuple[FiniteGroupWord, ...]
    direct_relators_matched: bool
    obstruction_index: int | None = None


def _reduce(letters: Any) -> tuple[WordLetter, ...]:
    out: list[WordLetter] = []
    for letter in letters:
        if (
            out
            and out[-1].generator == letter.generator
            and out[-1].exponent == -letter.exponent
        ):
            out.pop()
        else:
            out.append(letter)
    return tuple(out)


def _inverse(word: FiniteGroupWord) -> tuple[WordLetter, ...]:
    return tuple(
        WordLetter(generator=x.generator, exponent=-x.exponent)
        for x in reversed(word.letters)
    )


def direct_relator_match(
    source: FiniteGroupPresentation,
    target: FiniteGroupPresentation,
    generator_images: tuple[FiniteGroupWord, ...],
) -> DirectRelatorMatchResult:
    if len(generator_images) != len(source.generators):
        raise OperationDomainValidationError(
            location=("generator_images",),
            code="presentation_map.generator_axis",
            message="one target word is required for every source generator",
        )
    if any(
        letter.generator >= len(target.generators)
        for word in generator_images
        for letter in word.letters
    ):
        raise OperationDomainValidationError(
            location=("generator_images",),
            code="presentation_map.target_generator",
            message="a generator image names no target generator",
        )
    target_relators = {word.letters for word in target.relators} | {
        _inverse(word) for word in target.relators
    }
    images = []
    for relator in source.relators:
        expanded: list[WordLetter] = []
        for letter in relator.letters:
            image_word = generator_images[letter.generator].letters
            if letter.exponent == 1:
                expanded.extend(image_word)
            else:
                expanded.extend(_inverse(FiniteGroupWord(letters=image_word)))
        images.append(FiniteGroupWord(letters=_reduce(expanded)))
    for i, relator_image in enumerate(images):
        if relator_image.letters and relator_image.letters not in target_relators:
            return DirectRelatorMatchResult(
                source=source,
                target=target,
                generator_images=generator_images,
                relator_images=tuple(images),
                direct_relators_matched=False,
                obstruction_index=i,
            )
    return DirectRelatorMatchResult(
        source=source,
        target=target,
        generator_images=generator_images,
        relator_images=tuple(images),
        direct_relators_matched=True,
    )


def _tree_paths(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    base_vertex: str,
) -> dict[str, tuple[tuple[str, str], ...]]:
    adjacency: dict[str, list[str]] = {vertex: [] for vertex in vertices}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbours in adjacency.values():
        neighbours.sort()

    paths: dict[str, tuple[tuple[str, str], ...]] = {base_vertex: ()}
    pending: deque[str] = deque((base_vertex,))
    while pending:
        vertex = pending.popleft()
        for neighbour in adjacency[vertex]:
            if neighbour not in paths:
                paths[neighbour] = paths[vertex] + ((vertex, neighbour),)
                pending.append(neighbour)
    if set(paths) != set(vertices):
        raise OperationDomainValidationError(
            location=("presentation", "spanning_tree_edges"),
            code="fundamental_group_map.tree_not_spanning",
            message="the supplied presentation tree does not span its component",
        )
    return paths


def _map_edge_path(
    path: tuple[tuple[str, str], ...],
    vertex_map: dict[str, str],
    target_edge_words: dict[tuple[str, str], EdgeWordEntry],
) -> tuple[WordLetter, ...]:
    letters: list[WordLetter] = []
    for left, right in path:
        image_left, image_right = vertex_map[left], vertex_map[right]
        if image_left == image_right:
            continue
        edge = (
            (image_left, image_right)
            if image_left < image_right
            else (image_right, image_left)
        )
        entry = target_edge_words.get(edge)
        if entry is None:
            raise OperationDomainValidationError(
                location=("map", "vertex_map"),
                code="fundamental_group_map.edge_image_missing",
                message="a source path edge does not map into the target base component",
            )
        word = entry.forward if image_left < image_right else entry.backward
        letters.extend(word.letters)
    return _reduce(letters)


def _edge_path_to_base_loop(
    edge: tuple[str, str],
    source_paths: dict[str, tuple[tuple[str, str], ...]],
    vertex_map: dict[str, str],
    target_edge_words: dict[tuple[str, str], EdgeWordEntry],
) -> tuple[WordLetter, ...]:
    left, right = edge
    path_back = tuple((end, start) for start, end in reversed(source_paths[right]))
    path = source_paths[left] + (edge,) + path_back
    return _map_edge_path(path, vertex_map, target_edge_words)


def _permutation_sign(values: tuple[str, ...]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _validate_simplicial_map(value: SimplicialMap) -> None:
    """Run model-level simplicial validation under topology admission."""
    value.__class__.model_validate(value.model_dump())


def induced_fundamental_group_map(
    request: FundamentalGroupMapRequest,
) -> FundamentalGroupMapResult:
    """Construct the based generator map and replay triangle relations.

    Presentations are derived from the exact map's canonical complexes, so
    callers cannot substitute similarly shaped presentation data. The base
    vertex must map to the target base vertex; an unbased change-of-basepoint
    path is deliberately not inferred.
    """
    simplicial_map: SimplicialMap = request.map
    run_topology_admission(
        lambda: _validate_simplicial_map(simplicial_map), location=("map",)
    )
    source_vertices = simplicial_map.source.vertices
    target_vertices = simplicial_map.target.vertices
    vertex_map = dict(zip(source_vertices, simplicial_map.vertex_map, strict=True))
    if request.source_base_vertex not in set(source_vertices):
        raise OperationDomainValidationError(
            location=("source_base_vertex",),
            code="fundamental_group_map.base_vertex_missing",
            message="the source base vertex must belong to the source complex",
        )
    if request.target_base_vertex not in set(target_vertices):
        raise OperationDomainValidationError(
            location=("target_base_vertex",),
            code="fundamental_group_map.base_vertex_missing",
            message="the target base vertex must belong to the target complex",
        )
    if vertex_map[request.source_base_vertex] != request.target_base_vertex:
        raise OperationDomainValidationError(
            location=("target_base_vertex",),
            code="fundamental_group_map.basepoint_mismatch",
            message="the simplicial map must send the source base vertex to the target base vertex",
        )

    source = fundamental_group_presentation(
        simplicial_map.source, request.source_base_vertex
    )
    target = fundamental_group_presentation(
        simplicial_map.target, request.target_base_vertex
    )
    source_paths = _tree_paths(
        source.component_vertices, source.spanning_tree_edges, source.base_vertex
    )
    target_edge_words = {entry.edge: entry for entry in target.edge_words}

    # Bound aggregate path expansion before generating any word rows.
    generator_path_letters = sum(
        len(source_paths[left]) + 1 + len(source_paths[right])
        for left, right in source.non_tree_edges
    )
    conjugator_path_letters = sum(
        len(source_paths[entry.simplex[0]]) for entry in source.triangle_relators
    )
    estimated_letters = (
        generator_path_letters
        + 3 * conjugator_path_letters
        + MAX_PRESENTATION_RELATOR_LETTERS * MAX_WORD
        + 3 * len(source.triangle_relators)
    )
    if estimated_letters > MAX_INDUCED_MAP_EDGE_LETTERS:
        raise OperationResourceAdmissionError(
            location=("map",),
            code="fundamental_group_map.edge_path_budget",
            message="induced presentation paths exceed the admitted aggregate work bound",
        )

    generator_images = tuple(
        FiniteGroupWord(
            letters=_edge_path_to_base_loop(
                edge, source_paths, vertex_map, target_edge_words
            )
        )
        for edge in source.non_tree_edges
    )

    target_relator_for_simplex = {
        item.simplex: index for index, item in enumerate(target.triangle_relators)
    }
    relator_images: list[PresentationRelatorImage] = []
    for index, source_triangle in enumerate(source.triangle_relators):
        mapped_triangle = tuple(
            vertex_map[vertex] for vertex in source_triangle.simplex
        )
        conjugator = FiniteGroupWord(
            letters=_map_edge_path(
                source_paths[source_triangle.simplex[0]],
                vertex_map,
                target_edge_words,
            )
        )
        if len(set(mapped_triangle)) < 3:
            relator_images.append(
                PresentationRelatorImage(
                    source_relator_index=index,
                    target_relator_index=None,
                    target_orientation=None,
                    conjugator=conjugator,
                )
            )
            continue
        target_simplex = tuple(sorted(mapped_triangle))
        target_index = target_relator_for_simplex.get(
            (target_simplex[0], target_simplex[1], target_simplex[2])
        )
        if target_index is None:
            raise OperationDomainValidationError(
                location=("map", "vertex_map"),
                code="fundamental_group_map.triangle_image_missing",
                message="a nondegenerate source triangle image is absent from the target presentation",
            )
        relator_images.append(
            PresentationRelatorImage(
                source_relator_index=index,
                target_relator_index=target_index,
                target_orientation=cast(
                    Literal[-1, 1], _permutation_sign(mapped_triangle)
                ),
                conjugator=conjugator,
            )
        )

    for source_relator, witness in zip(
        source.presentation.relators, relator_images, strict=True
    ):
        substituted: list[WordLetter] = []
        for letter in source_relator.letters:
            image = generator_images[letter.generator]
            substituted.extend(
                image.letters if letter.exponent == 1 else _inverse(image)
            )
        image_word = _reduce(substituted)
        if witness.target_relator_index is None:
            expected_word: tuple[WordLetter, ...] = ()
        else:
            target_relator = target.presentation.relators[witness.target_relator_index]
            relation_word = (
                target_relator.letters
                if witness.target_orientation == 1
                else _inverse(target_relator)
            )
            expected_word = _reduce(
                [
                    *witness.conjugator.letters,
                    *relation_word,
                    *_inverse(witness.conjugator),
                ]
            )
        if image_word != expected_word:
            raise OperationDomainValidationError(
                location=("map", "vertex_map"),
                code="fundamental_group_map.relator_replay_failed",
                message="a simplicial-map generator substitution failed exact triangle-relation replay",
            )

    target_generator_count = len(target.presentation.generators)
    abelianization_entries = tuple(
        tuple(
            sum(
                letter.exponent
                for letter in image.letters
                if letter.generator == target_generator
            )
            for image in generator_images
        )
        for target_generator in range(target_generator_count)
    )
    abelianization_map = IntegerMatrix(
        row_count=target_generator_count,
        column_count=len(generator_images),
        entries=abelianization_entries,
    )

    return FundamentalGroupMapResult._from_kernel(
        map=simplicial_map,
        source_presentation=source,
        target_presentation=target,
        generator_images=generator_images,
        abelianization_map=abelianization_map,
        relator_images=tuple(relator_images),
    )


__all__ = [
    "DirectRelatorMatchRequest",
    "DirectRelatorMatchResult",
    "FundamentalGroupMapRequest",
    "FundamentalGroupMapResult",
    "PresentationRelatorImage",
    "direct_relator_match",
    "induced_fundamental_group_map",
]
