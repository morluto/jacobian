"""Exact bounded maps between finite presentation carriers."""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, cast

from pydantic import Field, ValidationError

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    MAX_INDUCED_MAP_EDGE_LETTERS,
    MAX_PRESENTATION_GENERATORS,
    MAX_PRESENTATION_RELATOR_LETTERS,
    MAX_WORD,
    EdgeWordEntry,
    FiniteGroupPresentation,
    FiniteGroupWord,
    FundamentalGroupBasepointChangeRequest,
    FundamentalGroupMapRequest,
    FundamentalGroupMapResult,
    FundamentalGroupPresentationResult,
    PresentationBasepointChangePath,
    PresentationMapCompositionRequest,
    PresentationMapCompositionResult,
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


def _permutation_sign(values: tuple[str, str, str]) -> Literal[-1, 1]:
    inversions = sum(
        values[left] > values[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def induced_fundamental_group_map(
    request: FundamentalGroupMapRequest,
) -> FundamentalGroupMapResult:
    """Construct the based generator map and replay triangle relations.

    Presentations are derived from the exact map's canonical complexes, so
    callers cannot substitute similarly shaped presentation data. The base
    vertex must map to the target base vertex; an unbased change-of-basepoint
    path is deliberately not inferred.
    """
    try:
        simplicial_map: SimplicialMap = SimplicialMap.model_validate(
            request.map.model_dump(mode="python")
        )
    except ValidationError as error:
        raise OperationDomainValidationError(
            location=("map",),
            code="fundamental_group_map.simplicial_map_invalid",
            message="the supplied map must be simplicial on its exact complexes",
        ) from error
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
        mapped_triangle = (
            vertex_map[source_triangle.simplex[0]],
            vertex_map[source_triangle.simplex[1]],
            vertex_map[source_triangle.simplex[2]],
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
        target_simplex = cast("tuple[str, str, str]", tuple(sorted(mapped_triangle)))
        target_index = target_relator_for_simplex.get(target_simplex)
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
                target_orientation=_permutation_sign(mapped_triangle),
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


MAX_BASEPOINT_TRANSPORT_WORK = 131_072


def change_fundamental_group_basepoint(
    request: FundamentalGroupBasepointChangeRequest,
) -> FundamentalGroupMapResult:
    """Transport generator words along an explicit path between basepoints.

    If ``p`` runs from source to target, each source loop ``a`` is sent to
    ``p^-1 a p``. The returned path-bound morphism is accepted by the existing
    exact presentation-map composition operation.
    """

    if type(request) is not FundamentalGroupBasepointChangeRequest or type(
        request.path
    ) is not PresentationBasepointChangePath:
        raise OperationDomainValidationError(
            location=("path",),
            code="fundamental_group_map.basepoint_path_type",
            message="request must contain a canonical based edge path",
        )
    try:
        path_value = PresentationBasepointChangePath.model_validate(
            request.path.model_dump(mode="python")
        )
    except ValidationError as error:
        raise OperationDomainValidationError(
            location=("path",),
            code="fundamental_group_map.basepoint_path_invalid",
            message="the supplied path must be a valid edge path on one exact complex",
        ) from error
    source = fundamental_group_presentation(
        path_value.complex, path_value.source_base_vertex
    )
    target = fundamental_group_presentation(
        path_value.complex, path_value.target_base_vertex
    )
    source_paths = _tree_paths(
        source.component_vertices, source.spanning_tree_edges, source.base_vertex
    )
    target_paths = _tree_paths(
        target.component_vertices, target.spanning_tree_edges, target.base_vertex
    )
    target_edge_words = {entry.edge: entry for entry in target.edge_words}
    identity_vertex_map = {vertex: vertex for vertex in path_value.complex.vertices}
    path_edges = tuple(
        zip(path_value.path_vertices, path_value.path_vertices[1:], strict=False)
    )
    inverse_path = tuple((right, left) for left, right in reversed(path_edges))

    generator_path_work = sum(
        len(source_paths[left]) + 1 + len(source_paths[right]) + 2 * len(path_edges)
        for left, right in source.non_tree_edges
    )
    conjugator_paths = tuple(
        len(path_edges)
        + len(source_paths[triangle.simplex[0]])
        + len(target_paths[triangle.simplex[0]])
        for triangle in source.triangle_relators
    )
    estimate = (
        generator_path_work
        + 2 * sum(conjugator_paths)
        + MAX_PRESENTATION_RELATOR_LETTERS * MAX_WORD
        + 3 * len(source.triangle_relators)
    )
    if estimate > MAX_BASEPOINT_TRANSPORT_WORK:
        raise OperationResourceAdmissionError(
            location=("path",),
            code="fundamental_group_map.basepoint_transport_work",
            message="basepoint transport and relation replay exceed the admitted work bound",
        )

    generator_images_list: list[FiniteGroupWord] = []
    for left, right in source.non_tree_edges:
        based_loop = source_paths[left] + ((left, right),) + tuple(
            (end, start) for start, end in reversed(source_paths[right])
        )
        transported = _map_edge_path(
            inverse_path + based_loop + path_edges,
            identity_vertex_map,
            target_edge_words,
        )
        if len(transported) > MAX_WORD:
            raise OperationResourceAdmissionError(
                location=("path",),
                code="fundamental_group_map.basepoint_word_output",
                message="a transported generator image exceeds the word output bound",
            )
        generator_images_list.append(FiniteGroupWord(letters=transported))
    generator_images = tuple(generator_images_list)

    target_relat_order = {
        triangle.simplex: index
        for index, triangle in enumerate(target.triangle_relators)
    }
    relator_images: list[PresentationRelatorImage] = []
    for index, triangle in enumerate(source.triangle_relators):
        vertex = triangle.simplex[0]
        conjugator_path = (
            inverse_path
            + source_paths[vertex]
            + tuple((end, start) for start, end in reversed(target_paths[vertex]))
        )
        conjugator_letters = _map_edge_path(
            conjugator_path, identity_vertex_map, target_edge_words
        )
        if len(conjugator_letters) > MAX_WORD:
            raise OperationResourceAdmissionError(
                location=("path",),
                code="fundamental_group_map.basepoint_conjugator_output",
                message="a relation conjugator exceeds the word output bound",
            )
        target_index = target_relat_order[triangle.simplex]
        relator_images.append(
            PresentationRelatorImage(
                source_relator_index=index,
                target_relator_index=target_index,
                target_orientation=1,
                conjugator=FiniteGroupWord(letters=conjugator_letters),
            )
        )

    _replay_generator_relation_images(
        source, target, generator_images, tuple(relator_images)
    )
    abelianization_map = _generator_word_matrix(generator_images, len(target.presentation.generators))
    return FundamentalGroupMapResult._from_kernel(
        map=path_value,
        source_presentation=source,
        target_presentation=target,
        generator_images=generator_images,
        abelianization_map=abelianization_map,
        relator_images=tuple(relator_images),
    )


def _replay_generator_relation_images(
    source: FundamentalGroupPresentationResult,
    target: FundamentalGroupPresentationResult,
    generator_images: tuple[FiniteGroupWord, ...],
    relator_images: tuple[PresentationRelatorImage, ...],
) -> None:
    for relator, witness in zip(
        source.presentation.relators, relator_images, strict=True
    ):
        substituted: list[WordLetter] = []
        for letter in relator.letters:
            image = generator_images[letter.generator]
            substituted.extend(
                image.letters if letter.exponent == 1 else _inverse(image)
            )
        target_index = witness.target_relator_index
        assert target_index is not None
        target_relator = target.presentation.relators[target_index]
        expected = _reduce(
            [
                *witness.conjugator.letters,
                *target_relator.letters,
                *_inverse(witness.conjugator),
            ]
        )
        if _reduce(substituted) != expected:
            raise OperationDomainValidationError(
                location=("path",),
                code="fundamental_group_map.basepoint_relation_replay",
                message="transported source relator disagrees with its target conjugacy witness",
            )


def _generator_word_matrix(
    generator_images: tuple[FiniteGroupWord, ...], target_generator_count: int
) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=target_generator_count,
        column_count=len(generator_images),
        entries=tuple(
            tuple(
                sum(
                    letter.exponent
                    for letter in image.letters
                    if letter.generator == target_generator
                )
                for image in generator_images
            )
            for target_generator in range(target_generator_count)
        ),
    )


MAX_COMPOSITION_SUBSTITUTION_LETTERS = 65_536
MAX_COMPOSITION_RELATOR_REPLAY_LETTERS = 65_536


def _inverse_word(word: FiniteGroupWord) -> tuple[WordLetter, ...]:
    return tuple(
        WordLetter(generator=letter.generator, exponent=-letter.exponent)
        for letter in reversed(word.letters)
    )


def _reduce_word_letters(letters: list[WordLetter]) -> tuple[WordLetter, ...]:
    reduced: list[WordLetter] = []
    for letter in letters:
        if (
            reduced
            and reduced[-1].generator == letter.generator
            and reduced[-1].exponent == -letter.exponent
        ):
            reduced.pop()
        else:
            reduced.append(letter)
    return tuple(reduced)


def _require_map_homomorphism(result: FundamentalGroupMapResult) -> None:
    source_relators = result.source_presentation.presentation.relators
    target_relators = result.target_presentation.presentation.relators
    if len(source_relators) != len(result.relator_images):
        raise OperationDomainValidationError(
            location=("relator_images",),
            code="fundamental_group_map.composition_relator_axis",
            message="each source relator needs a target relation witness",
        )
    for index, (relator, witness) in enumerate(
        zip(source_relators, result.relator_images, strict=True)
    ):
        if witness.source_relator_index != index:
            raise OperationDomainValidationError(
                location=("relator_images", index),
                code="fundamental_group_map.composition_relator_axis",
                message="relation witnesses must follow the source relator axis",
            )
        substituted: list[WordLetter] = []
        for letter in relator.letters:
            image = result.generator_images[letter.generator]
            substituted.extend(
                image.letters if letter.exponent == 1 else _inverse_word(image)
            )
        image_word = _reduce_word_letters(substituted)
        if witness.target_relator_index is None:
            expected: tuple[WordLetter, ...] = ()
        else:
            target_relator = target_relators[witness.target_relator_index]
            relation = (
                target_relator.letters
                if witness.target_orientation == 1
                else _inverse_word(target_relator)
            )
            expected = _reduce_word_letters(
                [
                    *witness.conjugator.letters,
                    *relation,
                    *_inverse_word(witness.conjugator),
                ]
            )
        if image_word != expected:
            raise OperationDomainValidationError(
                location=("relator_images", index),
                code="fundamental_group_map.composition_relator_replay",
                message="an input map does not preserve its source relation under the supplied witness",
            )

    target_count = len(result.target_presentation.presentation.generators)
    source_count = len(result.source_presentation.presentation.generators)
    expected_matrix = tuple(
        tuple(
            sum(letter.exponent for letter in image.letters if letter.generator == row)
            for image in result.generator_images
        )
        for row in range(target_count)
    )
    matrix = result.abelianization_map
    if (
        matrix.row_count != target_count
        or matrix.column_count != source_count
        or matrix.entries != expected_matrix
    ):
        raise OperationDomainValidationError(
            location=("abelianization_map",),
            code="fundamental_group_map.composition_matrix_claim",
            message="an input abelianization matrix disagrees with its generator words",
        )


def compose_fundamental_group_maps(
    request: PresentationMapCompositionRequest,
) -> PresentationMapCompositionResult:
    """Compose two based simplicial maps after applying the pi_1 construction."""
    first, second = request.first, request.second
    if first.target_presentation != second.source_presentation:
        raise OperationDomainValidationError(
            location=("second", "source_presentation"),
            code="fundamental_group_map.composition_carrier",
            message="the target presentation of the first map must equal the source presentation of the second",
        )

    source_count = len(first.source_presentation.presentation.generators)
    middle_count = len(first.target_presentation.presentation.generators)
    target_count = len(second.target_presentation.presentation.generators)
    if (
        len(first.generator_images) != source_count
        or len(second.generator_images) != middle_count
        or source_count > MAX_PRESENTATION_GENERATORS
        or middle_count > MAX_PRESENTATION_GENERATORS
        or target_count > MAX_PRESENTATION_GENERATORS
    ):
        raise OperationDomainValidationError(
            location=("first", "generator_images"),
            code="fundamental_group_map.composition_axes",
            message="map generator words must cover their complete presentation axes",
        )

    # This exact count bounds substitution before allocating result words.
    per_image_estimates = tuple(
        sum(
            len(second.generator_images[letter.generator].letters)
            for letter in image.letters
        )
        for image in first.generator_images
    )
    estimated_letters = sum(per_image_estimates)
    if estimated_letters > MAX_COMPOSITION_SUBSTITUTION_LETTERS:
        raise OperationResourceAdmissionError(
            location=("first", "generator_images"),
            code="fundamental_group_map.composition_work",
            message="composed generator-word expansion exceeds the admitted work bound",
        )
    if any(length > MAX_WORD for length in per_image_estimates):
        raise OperationResourceAdmissionError(
            location=("first", "generator_images"),
            code="fundamental_group_map.composition_word_output",
            message="a composed generator word exceeds the admitted output length",
        )

    def replay_work(result: FundamentalGroupMapResult) -> int:
        total = 0
        target_relators = result.target_presentation.presentation.relators
        for relator, witness in zip(
            result.source_presentation.presentation.relators,
            result.relator_images,
            strict=True,
        ):
            total += sum(
                len(result.generator_images[letter.generator].letters)
                for letter in relator.letters
            )
            if witness.target_relator_index is not None:
                total += 2 * len(witness.conjugator.letters) + len(
                    target_relators[witness.target_relator_index].letters
                )
        return total

    relation_work = replay_work(first) + replay_work(second)
    if relation_work > MAX_COMPOSITION_RELATOR_REPLAY_LETTERS:
        raise OperationResourceAdmissionError(
            location=("first", "relator_images"),
            code="fundamental_group_map.composition_relation_work",
            message="input relator replay exceeds the admitted work bound",
        )

    _require_map_homomorphism(first)
    _require_map_homomorphism(second)

    composed_images = []
    for image in first.generator_images:
        substituted: list[WordLetter] = []
        for letter in image.letters:
            target_word = second.generator_images[letter.generator]
            substituted.extend(
                target_word.letters
                if letter.exponent == 1
                else _inverse_word(target_word)
            )
        composed_images.append(
            FiniteGroupWord(letters=_reduce_word_letters(substituted))
        )

    left, right = first.abelianization_map, second.abelianization_map
    if (
        left.row_count != middle_count
        or left.column_count != source_count
        or right.row_count != target_count
        or right.column_count != middle_count
    ):
        raise OperationDomainValidationError(
            location=("first", "abelianization_map"),
            code="fundamental_group_map.composition_matrix_axes",
            message="abelianization maps must use the same intermediate generator axis",
        )
    entries = tuple(
        tuple(
            sum(
                right.entries[row][middle] * left.entries[middle][column]
                for middle in range(middle_count)
            )
            for column in range(source_count)
        )
        for row in range(target_count)
    )
    word_matrix = tuple(
        tuple(
            sum(letter.exponent for letter in image.letters if letter.generator == row)
            for image in composed_images
        )
        for row in range(target_count)
    )
    if entries != word_matrix:
        raise OperationDomainValidationError(
            location=("second", "abelianization_map"),
            code="fundamental_group_map.composition_matrix_replay",
            message="composed word exponent sums disagree with matrix composition",
        )
    return PresentationMapCompositionResult._from_kernel(
        source_presentation=first.source_presentation,
        intermediate_presentation=first.target_presentation,
        target_presentation=second.target_presentation,
        generator_images=tuple(composed_images),
        abelianization_map=IntegerMatrix(
            row_count=target_count,
            column_count=source_count,
            entries=entries,
        ),
    )


__all__ = [
    "DirectRelatorMatchRequest",
    "DirectRelatorMatchResult",
    "FundamentalGroupMapRequest",
    "FundamentalGroupMapResult",
    "PresentationMapCompositionRequest",
    "PresentationMapCompositionResult",
    "PresentationRelatorImage",
    "compose_fundamental_group_maps",
    "direct_relator_match",
    "induced_fundamental_group_map",
]
