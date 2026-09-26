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
MAX_COMPOSITION_WITNESS_LETTERS = 65_536


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


def _substitute_word(
    word: FiniteGroupWord,
    generator_images: tuple[FiniteGroupWord, ...],
) -> tuple[WordLetter, ...]:
    """Substitute generator images and freely reduce the resulting word."""

    substituted: list[WordLetter] = []
    for letter in word.letters:
        image = generator_images[letter.generator]
        substituted.extend(
            image.letters if letter.exponent == 1 else _inverse_word(image)
        )
    return _reduce_word_letters(substituted)


def _require_composition_operand_axes(
    result: FundamentalGroupMapResult,
    *,
    location: tuple[str | int, ...],
) -> None:
    """Admit every caller word and witness against its declared axes.

    The native callable is itself an admission boundary: a constructed
    request can carry generator words, source relators, or relation witnesses
    that name indices outside the presentation axes their carrier declares.
    Every index is checked here before ``compose_fundamental_group_maps``
    indexes a generator image or relator, so an out-of-axis word cannot leak a
    bare ``IndexError`` instead of a structured domain error.
    """

    source_relators = result.source_presentation.presentation.relators
    source_count = len(result.source_presentation.presentation.generators)
    target_relators = result.target_presentation.presentation.relators
    target_count = len(result.target_presentation.presentation.generators)
    if len(result.generator_images) != source_count:
        raise OperationDomainValidationError(
            location=(*location, "generator_images"),
            code="fundamental_group_map.composition_axes",
            message="map generator words must cover their complete presentation axis",
        )
    if any(
        letter.generator >= target_count
        for word in result.generator_images
        for letter in word.letters
    ):
        raise OperationDomainValidationError(
            location=(*location, "generator_images"),
            code="fundamental_group_map.composition_generator",
            message="a generator image names a generator outside the target axis",
        )
    if any(
        letter.generator >= source_count
        for relator in source_relators
        for letter in relator.letters
    ):
        raise OperationDomainValidationError(
            location=(*location, "source_presentation", "relators"),
            code="fundamental_group_map.composition_source_relator",
            message="a source relator names a generator outside the source axis",
        )
    if len(result.relator_images) != len(source_relators):
        raise OperationDomainValidationError(
            location=(*location, "relator_images"),
            code="fundamental_group_map.composition_relator_axis",
            message="each source relator needs one target relation witness",
        )
    for index, witness in enumerate(result.relator_images):
        if witness.source_relator_index != index:
            raise OperationDomainValidationError(
                location=(*location, "relator_images", index),
                code="fundamental_group_map.composition_relator_axis",
                message="relation witnesses must follow the source relator axis",
            )
        if (witness.target_relator_index is None) != (
            witness.target_orientation is None
        ):
            raise OperationDomainValidationError(
                location=(*location, "relator_images", index),
                code="fundamental_group_map.composition_relator_axis",
                message="a relation witness must bind index and orientation together",
            )
        if (
            witness.target_relator_index is not None
            and witness.target_relator_index >= len(target_relators)
        ):
            raise OperationDomainValidationError(
                location=(*location, "relator_images", index),
                code="fundamental_group_map.composition_relator_axis",
                message="a relation witness names a relator outside the target axis",
            )
        if any(
            letter.generator >= target_count for letter in witness.conjugator.letters
        ):
            raise OperationDomainValidationError(
                location=(*location, "relator_images", index, "conjugator"),
                code="fundamental_group_map.composition_conjugator",
                message="a relation conjugator names a generator outside the target axis",
            )


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


def _replay_work(result: FundamentalGroupMapResult) -> int:
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


def _identity_witness(index: int) -> PresentationRelatorImage:
    return PresentationRelatorImage(
        source_relator_index=index,
        target_relator_index=None,
        target_orientation=None,
        conjugator=FiniteGroupWord(letters=()),
    )


def _compose_orientation(first: int | None, second: int | None) -> Literal[-1, 1]:
    if first is None or second is None:
        raise OperationDomainValidationError(
            location=("relator_images",),
            code="fundamental_group_map.composition_relator_axis",
            message="a relation witness must bind index and orientation together",
        )
    return cast("Literal[-1, 1]", first * second)


def _compose_relator_images(
    first: FundamentalGroupMapResult,
    second: FundamentalGroupMapResult,
) -> tuple[PresentationRelatorImage, ...]:
    """Telescope the first map's relation witnesses through the second map.

    Substituting the second map into ``c1 * rel_M^o1 * c1^-1`` gives
    ``(C1 c2) * rel_N^(o1 o2) * (C1 c2)^-1`` with ``C1 = second(c1)``, so the
    composed witness reuses the second map's target relator and orientation.
    """

    witness_estimates = []
    for witness in first.relator_images:
        if witness.target_relator_index is None:
            witness_estimates.append(0)
            continue
        middle_witness = second.relator_images[witness.target_relator_index]
        if middle_witness.target_relator_index is None:
            witness_estimates.append(0)
            continue
        conjugator_growth = sum(
            len(second.generator_images[letter.generator].letters)
            for letter in witness.conjugator.letters
        )
        witness_estimates.append(
            conjugator_growth + len(middle_witness.conjugator.letters)
        )
    if sum(witness_estimates) > MAX_COMPOSITION_WITNESS_LETTERS:
        raise OperationResourceAdmissionError(
            location=("first", "relator_images"),
            code="fundamental_group_map.composition_witness_work",
            message="composed relation witnesses exceed the admitted work bound",
        )
    if any(length > MAX_WORD for length in witness_estimates):
        raise OperationResourceAdmissionError(
            location=("first", "relator_images"),
            code="fundamental_group_map.composition_witness_output",
            message="a composed relation conjugator exceeds the admitted output length",
        )

    composed: list[PresentationRelatorImage] = []
    for index, witness in enumerate(first.relator_images):
        if witness.target_relator_index is None:
            composed.append(_identity_witness(index))
            continue
        middle_witness = second.relator_images[witness.target_relator_index]
        if middle_witness.target_relator_index is None:
            composed.append(_identity_witness(index))
            continue
        conjugator = _reduce_word_letters(
            [
                *_substitute_word(witness.conjugator, second.generator_images),
                *middle_witness.conjugator.letters,
            ]
        )
        composed.append(
            PresentationRelatorImage(
                source_relator_index=index,
                target_relator_index=middle_witness.target_relator_index,
                target_orientation=_compose_orientation(
                    witness.target_orientation,
                    middle_witness.target_orientation,
                ),
                conjugator=FiniteGroupWord(letters=conjugator),
            )
        )
    return tuple(composed)


def _compose_simplicial_map(
    first: FundamentalGroupMapResult,
    second: FundamentalGroupMapResult,
) -> SimplicialMap:
    """Compose the two validated vertex maps into the canonical carrier map."""

    run_topology_admission(first.map.require_simplicial_map, location=("first", "map"))
    run_topology_admission(
        second.map.require_simplicial_map, location=("second", "map")
    )
    if (
        first.source_presentation.complex != first.map.source
        or first.target_presentation.complex != first.map.target
        or second.source_presentation.complex != second.map.source
        or second.target_presentation.complex != second.map.target
        or first.map.target != second.map.source
    ):
        raise OperationDomainValidationError(
            location=("first", "map"),
            code="fundamental_group_map.composition_complex",
            message="composition requires each presentation to bind its map and a common middle complex",
        )
    second_positions = {
        label: index for index, label in enumerate(second.map.source.vertices)
    }
    return SimplicialMap(
        source=first.map.source,
        target=second.map.target,
        vertex_map=tuple(
            second.map.vertex_map[second_positions[label]]
            for label in first.map.vertex_map
        ),
    )


def compose_fundamental_group_maps(
    request: PresentationMapCompositionRequest,
) -> FundamentalGroupMapResult:
    """Compose two based simplicial maps after applying the pi_1 construction.

    The result is the canonical :class:`FundamentalGroupMapResult` carrier for
    the composite map, so it can be supplied unchanged as either operand of a
    later composition. Its simplicial map is the composition of the two input
    maps and its relation witnesses are the input witnesses composed through the
    common middle presentation.
    """

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
        source_count > MAX_PRESENTATION_GENERATORS
        or middle_count > MAX_PRESENTATION_GENERATORS
        or target_count > MAX_PRESENTATION_GENERATORS
    ):
        raise OperationDomainValidationError(
            location=("first", "generator_images"),
            code="fundamental_group_map.composition_axes",
            message="map generator words must cover their complete presentation axes",
        )

    # Admit every caller word and witness axis before any substitution indexes
    # a generator image or relator in the work estimate or relation replay.
    _require_composition_operand_axes(first, location=("first",))
    _require_composition_operand_axes(second, location=("second",))

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

    relation_work = _replay_work(first) + _replay_work(second)
    if relation_work > MAX_COMPOSITION_RELATOR_REPLAY_LETTERS:
        raise OperationResourceAdmissionError(
            location=("first", "relator_images"),
            code="fundamental_group_map.composition_relation_work",
            message="input relator replay exceeds the admitted work bound",
        )

    _require_map_homomorphism(first)
    _require_map_homomorphism(second)

    composed_images = tuple(
        FiniteGroupWord(letters=_substitute_word(image, second.generator_images))
        for image in first.generator_images
    )
    composed_relator_images = _compose_relator_images(first, second)

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

    return FundamentalGroupMapResult._from_kernel(
        map=_compose_simplicial_map(first, second),
        source_presentation=first.source_presentation,
        target_presentation=second.target_presentation,
        generator_images=composed_images,
        abelianization_map=IntegerMatrix(
            row_count=target_count,
            column_count=source_count,
            entries=entries,
        ),
        relator_images=composed_relator_images,
    )


__all__ = [
    "DirectRelatorMatchRequest",
    "DirectRelatorMatchResult",
    "FundamentalGroupMapRequest",
    "FundamentalGroupMapResult",
    "PresentationMapCompositionRequest",
    "PresentationRelatorImage",
    "compose_fundamental_group_maps",
    "direct_relator_match",
    "induced_fundamental_group_map",
]
