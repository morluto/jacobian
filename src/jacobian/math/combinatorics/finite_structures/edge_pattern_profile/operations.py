"""Hypergraph vertex-colouring edge pattern profile kernel."""

from __future__ import annotations

import unicodedata

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile._models import (
    EdgePatternEntry,
    EdgePatternProfileResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["compute_edge_pattern_profile"]

MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _strict_json_array_size(item_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _encoded_size(value: str, encoded: dict[str, int]) -> int:
    size = encoded.get(value)
    if size is None:
        size = len(encode_strict_json(value))
        encoded[value] = size
    return size


def _entry_size(
    edge_id: str,
    members: tuple[str, ...],
    colors: tuple[str, ...],
    encoded: dict[str, int],
) -> int:
    color_to_block: dict[str, int] = {}
    partitions: list[int] = []
    labels: list[str] = []
    for color in colors:
        if color not in color_to_block:
            color_to_block[color] = len(color_to_block)
            labels.append(color)
        partitions.append(color_to_block[color])
    return strict_json_object_size(
        (
            ("edge_id", _encoded_size(edge_id, encoded)),
            (
                "members",
                _strict_json_array_size(
                    tuple(_encoded_size(member, encoded) for member in members)
                ),
            ),
            (
                "equality_partition",
                _strict_json_array_size(
                    tuple(len(encode_strict_json(index)) for index in partitions)
                ),
            ),
            ("num_color_blocks", len(encode_strict_json(len(labels)))),
            (
                "color_labels",
                _strict_json_array_size(
                    tuple(_encoded_size(color, encoded) for color in labels)
                ),
            ),
        )
    )


def _admit_edge_pattern_profile(
    hypergraph: FiniteHypergraph, vertex_colors: dict[str, str]
) -> None:
    if not isinstance(hypergraph, FiniteHypergraph):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="edge_pattern.invalid_hypergraph",
            message="hypergraph must be a FiniteHypergraph value",
        )
    if not isinstance(vertex_colors, dict) or set(vertex_colors) != set(
        hypergraph.vertices
    ):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.color_map_must_cover_all_vertices",
            message="vertex_colors must cover exactly all declared vertices",
        )
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in vertex_colors.items()
    ):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.invalid_color_map",
            message="vertex_colors must be a string-to-string mapping",
        )
    normalized_keys = [unicodedata.normalize("NFC", key) for key in vertex_colors]
    if len(set(normalized_keys)) != len(normalized_keys):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.color_map_key_collision",
            message="vertex_colors keys collide after Unicode normalization",
        )
    try:
        encoded: dict[str, int] = {}
        source_size = len(encode_strict_json(hypergraph.model_dump(mode="json")))
        colors_size = len(encode_strict_json(vertex_colors))
        entry_sizes: list[int] = []
        monochromatic_sizes: list[int] = []
        rainbow_sizes: list[int] = []
        result_bytes = strict_json_object_size(
            (
                ("hypergraph", source_size),
                ("vertex_colors", colors_size),
                ("entries", _strict_json_array_size(())),
                ("monochromatic_edge_ids", _strict_json_array_size(())),
                ("rainbow_edge_ids", _strict_json_array_size(())),
            )
        )
        for edge_id, members in hypergraph.edges:
            colors = tuple(
                unicodedata.normalize("NFC", vertex_colors[member])
                for member in members
            )
            entry_sizes.append(_entry_size(edge_id, members, colors, encoded))
            edge_id_size = _encoded_size(edge_id, encoded)
            blocks = len(set(colors))
            if blocks == 1:
                monochromatic_sizes.append(edge_id_size)
            if blocks == len(members):
                rainbow_sizes.append(edge_id_size)
            result_bytes = strict_json_object_size(
                (
                    ("hypergraph", source_size),
                    ("vertex_colors", colors_size),
                    ("entries", _strict_json_array_size(tuple(entry_sizes))),
                    (
                        "monochromatic_edge_ids",
                        _strict_json_array_size(tuple(monochromatic_sizes)),
                    ),
                    ("rainbow_edge_ids", _strict_json_array_size(tuple(rainbow_sizes))),
                )
            )
            if result_bytes > MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES:
                raise OperationDomainValidationError(
                    location=("hypergraph", "vertex_colors"),
                    code="edge_pattern.result_too_large",
                    message="the complete edge-pattern profile exceeds the canonical output envelope",
                )
    except (CanonicalizationError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("hypergraph", "vertex_colors"),
            code="edge_pattern.result_too_large",
            message="the complete edge-pattern profile exceeds the canonical output envelope",
        ) from error
    if result_bytes > MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("hypergraph", "vertex_colors"),
            code="edge_pattern.result_too_large",
            message=(
                "the complete edge-pattern profile exceeds the "
                f"{MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES}-byte result envelope"
            ),
        )


def compute_edge_pattern_profile(
    hypergraph: FiniteHypergraph,
    vertex_colors: dict[str, str],
) -> EdgePatternProfileResult:
    """Return the complete edge-pattern profile of a vertex-coloured hypergraph.

    For each edge, compute the equality partition of its member colours,
    the number of colour blocks, and classify as monochromatic or rainbow.
    """
    _admit_edge_pattern_profile(hypergraph, vertex_colors)
    entries: list[EdgePatternEntry] = []
    monochromatic: list[str] = []
    rainbow: list[str] = []

    for edge_id, members in hypergraph.edges:
        colors = [vertex_colors[m] for m in members]
        color_to_block: dict[str, int] = {}
        equality_partition: list[int] = []
        color_labels: list[str] = []
        for c in colors:
            if c not in color_to_block:
                color_to_block[c] = len(color_to_block)
                color_labels.append(c)
            equality_partition.append(color_to_block[c])

        num_blocks = len(color_to_block)
        entries.append(
            EdgePatternEntry(
                edge_id=edge_id,
                members=members,
                equality_partition=tuple(equality_partition),
                num_color_blocks=num_blocks,
                color_labels=tuple(color_labels),
            )
        )
        if num_blocks == 1:
            monochromatic.append(edge_id)
        if num_blocks == len(members):
            rainbow.append(edge_id)

    return EdgePatternProfileResult(
        hypergraph=hypergraph,
        vertex_colors=vertex_colors,
        entries=tuple(entries),
        monochromatic_edge_ids=tuple(monochromatic),
        rainbow_edge_ids=tuple(rainbow),
    )
