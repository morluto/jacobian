"""Hypergraph vertex-colouring edge pattern profile kernel."""

from __future__ import annotations

import unicodedata

import rfc8785

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
    VertexColorPair,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = [
    "EdgePatternEntry",
    "EdgePatternProfileResult",
    "VertexColorPair",
    "compute_edge_pattern_profile",
]

MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _strict_json_array_size(item_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _encoded_size(value: str, encoded: dict[str, int]) -> int:
    size = encoded.get(value)
    if size is None:
        # Size with the same non-normalizing strict JSON used by direct
        # delivery (encode_strict_json on model_dump), not the normalized
        # canonical form, so decomposed labels cannot be undercounted.
        size = len(rfc8785.dumps(value))
        encoded[value] = size
    return size


def _compute_edge_admission(
    edge_id: str,
    members: tuple[str, ...],
    colors: tuple[str, ...],
    encoded: dict[str, int],
) -> tuple[int, tuple[int, ...], int, tuple[str, ...], bool, bool]:
    """Compute one edge's equality partition and classification for admission.

    Returns ``(entry_size, equality_partition, num_blocks, color_labels,
    is_monochromatic, is_rainbow)``.
    """
    color_to_block: dict[str, int] = {}
    equality_partition: list[int] = []
    color_labels: list[str] = []
    for color in colors:
        if color not in color_to_block:
            color_to_block[color] = len(color_to_block)
            color_labels.append(color)
        equality_partition.append(color_to_block[color])

    num_blocks = len(color_to_block)
    is_mono = num_blocks <= 1
    is_rainbow = num_blocks == len(members)

    # Compute the entry size using the precomputed partition data
    entry_size = strict_json_object_size(
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
                    tuple(
                        len(encode_strict_json(index)) for index in equality_partition
                    )
                ),
            ),
            ("num_color_blocks", len(encode_strict_json(num_blocks))),
            (
                "color_labels",
                _strict_json_array_size(
                    tuple(_encoded_size(color, encoded) for color in color_labels)
                ),
            ),
        )
    )

    return (
        entry_size,
        tuple(equality_partition),
        num_blocks,
        tuple(color_labels),
        is_mono,
        is_rainbow,
    )


def _admit_edge_pattern_profile(
    hypergraph: FiniteHypergraph, vertex_colors: tuple[VertexColorPair, ...]
) -> tuple[
    tuple[VertexColorPair, ...],
    list[
        tuple[str, tuple[str, ...], tuple[int, ...], int, tuple[str, ...], bool, bool]
    ],
]:
    """Admit the request and return the precomputed admission plan.

    Returns ``(color_rows, edge_plans)`` where ``color_rows`` is the retained
    colour carrier and ``edge_plans`` is a list of per-edge tuples containing
    ``(edge_id, members, equality_partition, num_blocks, color_labels,
    is_monochromatic, is_rainbow)`` that the kernel reuses directly without
    recomputing the equality partition.
    """
    if not isinstance(hypergraph, FiniteHypergraph):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="edge_pattern.invalid_hypergraph",
            message="hypergraph must be a FiniteHypergraph value",
        )
    if not isinstance(vertex_colors, tuple) or any(
        not isinstance(pair, VertexColorPair)
        or not isinstance(pair.vertex, str)
        or not isinstance(pair.color, str)
        for pair in vertex_colors
    ):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.invalid_color_map",
            message="vertex_colors must be a sequence of VertexColorPair rows",
        )
    vertices = set(hypergraph.vertices)
    row_vertices = [pair.vertex for pair in vertex_colors]
    if len(set(row_vertices)) != len(row_vertices):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.duplicate_vertex_color",
            message="vertex_colors must not repeat a vertex",
        )
    if set(row_vertices) != vertices:
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.color_map_must_cover_all_vertices",
            message="vertex_colors must cover exactly all declared vertices",
        )
    # Bounded aggregate color-label check before any encoding/allocation:
    # every color label is echoed verbatim into the result. First reject any
    # request whose aggregate character count already exceeds the byte
    # envelope (each Unicode character encodes to at least one UTF-8 byte),
    # then encode only the bounded remainder to compute the exact byte count.
    total_chars = sum(len(pair.color) for pair in vertex_colors)
    if total_chars > MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.result_too_large",
            message=(
                "the complete edge-pattern profile exceeds the canonical "
                "output envelope"
            ),
        )
    try:
        total_color_bytes = sum(
            len(pair.color.encode("utf-8")) for pair in vertex_colors
        )
    except UnicodeEncodeError as exc:
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.invalid_color_encoding",
            message="vertex color labels must be valid UTF-8",
        ) from exc
    if total_color_bytes > MAX_EDGE_PATTERN_PROFILE_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.result_too_large",
            message=(
                "the complete edge-pattern profile exceeds the canonical "
                "output envelope"
            ),
        )
    # Retain exact source-label identity: lookups are keyed by the exact vertex
    # label, while only the color *values* are NFC-normalized for a canonical
    # result. NFC-colliding vertex labels can therefore both be represented.
    color_by_vertex = {pair.vertex: pair.color for pair in vertex_colors}
    normalized_colors = {
        vertex: unicodedata.normalize("NFC", color)
        for vertex, color in color_by_vertex.items()
    }

    # Thread 1: Compute equality partitions during admission and return them as
    # part of the plan so the kernel reuses them without recomputing the same
    # semantic work. Color labels are measured after transport normalization.
    try:
        encoded: dict[str, int] = {}
        color_rows = tuple(
            VertexColorPair(vertex=v, color=normalized_colors[v])
            for v in sorted(normalized_colors)
        )
        source_size = len(encode_strict_json(hypergraph.model_dump(mode="json")))
        colors_size = len(
            encode_strict_json([p.model_dump(mode="json") for p in color_rows])
        )
        entries_bytes = monochromatic_bytes = rainbow_bytes = 2
        monochromatic_count = rainbow_count = 0
        result_bytes = strict_json_object_size(
            (
                ("hypergraph", source_size),
                ("vertex_colors", colors_size),
                ("entries", entries_bytes),
                ("monochromatic_edge_ids", monochromatic_bytes),
                ("rainbow_edge_ids", rainbow_bytes),
            )
        )
        edge_plans: list[
            tuple[
                str, tuple[str, ...], tuple[int, ...], int, tuple[str, ...], bool, bool
            ]
        ] = []
        for entries_count, (edge_id, members) in enumerate(hypergraph.edges):
            colors = tuple(normalized_colors[member] for member in members)
            (
                entry_size,
                equality_partition,
                num_blocks,
                color_labels,
                is_mono,
                is_rainbow,
            ) = _compute_edge_admission(edge_id, members, colors, encoded)
            edge_plans.append(
                (
                    edge_id,
                    members,
                    equality_partition,
                    num_blocks,
                    color_labels,
                    is_mono,
                    is_rainbow,
                )
            )
            entries_bytes += entry_size + (1 if entries_count else 0)
            edge_id_size = _encoded_size(edge_id, encoded)
            if is_mono:
                monochromatic_bytes += edge_id_size + (1 if monochromatic_count else 0)
                monochromatic_count += 1
            if is_rainbow:
                rainbow_bytes += edge_id_size + (1 if rainbow_count else 0)
                rainbow_count += 1
            result_bytes = strict_json_object_size(
                (
                    ("hypergraph", source_size),
                    ("vertex_colors", colors_size),
                    ("entries", entries_bytes),
                    (
                        "monochromatic_edge_ids",
                        monochromatic_bytes,
                    ),
                    ("rainbow_edge_ids", rainbow_bytes),
                ),
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
    return color_rows, edge_plans


def compute_edge_pattern_profile(
    hypergraph: FiniteHypergraph,
    vertex_colors: tuple[VertexColorPair, ...],
) -> EdgePatternProfileResult:
    """Return the complete edge-pattern profile of a vertex-coloured hypergraph.

    For each edge, compute the equality partition of its member colours,
    the number of colour blocks, and classify as monochromatic or rainbow.
    """
    color_rows, edge_plans = _admit_edge_pattern_profile(hypergraph, vertex_colors)

    # Thread 1: reuse the precomputed equality partitions from admission
    # instead of recomputing them in the kernel.
    entries: list[EdgePatternEntry] = []
    monochromatic: list[str] = []
    rainbow: list[str] = []

    for (
        edge_id,
        members,
        equality_partition,
        num_blocks,
        color_labels,
        is_mono,
        is_rainbow,
    ) in edge_plans:
        entries.append(
            EdgePatternEntry(
                edge_id=edge_id,
                members=members,
                equality_partition=equality_partition,
                num_color_blocks=num_blocks,
                color_labels=color_labels,
            )
        )
        if is_mono:
            monochromatic.append(edge_id)
        if is_rainbow:
            rainbow.append(edge_id)

    # Thread 2: the coloring is a sequence of vertex-color pairs, so it cannot
    # be confused with the repository's rational encoding during transport.
    return EdgePatternProfileResult(
        hypergraph=hypergraph,
        vertex_colors=color_rows,
        entries=tuple(entries),
        monochromatic_edge_ids=tuple(monochromatic),
        rainbow_edge_ids=tuple(rainbow),
    )
