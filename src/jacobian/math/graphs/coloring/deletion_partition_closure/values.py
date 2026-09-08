"""Canonical labelled vertex-deletion partition templates."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel

Label = Annotated[str, StringConstraints(strict=True, max_length=64)]
Block = Annotated[tuple[Label, ...], Field(min_length=1, max_length=255)]


class DeletionPartitionRow(StrictModel):
    """Unordered nonempty blocks partitioning the carrier except one vertex."""

    deleted_vertex: Label
    blocks: tuple[Block, ...] = Field(max_length=255)


class DeletionPartitionTemplate(StrictModel):
    """One partition of V minus v for each v in the ordered labelled carrier.

    Members and blocks are canonicalized by carrier position, and rows by
    deleted-vertex position. The positive block bound is retained exactly.
    Parsing checks partition structure only, without constructing a graph.
    """

    vertices: tuple[Label, ...] = Field(max_length=256)
    block_bound: ExactInteger = Field(ge=1)
    rows: tuple[DeletionPartitionRow, ...] = Field(max_length=256)

    @model_validator(mode="after")
    def canonicalize_partitions(self) -> Self:
        if self.block_bound < 1:
            raise ValueError("block bound must be positive")
        positions = {vertex: index for index, vertex in enumerate(self.vertices)}
        if len(positions) != len(self.vertices):
            raise ValueError("carrier labels must be distinct")
        for vertex in self.vertices:
            if unicodedata.normalize("NFC", vertex) != vertex:
                raise ValueError("carrier labels must be NFC normalized")
            try:
                encoded = vertex.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("carrier labels must be Unicode scalar text") from exc
            if len(encoded) > 64:
                raise ValueError("carrier labels must have at most 64 UTF-8 bytes")
        if len(self.rows) != len(self.vertices):
            raise ValueError("one deletion row is required per carrier vertex")
        canonical: dict[str, DeletionPartitionRow] = {}
        for row in self.rows:
            if row.deleted_vertex not in positions or row.deleted_vertex in canonical:
                raise ValueError("deleted vertices must enumerate the carrier once")
            if len(row.blocks) > self.block_bound:
                raise ValueError("partition exceeds its block bound")
            members = [vertex for block in row.blocks for vertex in block]
            if len(members) != len(set(members)) or set(members) != (
                set(positions) - {row.deleted_vertex}
            ):
                raise ValueError("row blocks must partition exactly V minus its vertex")
            blocks = tuple(
                sorted(
                    (
                        tuple(sorted(block, key=positions.__getitem__))
                        for block in row.blocks
                    ),
                    key=lambda block: positions[block[0]],
                )
            )
            canonical[row.deleted_vertex] = DeletionPartitionRow(
                deleted_vertex=row.deleted_vertex, blocks=blocks
            )
        object.__setattr__(self, "rows", tuple(canonical[v] for v in self.vertices))
        return self
