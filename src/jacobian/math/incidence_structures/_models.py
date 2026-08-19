"""Typed wire contracts for incidence structure operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POINTS = 100
MAX_BLOCKS = 100


class IncidenceStructure(StrictModel):
    """A finite incidence structure: points and blocks."""

    points: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    block_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_BLOCKS)
    blocks: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_BLOCKS)

    @model_validator(mode="after")
    def require_valid_incidence(self) -> Self:
        if len(set(self.points)) != len(self.points):
            raise ValueError("point labels must be distinct")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block IDs must be distinct")
        if len(self.blocks) != len(self.block_ids):
            raise ValueError("blocks and block IDs must have same length")
        point_set = set(self.points)
        for block in self.blocks:
            if len(set(block)) != len(block):
                raise ValueError(
                    "duplicate point labels within a block are not allowed"
                )
            for p in block:
                if p not in point_set:
                    raise ValueError("every block member must be a declared point")
        return self


class IncidenceMatrixRequest(StrictModel):
    incidence: IncidenceStructure


class IncidenceMatrixResult(StrictModel):
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]


class DegreeProfileResult(StrictModel):
    """Per-point and per-block degree profiles."""

    point_degrees: tuple[tuple[str, int], ...]
    block_degrees: tuple[tuple[str, int], ...]
    total_incidences: int
