"""Backend-free private carriers for affine-torus worker projections."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class AffineTorusKernelSource:
    """Owner-private exact source decoded inside the bounded worker."""

    dimension: int
    linear_part: tuple[tuple[int, ...], ...]
    translation: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class EmptyFixedLocusKernel:
    character: tuple[int, ...]
    pairing: Fraction


@dataclass(frozen=True, slots=True)
class NonemptyFixedLocusKernel:
    base_point: tuple[Fraction, ...]
    identity_embedding: tuple[tuple[int, ...], ...]
    component_generators: tuple[tuple[Fraction, ...], ...]
    relation_matrix: tuple[tuple[int, ...], ...]
    generator_orders: tuple[int, ...]
    invariant_factors: tuple[int, ...]
    component_count: int


type FixedLocusKernel = EmptyFixedLocusKernel | NonemptyFixedLocusKernel


__all__ = [
    "AffineTorusKernelSource",
    "EmptyFixedLocusKernel",
    "FixedLocusKernel",
    "NonemptyFixedLocusKernel",
]
