"""Provider-independent values for exact finite stochastic processes."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_SAMPLES = 64


class FiniteProbabilitySpace(StrictModel):
    """An immutable finite probability space with positive-mass atoms.

    ``samples`` are unique labels. ``masses`` are positive rational strings
    that sum to exactly 1 (represented as strings for exactness).
    """

    samples: tuple[str, ...] = Field(min_length=1, max_length=MAX_SAMPLES)
    masses: tuple[str, ...] = Field(min_length=1, max_length=MAX_SAMPLES)

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.samples) != len(self.masses):
            raise ValueError("samples and masses must have equal length")
        if len(set(self.samples)) != len(self.samples):
            raise ValueError("sample labels must be unique")
        from fractions import Fraction

        total = Fraction(0)
        for m in self.masses:
            f = Fraction(m)
            if f <= 0:
                raise ValueError("masses must be positive")
            total += f
        if total != 1:
            raise ValueError("masses must sum to exactly 1")
        return self


class FiniteRandomVariable(StrictModel):
    """An immutable finite random variable on a probability space.

    ``values`` is a tuple of rational strings, one per sample, in the same
    order as the probability space's samples.
    """

    space: FiniteProbabilitySpace
    values: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_rv(self) -> Self:
        if len(self.values) != len(self.space.samples):
            raise ValueError("values must have one entry per sample")
        from fractions import Fraction

        for v in self.values:
            Fraction(v)
        return self


class FiniteSigmaAlgebra(StrictModel):
    """An immutable finite sigma algebra represented by its atom partition.

    ``blocks`` is a tuple of frozensets of sample labels. The blocks partition
    the sample space (disjoint, nonempty, union = all samples).
    """

    space: FiniteProbabilitySpace
    blocks: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_partition(self) -> Self:
        all_samples = set()
        seen: set[str] = set()
        for block in self.blocks:
            if not block:
                raise ValueError("partition blocks must be nonempty")
            for s in block:
                if s in seen:
                    raise ValueError("partition blocks must be disjoint")
                if s not in self.space.samples:
                    raise ValueError("block element not in sample space")
                seen.add(s)
                all_samples.add(s)
        if all_samples != set(self.space.samples):
            raise ValueError("blocks must partition the entire sample space")
        return self


__all__ = [
    "MAX_SAMPLES",
    "FiniteProbabilitySpace",
    "FiniteRandomVariable",
    "FiniteSigmaAlgebra",
]
